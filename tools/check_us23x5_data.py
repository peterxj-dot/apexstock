"""Diagnostic auditor for US 23/5 trading hours launch (Dec 6, 2026).

Checks yfinance daily OHLCV schema, premature overnight/intraday bar leakage,
regular-vs-extended session OHLCV structure, and local ApexStock CSV integrity.
Strictly read-only on data/*.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.daily_report_helper import _md_to_telegram_text
from tools.build_soxl_zones import send_tg_notification

DATA_DIR = ROOT / "data"
CHECKS_DIR = ROOT / "logs" / "us23x5_checks"
TICKERS = ["SOXL", "SMH", "MSTR", "QQQ", "BRK-B"]
EXPECTED_COLS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


def _market_phase_et(now_et: datetime) -> str:
    wd = now_et.weekday()  # 0=Mon .. 6=Sun
    t = now_et.time()
    if wd == 5:
        return "WEEKEND"
    if wd == 6:
        return "OVERNIGHT_2100_0400_ET" if t >= time(21, 0) else "WEEKEND"
    if wd == 4 and t >= time(20, 0):
        return "WEEKEND"
    if t < time(4, 0) or t >= time(21, 0):
        return "OVERNIGHT_2100_0400_ET"
    if t < time(9, 30):
        return "PREMARKET_0400_0930_ET"
    if t < time(16, 0):
        return "REGULAR_0930_1600_ET"
    if t < time(20, 0):
        return "AFTERHOURS_1600_2000_ET"
    return "MAINTENANCE_2000_2100_ET"


def _trade_date_for_et_ts(ts: pd.Timestamp) -> str:
    """Per SIP 23/5 rule: 21:00-24:00 ET belongs to the next calendar trade date."""
    if ts.time() >= time(21, 0):
        return (ts.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    return ts.date().strftime("%Y-%m-%d")


def _session_bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    if t >= time(21, 0) or t < time(4, 0):
        return "overnight"
    if t < time(9, 30):
        return "premarket"
    if t < time(16, 0):
        return "regular"
    if t < time(20, 0):
        return "afterhours"
    return "maintenance"


def _safe_float(v: object, ndigits: int = 4) -> float | None:
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return round(f, ndigits)
    except Exception:
        return None


def _agg_ohlcv(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty:
        return None
    return {
        "bars": int(len(df)),
        "first_ts_et": df.index[0].strftime("%Y-%m-%d %H:%M"),
        "last_ts_et": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "open": _safe_float(df["Open"].iloc[0]),
        "high": _safe_float(df["High"].max()),
        "low": _safe_float(df["Low"].min()),
        "close": _safe_float(df["Close"].iloc[-1]),
        "volume": int(pd.to_numeric(df["Volume"], errors="coerce").fillna(0).sum()),
    }


def _pct_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return round((a - b) / abs(b) * 100.0, 4)


def _inspect_ticker(ticker: str, now_et: datetime) -> dict:
    et_today_str = now_et.date().strftime("%Y-%m-%d")
    et_next_str = (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    before_regular_close = now_et.time() < time(16, 0)

    # 1) Daily bars fetch (mirroring tools/download_data.py exactly)
    start_str = (now_et.date() - timedelta(days=14)).strftime("%Y-%m-%d")
    end_str = (now_et.date() + timedelta(days=2)).strftime("%Y-%m-%d")
    raw_daily = yf.download(
        ticker,
        start=start_str,
        end=end_str,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    schema_info: dict = {
        "empty": bool(raw_daily is None or raw_daily.empty),
        "is_multiindex_cols": bool(isinstance(raw_daily.columns, pd.MultiIndex)) if raw_daily is not None else False,
        "raw_columns": [str(c) for c in raw_daily.columns] if raw_daily is not None else [],
        "index_name": str(raw_daily.index.name) if raw_daily is not None else None,
        "index_tz": str(getattr(raw_daily.index, "tz", None)) if raw_daily is not None else None,
    }

    daily_rows: list[dict] = []
    premature_bar_dates: list[str] = []
    nan_rows = 0
    duplicate_dates = 0
    adj_ratio_outliers: list[dict] = []

    if raw_daily is not None and not raw_daily.empty:
        df = raw_daily.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        schema_info["missing_expected_cols"] = [c for c in EXPECTED_COLS if c not in df.columns]
        schema_info["extra_cols"] = [c for c in df.columns if c not in EXPECTED_COLS]
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
        df["DateStr"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        nan_rows = int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
        duplicate_dates = int(df["DateStr"].duplicated().sum())

        for _, r in df.tail(6).iterrows():
            d_str = str(r["DateStr"])
            o = _safe_float(r.get("Open"))
            h = _safe_float(r.get("High"))
            lo = _safe_float(r.get("Low"))
            c = _safe_float(r.get("Close"))
            ac = _safe_float(r.get("Adj Close"))
            vol = int(r.get("Volume")) if pd.notna(r.get("Volume")) else None
            daily_rows.append({
                "date": d_str,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "adj_close": ac,
                "volume": vol,
            })
            if c and ac and abs(ac / c - 1.0) > 0.05:
                adj_ratio_outliers.append({"date": d_str, "adj_close_div_close": round(ac / c, 4)})
            if (d_str == et_today_str and before_regular_close) or (d_str >= et_next_str):
                premature_bar_dates.append(d_str)

    # 2) Intraday 5m prepost=True comparison
    intraday_raw = yf.download(
        ticker,
        period="5d",
        interval="5m",
        prepost=True,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    intraday_summary: dict = {"empty": True}
    structural_comparisons: list[dict] = []

    if intraday_raw is not None and not intraday_raw.empty:
        idf = intraday_raw.copy()
        if isinstance(idf.columns, pd.MultiIndex):
            idf.columns = idf.columns.get_level_values(0)
        if idf.index.tz is None:
            idf.index = idf.index.tz_localize("UTC")
        idf.index = idf.index.tz_convert("America/New_York")
        idf["trade_date"] = [_trade_date_for_et_ts(ts) for ts in idf.index]
        idf["session"] = [_session_bucket(ts) for ts in idf.index]

        session_counts = {k: int(v) for k, v in idf["session"].value_counts().items()}
        overnight_df = idf[idf["session"] == "overnight"]
        intraday_summary = {
            "empty": False,
            "total_5m_bars": int(len(idf)),
            "session_bar_counts": session_counts,
            "has_overnight_2100_0400_bars": bool(len(overnight_df) > 0),
            "latest_5m_ts_et": idf.index[-1].strftime("%Y-%m-%d %H:%M %Z"),
            "overnight_dates_seen": sorted(overnight_df["trade_date"].unique().tolist()),
        }

        daily_by_date = {row["date"]: row for row in daily_rows}
        for td in sorted(idf["trade_date"].unique().tolist())[-3:]:
            sub = idf[idf["trade_date"] == td]
            reg = sub[sub["session"] == "regular"]
            ovn = sub[sub["session"] == "overnight"]
            reg_agg = _agg_ohlcv(reg)
            all_agg = _agg_ohlcv(sub)
            ovn_agg = _agg_ohlcv(ovn)
            d_bar = daily_by_date.get(td)

            comp: dict = {
                "trade_date": td,
                "daily_bar": d_bar,
                "regular_5m": reg_agg,
                "all_sessions_5m": all_agg,
                "overnight_5m": ovn_agg,
            }
            if d_bar and reg_agg and all_agg:
                dh_vs_reg = _pct_diff(d_bar["high"], reg_agg["high"])
                dl_vs_reg = _pct_diff(d_bar["low"], reg_agg["low"])
                dc_vs_reg = _pct_diff(d_bar["close"], reg_agg["close"])
                do_vs_reg = _pct_diff(d_bar["open"], reg_agg["open"])
                dh_vs_all = _pct_diff(d_bar["high"], all_agg["high"])
                dl_vs_all = _pct_diff(d_bar["low"], all_agg["low"])

                # Detect if daily High/Low actually absorbed extended/overnight extremes
                # (must be significantly beyond regular AND closer to extended extreme than regular)
                extended_hl_leak = False
                if (
                    all_agg["high"] is not None
                    and reg_agg["high"] is not None
                    and d_bar["high"] is not None
                    and all_agg["high"] > reg_agg["high"] * 1.003
                    and d_bar["high"] > reg_agg["high"] * 1.002
                    and abs(d_bar["high"] - all_agg["high"]) < abs(d_bar["high"] - reg_agg["high"])
                ):
                    extended_hl_leak = True
                if (
                    all_agg["low"] is not None
                    and reg_agg["low"] is not None
                    and d_bar["low"] is not None
                    and all_agg["low"] < reg_agg["low"] * 0.997
                    and d_bar["low"] < reg_agg["low"] * 0.998
                    and abs(d_bar["low"] - all_agg["low"]) < abs(d_bar["low"] - reg_agg["low"])
                ):
                    extended_hl_leak = True

                comp["diffs_vs_regular_pct"] = {
                    "open": do_vs_reg,
                    "high": dh_vs_reg,
                    "low": dl_vs_reg,
                    "close": dc_vs_reg,
                }
                comp["diffs_vs_all_pct"] = {
                    "high": dh_vs_all,
                    "low": dl_vs_all,
                }
                if td in premature_bar_dates:
                    comp["verdict"] = "PREMATURE_UNCLOSED_BAR"
                elif extended_hl_leak:
                    comp["verdict"] = "EXTENDED_OR_OVERNIGHT_LEAK_IN_DAILY_HL"
                else:
                    comp["verdict"] = "MATCHES_REGULAR_SESSION"
            elif d_bar and not reg_agg:
                comp["verdict"] = "PREMATURE_OVERNIGHT_ONLY_DAILY_BAR"
            elif not d_bar and ovn_agg:
                comp["verdict"] = "CLEAN_NO_PREMATURE_DAILY_BAR"
            structural_comparisons.append(comp)

    # 3) Local CSV integrity check (read-only)
    csv_path = DATA_DIR / f"{ticker}.csv"
    local_info: dict = {"exists": csv_path.exists()}
    if csv_path.exists():
        ldf = pd.read_csv(csv_path)
        local_info.update({
            "columns": list(ldf.columns),
            "rows": int(len(ldf)),
            "nan_rows": int(ldf[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()),
            "duplicate_dates": int(ldf["Date"].duplicated().sum()) if "Date" in ldf.columns else None,
            "last_3_rows": ldf.tail(3).to_dict(orient="records"),
        })
        if daily_rows and not ldf.empty:
            last_local_date = str(ldf["Date"].iloc[-1])
            matched = [r for r in daily_rows if r["date"] == last_local_date]
            if matched:
                m = matched[0]
                local_close = _safe_float(ldf["Close"].iloc[-1])
                local_info["latest_vs_yf_close_diff_pct"] = _pct_diff(local_close, m["close"])

    return {
        "ticker": ticker,
        "schema": schema_info,
        "nan_rows": nan_rows,
        "duplicate_dates": duplicate_dates,
        "adj_ratio_outliers": adj_ratio_outliers,
        "premature_bar_dates": premature_bar_dates,
        "daily_last_6": daily_rows,
        "intraday_5m_prepost": intraday_summary,
        "structural_comparisons": structural_comparisons,
        "local_csv": local_info,
    }


def cmd_inspect() -> None:
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    now_et = now_jst.astimezone(ZoneInfo("America/New_York"))
    now_utc = now_jst.astimezone(ZoneInfo("UTC"))

    last_cron_file = ROOT / "logs" / "last_daily_btc_run_utc_date.txt"
    last_cron_date = last_cron_file.read_text(encoding="utf-8").strip() if last_cron_file.exists() else None

    results = []
    for t in TICKERS:
        try:
            results.append(_inspect_ticker(t, now_et))
        except Exception as exc:
            results.append({"ticker": t, "error": str(exc)})

    payload = {
        "generated_at_jst": now_jst.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "generated_at_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "report_date_jst": now_jst.strftime("%Y-%m-%d"),
        "market_phase_et": _market_phase_et(now_et),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "last_daily_update_utc_date": last_cron_date,
        "tickers": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_notify(md_path: Path | None = None) -> None:
    if md_path is None:
        files = sorted(CHECKS_DIR.glob("*.md")) if CHECKS_DIR.exists() else []
        if not files:
            raise FileNotFoundError(f"No Markdown check report found in {CHECKS_DIR}")
        md_path = files[-1]
    if not md_path.exists():
        raise FileNotFoundError(f"Report file not found: {md_path}")

    raw = md_path.read_text(encoding="utf-8")
    msg = _md_to_telegram_text(raw)

    max_len = 3600
    chunks: list[str] = []
    cur = ""
    for para in msg.split("\n\n"):
        cand = f"{cur}\n\n{para}" if cur else para
        if len(cand) <= max_len:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            if len(para) <= max_len:
                cur = para
            else:
                for i in range(0, len(para), max_len):
                    chunks.append(para[i : i + max_len])
                cur = ""
    if cur:
        chunks.append(cur)

    for idx, chunk in enumerate(chunks, 1):
        prefix = f"({idx}/{len(chunks)})\n" if len(chunks) > 1 else ""
        ok = send_tg_notification(prefix + chunk)
        if not ok:
            raise RuntimeError(f"Failed sending Telegram chunk {idx}/{len(chunks)}")
    print(f"[Telegram] Sent {len(chunks)} chunk(s) from {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit US 23/5 trading hours impact on yfinance daily data.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="Run read-only yfinance & local CSV structural audit and output JSON.")
    p_not = sub.add_parser("notify", help="Send latest saved Markdown check report to Telegram.")
    p_not.add_argument("path", nargs="?", type=Path, default=None, help="Optional Markdown report path.")
    args = parser.parse_args()
    if args.cmd == "inspect":
        cmd_inspect()
    elif args.cmd == "notify":
        cmd_notify(args.path)


if __name__ == "__main__":
    main()
