"""Helper CLI for ApexStock daily AI analysis reports (snapshot dumper + Telegram sender)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import send_tg_notification

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "logs" / "daily_reports"


def _extract_json(html_path: Path, var_name: str) -> dict | None:
    if not html_path.exists():
        return None
    txt = html_path.read_text(encoding="utf-8")
    m = re.search(rf"const {re.escape(var_name)}\s*=\s*(\{{.*?\}});", txt)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _ticker_metrics() -> list[dict]:
    tickers = ["SOXL", "SMH", "QQQ", "TQQQ", "VGT", "TECL", "MSTR", "BTC-USD"]
    out: list[dict] = []
    for sym in tickers:
        csv_path = DATA_DIR / f"{sym}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        if df.empty:
            continue
        c = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        c = pd.to_numeric(c, errors="coerce")
        raw_c = pd.to_numeric(df["Close"], errors="coerce").replace(0, np.nan)
        ratio = (c / raw_c).fillna(1.0)
        h = pd.to_numeric(df["High"], errors="coerce") * ratio
        l = pd.to_numeric(df["Low"], errors="coerce") * ratio

        ema5 = c.ewm(span=5, adjust=False).mean()
        sma200 = c.rolling(200).mean()
        win52 = 365 if sym == "BTC-USD" else 252
        h52 = h.rolling(win52, min_periods=min(60, len(df))).max()

        last_c = float(c.iloc[-1])
        prev_c = float(c.iloc[-2]) if len(df) >= 2 else last_c
        c5_ago = float(c.iloc[-6]) if len(df) >= 6 else prev_c
        sma200_v = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
        ema5_v = float(ema5.iloc[-1]) if pd.notna(ema5.iloc[-1]) else None
        h52_v = float(h52.iloc[-1]) if pd.notna(h52.iloc[-1]) else None

        out.append({
            "ticker": sym,
            "last_date": str(df["Date"].iloc[-1].date()),
            "close": round(last_c, 2),
            "ret_1d_pct": round((last_c / prev_c - 1.0) * 100.0, 2) if prev_c else 0.0,
            "ret_5d_pct": round((last_c / c5_ago - 1.0) * 100.0, 2) if c5_ago else 0.0,
            "low_5d": round(float(l.iloc[-5:].min()), 2),
            "high_5d": round(float(h.iloc[-5:].max()), 2),
            "dd52w_pct": round((last_c / h52_v - 1.0) * 100.0, 1) if h52_v else None,
            "vs_200d_pct": round((last_c / sma200_v - 1.0) * 100.0, 1) if sma200_v else None,
            "ema5": round(ema5_v, 2) if ema5_v else None,
            "vs_5ema_pct": round((last_c / ema5_v - 1.0) * 100.0, 1) if ema5_v else None,
        })
    return out


def _load_fred(series_id: str) -> pd.Series:
    p = DATA_DIR / f"FRED_{series_id}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    df.columns = ["Date", "Val"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Val"] = pd.to_numeric(df["Val"], errors="coerce")
    df = df.dropna().sort_values("Date")
    return df.set_index("Date")["Val"]


def _macro_liquidity() -> dict:
    walcl = _load_fred("WALCL")
    tga = _load_fred("WTREGEN")
    rrp = _load_fred("RRPONTSYD")
    m2 = _load_fred("M2SL")
    dfii10 = _load_fred("DFII10")

    res: dict = {}
    if not walcl.empty and not tga.empty and not rrp.empty:
        walcl_b = walcl / 1000.0 if walcl.median() > 100_000 else walcl
        tga_b = tga / 1000.0 if tga.median() > 10_000 else tga
        rrp_b = rrp / 1000.0 if rrp.median() > 10_000 else rrp
        idx = walcl_b.index.union(tga_b.index).union(rrp_b.index).sort_values()
        net_b = (walcl_b.reindex(idx).ffill() - tga_b.reindex(idx).ffill() - rrp_b.reindex(idx).ffill()).dropna()
        last_dt = net_b.index[-1]
        cur_val = float(net_b.iloc[-1])
        res["net_liq"] = {
            "last_date": str(last_dt.date()),
            "val_b": round(cur_val, 1),
            "walcl_b": round(float(walcl_b.iloc[-1]), 1),
            "walcl_date": str(walcl_b.index[-1].date()),
            "tga_b": round(float(tga_b.iloc[-1]), 1),
            "tga_date": str(tga_b.index[-1].date()),
            "rrp_b": round(float(rrp_b.iloc[-1]), 2),
            "rrp_date": str(rrp_b.index[-1].date()),
            "chg_14d_b": round(cur_val - float(net_b.asof(last_dt - pd.Timedelta(days=14))), 1),
            "chg_30d_b": round(cur_val - float(net_b.asof(last_dt - pd.Timedelta(days=30))), 1),
            "chg_60d_b": round(cur_val - float(net_b.asof(last_dt - pd.Timedelta(days=60))), 1),
            "chg_90d_b": round(cur_val - float(net_b.asof(last_dt - pd.Timedelta(days=90))), 1),
        }
    if not dfii10.empty:
        d_last = dfii10.index[-1]
        res["dfii10"] = {
            "last_date": str(d_last.date()),
            "val": round(float(dfii10.iloc[-1]), 2),
            "chg_7d": round(float(dfii10.iloc[-1]) - float(dfii10.asof(d_last - pd.Timedelta(days=7))), 2),
            "chg_30d": round(float(dfii10.iloc[-1]) - float(dfii10.asof(d_last - pd.Timedelta(days=30))), 2),
            "chg_90d": round(float(dfii10.iloc[-1]) - float(dfii10.asof(d_last - pd.Timedelta(days=90))), 2),
        }
    if not m2.empty:
        m_last = m2.index[-1]
        res["m2sl"] = {
            "last_date": str(m_last.date()),
            "val_b": round(float(m2.iloc[-1]), 1),
            "chg_90d_b": round(float(m2.iloc[-1]) - float(m2.asof(m_last - pd.Timedelta(days=90))), 1),
        }
    return res


def _previous_report() -> dict | None:
    if not REPORTS_DIR.exists():
        return None
    files = sorted(REPORTS_DIR.glob("*.md"))
    if not files:
        return None
    today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    candidates = [f for f in files if f.stem != today_jst]
    target = candidates[-1] if candidates else files[-1]
    return {
        "file": str(target),
        "date": target.stem,
        "content": target.read_text(encoding="utf-8"),
    }


def cmd_dump() -> None:
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    soxl_lm = _extract_json(ROOT / "soxl_monthly_zones.html", "LIVE_MARKET") or {}
    mstr_lm = _extract_json(ROOT / "mstr_monthly_zones.html", "LIVE_MARKET") or {}
    btc_lm = _extract_json(ROOT / "btc_monthly_zones.html", "LIVE_MARKET") or {}
    risk_live = _extract_json(ROOT / "risk_radar.html", "RISK_LIVE") or {}
    bottom_tracker = _extract_json(ROOT / "risk_radar.html", "BOTTOM_TRACKER") or {}
    news = _extract_json(ROOT / "risk_radar.html", "NEWS") or {}

    soxl_strats = _extract_json(ROOT / "soxl_monthly_zones.html", "STRATEGIES") or {}
    mstr_strats = _extract_json(ROOT / "mstr_monthly_zones.html", "STRATEGIES") or {}
    btc_strats = _extract_json(ROOT / "btc_monthly_zones.html", "STRATEGIES") or {}

    def _strat_live(strats: dict) -> dict:
        return {k: v.get("live_status") for k, v in strats.items() if isinstance(v, dict) and "live_status" in v}

    payload = {
        "generated_at_jst": now_jst.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "today_jst": now_jst.strftime("%Y-%m-%d"),
        "tickers": _ticker_metrics(),
        "macro_liquidity": _macro_liquidity(),
        "risk_radar": {
            "live": risk_live,
            "bottom_tracker": {k: v for k, v in bottom_tracker.items() if k != "recent20_series"},
            "top_news": (news.get("items") or [])[:10],
        },
        "soxl": {
            "live_market": {k: v for k, v in soxl_lm.items() if k != "soxl_gap_tracker"},
            "deep85_gap_tracker": soxl_lm.get("soxl_gap_tracker"),
            "strategies_live": _strat_live(soxl_strats),
        },
        "mstr": {
            "mstr_live": mstr_lm.get("mstr_live"),
            "mstr_t0_tracker": mstr_lm.get("mstr_t0"),
            "strategies_live": _strat_live(mstr_strats),
        },
        "btc": {
            "btc_live": btc_lm.get("btc_live"),
            "strategies_live": _strat_live(btc_strats),
        },
        "previous_report": _previous_report(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _md_to_telegram_text(md_text: str) -> str:
    lines: list[str] = []
    for raw in md_text.splitlines():
        s = raw.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^\|?[\s:\-]+\|[\s:\-\|]*$", s):
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            lines.append(" · ".join(cells))
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = s.replace("**", "").replace("`", "")
        if s.startswith("> "):
            s = s[2:]
        lines.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text


def cmd_notify(md_path: Path | None = None) -> None:
    if md_path is None:
        files = sorted(REPORTS_DIR.glob("*.md")) if REPORTS_DIR.exists() else []
        if not files:
            raise FileNotFoundError(f"No Markdown report found in {REPORTS_DIR}")
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
    parser = argparse.ArgumentParser(description="ApexStock daily AI report helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump", help="Dump full verified JSON snapshot for daily AI analysis.")
    p_not = sub.add_parser("notify", help="Send a saved Markdown report to Telegram (defaults to latest).")
    p_not.add_argument("path", nargs="?", type=Path, default=None, help="Optional path to Markdown report file.")
    args = parser.parse_args()
    if args.cmd == "dump":
        cmd_dump()
    elif args.cmd == "notify":
        cmd_notify(args.path)


if __name__ == "__main__":
    main()
