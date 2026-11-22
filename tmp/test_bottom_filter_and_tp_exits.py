"""Verify time/depth bottom filters (Part 1) and proactive take-profit exits (Part 2) on SOXL & TQQQ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL
from study_major_exits_and_backtest import compute_metrics, run_buy_and_hold
from test_cycle_v2_and_hybrid import extend_causal_data, run_hybrid_backtest, run_strategy1_baseline


def build_study_data(ticker: str, bench_ticker: str) -> dict:
    """Extend causal data with strictly causal time-since-top, crash speed, and weekly EMA5."""
    data = extend_causal_data(ticker, bench_ticker)
    w = data["weekly"]

    hi = w["High"].to_numpy(dtype=np.float64)
    lo = w["Low"].to_numpy(dtype=np.float64)
    n = len(w)
    wks_52h = np.full(n, np.nan, dtype=np.float64)
    dd52_8wl = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        s52 = max(0, i - 51)
        p52 = s52 + int(np.argmax(hi[s52 : i + 1]))
        wks_52h[i] = float(i - p52)
        min8l = float(np.min(lo[max(0, i - 7) : i + 1]))
        dd52_8wl[i] = (min8l / hi[p52] - 1.0) * 100.0

    w["wks_52H"] = wks_52h
    w["dd52_8wL%"] = dd52_8wl
    w["min8w_Low_200W%"] = w["Low_vs_200W%"].rolling(8, min_periods=1).min()
    w["crash_speed"] = np.abs(w["dd52_8wL%"]) / np.maximum(w["wks_52H"], 1.0)
    w["W_EMA5"] = ema(w["Close"], 5)

    filters = {
        "F0_Base": pd.Series(True, index=w.index),
        "F1_Wks30_or_DD80": (w["wks_52H"] >= 30) | (w["dd52_8wL%"] <= -80.0),
        "F2_Wks26_or_DD80": (w["wks_52H"] >= 26) | (w["dd52_8wL%"] <= -80.0),
        "F3_Wks30_or_200W_m40": (w["wks_52H"] >= 30) | (w["min8w_Low_200W%"] <= -40.0),
        "F4_Wks26_or_Vol30": (w["wks_52H"] >= 26) | (w["max4w_Vol_R"] >= 3.0),
        "F5_Wks30_or_Vol30": (w["wks_52H"] >= 30) | (w["max4w_Vol_R"] >= 3.0),
        "F6_Wks28_or_Speed5": (w["wks_52H"] >= 28) | ((w["dd52_8wL%"] <= -60.0) & (w["crash_speed"] >= 4.0)),
    }
    for fname, fcond in filters.items():
        pre_f = w["pre_c7_vol"] & fcond.fillna(False)
        w[f"{fname}_E2"] = pre_f & (w["Close"] > w["High"].shift(1))
        w[f"{fname}_E1"] = pre_f & (w["Close"] > w["W_EMA10"])

    return data


def list_entry_events(data: dict, col: str, start_date: str = "2014-01-01", cluster_weeks: int = 8) -> list[str]:
    """List clustered next-day Open entries triggered by a weekly boolean column."""
    w = data["weekly"]
    d = data["daily"]
    w_pos_map = {v: k for k, v in data["week_end_daily_pos"].items()}  # w_pos -> d_pos

    cond = w[col].fillna(False).to_numpy(dtype=bool)
    w_dates = w.index
    events: list[str] = []
    last_w = -1000
    for wp in np.where(cond)[0]:
        if w_dates[wp] < pd.Timestamp(start_date):
            continue
        if (wp - last_w) <= cluster_weeks:
            continue
        dp = w_pos_map[wp]
        if dp + 1 >= len(d):
            continue
        last_w = wp
        edate = d.index[dp + 1].strftime("%Y-%m-%d")
        epx = float(d["Open"].iloc[dp + 1])
        events.append(f"{edate}(${epx:.2f})")
    return events


def _hit_x1a_wtsl35(w_row: pd.Series, peak_w_close: float) -> tuple[bool, str]:
    mf = w_row["M_Fish_prev"]
    if pd.isna(mf) or float(mf) < 2.0:
        return False, ""
    vs30 = w_row["Close_vs_30W%"]
    wc = float(w_row["Close"])
    if pd.notna(vs30) and float(vs30) < 0.0:
        return True, "M>=2&<30W0"
    if peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
        return True, "M>=2&WTSL35%"
    return False, ""


def eval_tp_exit(
    exit_name: str,
    w_row: pd.Series,
    hold_weeks: int,
    avg_entry_px: float,
    peak_w_close: float,
    high_zone_armed: bool,
    scaled_out: bool,
) -> tuple[str, str, bool]:
    """Return (action in {'', 'SELL', 'SELL_HALF'}, reason, new_high_zone_armed)."""
    wc = float(w_row["Close"])
    mf = w_row["M_Fish_prev"]
    vs30 = w_row["Close_vs_30W%"]
    w_ema5 = w_row["W_EMA5"]
    w_rsi = w_row["W_RSI14"]
    wf = w_row["W_Fisher"]
    wf_sig = w_row["W_Fisher_Sig"]
    gain_pct = (wc / avg_entry_px - 1.0) * 100.0

    mf_ok = pd.notna(mf)
    mf_val = float(mf) if mf_ok else -99.0
    vs30_ok = pd.notna(vs30)
    vs30_val = float(vs30) if vs30_ok else -999.0

    if exit_name == "B0_X1a_WTSL35":
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    # TP1: Pure Time Exit
    if exit_name.startswith("TP1_Hold"):
        tgt_w = int(exit_name.replace("TP1_Hold", "").replace("w", ""))
        if hold_weeks >= tgt_w:
            return "SELL", f"Time>={tgt_w}w", high_zone_armed
        return "", "", high_zone_armed

    # TP2: Fixed Gain Target (Pure or + X1a safety net)
    if exit_name == "TP2a_Pure_300%":
        return ("SELL", "TP>=300%", high_zone_armed) if gain_pct >= 300.0 else ("", "", high_zone_armed)
    if exit_name == "TP2b_Pure_500%":
        return ("SELL", "TP>=500%", high_zone_armed) if gain_pct >= 500.0 else ("", "", high_zone_armed)
    if exit_name == "TP2c_Pure_700%":
        return ("SELL", "TP>=700%", high_zone_armed) if gain_pct >= 700.0 else ("", "", high_zone_armed)

    if exit_name in ("TP2d_300%_or_X1a", "TP2e_500%_or_X1a", "TP2f_700%_or_X1a"):
        tgt = float(exit_name.split("_")[1].replace("%", ""))
        if gain_pct >= tgt:
            return "SELL", f"TP>={int(tgt)}%", high_zone_armed
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    # TP3a..e: Pure proactive strength TP (no 30W breakdown fallback)
    if exit_name == "TP3a_W40_M20_30W45":
        if hold_weeks >= 40 and mf_val >= 2.0 and vs30_val >= 45.0:
            return "SELL", "W40&M2&30W>=45", high_zone_armed
        return "", "", high_zone_armed
    if exit_name == "TP3b_W52_M20_30W40":
        if hold_weeks >= 52 and mf_val >= 2.0 and vs30_val >= 40.0:
            return "SELL", "W52&M2&30W>=40", high_zone_armed
        return "", "", high_zone_armed
    if exit_name == "TP3c_W40_M25":
        if hold_weeks >= 40 and mf_val >= 2.5:
            return "SELL", "W40&M>=2.5", high_zone_armed
        return "", "", high_zone_armed
    if exit_name == "TP3d_W52_M25":
        if hold_weeks >= 52 and mf_val >= 2.5:
            return "SELL", "W52&M>=2.5", high_zone_armed
        return "", "", high_zone_armed
    if exit_name == "TP3e_W52_RSI72":
        if hold_weeks >= 52 and mf_val >= 2.0 and pd.notna(w_rsi) and float(w_rsi) >= 72.0:
            return "SELL", "W52&M2&RSI72", high_zone_armed
        return "", "", high_zone_armed

    # TP3f..h: Proactive High-Zone Momentum-Crack TP (+ X1a safety fallback)
    if exit_name == "TP3f_W45_M20_Str40_EMA5":
        if hold_weeks >= 45 and mf_val >= 2.0 and vs30_val >= 40.0:
            high_zone_armed = True
        if high_zone_armed and pd.notna(w_ema5) and wc < float(w_ema5):
            return "SELL", "HighStr40&<EMA5", high_zone_armed
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    if exit_name == "TP3g_W52_M20_Str35_EMA5":
        if hold_weeks >= 52 and mf_val >= 2.0 and vs30_val >= 35.0:
            high_zone_armed = True
        if high_zone_armed and pd.notna(w_ema5) and wc < float(w_ema5):
            return "SELL", "HighStr35&<EMA5", high_zone_armed
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    if exit_name == "TP3h_W45_M20_Str40_WFishDown":
        if hold_weeks >= 45 and mf_val >= 2.0 and vs30_val >= 40.0:
            high_zone_armed = True
        if high_zone_armed and pd.notna(wf) and pd.notna(wf_sig) and float(wf) < float(wf_sig):
            return "SELL", "HighStr40&WFish<Sig", high_zone_armed
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    # TP3i..j: Mature tight trailing stop
    if exit_name in ("TP3i_W52_M20_TightTSL20", "TP3j_W52_M20_TightTSL25"):
        tsl_pct = 0.20 if "20" in exit_name else 0.25
        if hold_weeks >= 52 and mf_val >= 2.0:
            if peak_w_close > 0.0 and wc <= peak_w_close * (1.0 - tsl_pct):
                return "SELL", f"W52&M2&TSL{int(tsl_pct*100)}%", high_zone_armed
            if vs30_ok and vs30_val < 0.0:
                return "SELL", "M>=2&<30W0", high_zone_armed
            return "", "", high_zone_armed
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        return ("SELL", r, high_zone_armed) if hit else ("", "", high_zone_armed)

    # TP4: 50/50 Scale-out + X1a rest
    if exit_name == "TP4a_Half_W52_M20_30W40_Rest_X1a":
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        if hit:
            return "SELL", r, high_zone_armed
        if (not scaled_out) and hold_weeks >= 52 and mf_val >= 2.0 and vs30_val >= 40.0:
            return "SELL_HALF", "TP1_50%_W52_30W40", high_zone_armed
        return "", "", high_zone_armed

    if exit_name == "TP4b_Half_300%_Rest_X1a":
        hit, r = _hit_x1a_wtsl35(w_row, peak_w_close)
        if hit:
            return "SELL", r, high_zone_armed
        if (not scaled_out) and gain_pct >= 300.0:
            return "SELL_HALF", "TP1_50%_>=300%", high_zone_armed
        return "", "", high_zone_armed

    raise ValueError(f"Unknown TP exit rule: {exit_name}")


def run_tp_backtest(
    data: dict,
    filter_name: str,
    exit_name: str,
    start_date: str,
    end_date: str,
    use_split_50_50: bool = False,
    use_sl_w25: bool = True,
) -> dict:
    """Strictly causal weekly-signal backtester supporting Part 1 filters, Split50, and TP exits."""
    daily_df = data["daily"]
    w_df = data["weekly"]
    week_end_daily_pos = data["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    e2_col = f"{filter_name}_E2"
    e1_col = f"{filter_name}_E1"
    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    tranches = 0
    avg_entry_px = 0.0
    entry_date_str = ""
    tp1_info_str = "-"
    entry_fill_pos = -1
    entry_sig_w_pos = -1
    trade_cash_start = INITIAL_CAPITAL
    scaled_out = False
    high_zone_armed = False

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""  # "", "BUY_FULL", "BUY_HALF", "BUY_TOPUP", "SELL_HALF", "SELL"
    pending_reason = ""
    pending_sig_w_pos = -1
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trade_logs: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")

        just_exited = False
        if pending_action == "BUY_FULL" and tranches == 0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            tranches = 2
            avg_entry_px = float(o)
            entry_date_str = dt_str
            tp1_info_str = "-"
            entry_fill_pos = pos
            entry_sig_w_pos = pending_sig_w_pos
            scaled_out = False
            high_zone_armed = False
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_HALF" and tranches == 0:
            trade_cash_start = cash
            cost = 0.5 * cash
            shares = (cost * (1.0 - FEE_RATE)) / o
            cash -= cost
            tranches = 1
            avg_entry_px = float(o)
            entry_date_str = dt_str
            tp1_info_str = "-"
            entry_fill_pos = pos
            entry_sig_w_pos = pending_sig_w_pos
            scaled_out = False
            high_zone_armed = False
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_TOPUP" and tranches == 1 and shares > 0:
            new_shares = (cash * (1.0 - FEE_RATE)) / o
            tot = shares + new_shares
            avg_entry_px = (shares * avg_entry_px + new_shares * o) / tot
            shares = tot
            cash = 0.0
            tranches = 2
            tp1_info_str = f"T2:{dt_str}(${o:.2f})"
            if c > peak_d_close:
                peak_d_close = c
        elif pending_action == "SELL_HALF" and shares > 0 and not scaled_out:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day scaleout at {dt_str}")
            sell_sh = 0.5 * shares
            cash += sell_sh * o * (1.0 - FEE_RATE)
            shares -= sell_sh
            scaled_out = True
            tp1_info_str = f"TP1:{dt_str}(${o:.2f},{pending_reason})"
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            cash += shares * o * (1.0 - FEE_RATE)
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trade_logs.append(
                {
                    "entry_date": entry_date_str,
                    "partial": tp1_info_str,
                    "entry_px": round(avg_entry_px, 2),
                    "exit_date": dt_str,
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                }
            )
            if pending_reason.startswith("SL_"):
                cooldown_until_pos = pos + 10
            shares = 0.0
            tranches = 0
            avg_entry_px = 0.0
            just_exited = True
        pending_action = ""
        pending_reason = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end or pos not in week_end_daily_pos:
            continue

        w_pos = week_end_daily_pos[pos]
        w_row = w_df.iloc[w_pos]

        if shares > 0:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            mf = w_row["M_Fish_prev"]
            unmatured = (pd.isna(mf) or float(mf) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if use_sl_w25 and unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
            elif pos > entry_fill_pos:
                hold_weeks = w_pos - entry_sig_w_pos
                act, rsn, high_zone_armed = eval_tp_exit(
                    exit_name, w_row, hold_weeks, avg_entry_px, peak_w_close, high_zone_armed, scaled_out
                )
                if act:
                    pending_action = act
                    pending_reason = rsn
                elif use_split_50_50 and tranches == 1 and bool(w_row[e1_col]):
                    pending_action = "BUY_TOPUP"

        elif not just_exited and pos >= cooldown_until_pos:
            if use_split_50_50:
                if bool(w_row[e1_col]):
                    pending_action = "BUY_FULL"
                    pending_sig_w_pos = w_pos
                elif bool(w_row[e2_col]):
                    pending_action = "BUY_HALF"
                    pending_sig_w_pos = w_pos
            elif bool(w_row[e2_col]):
                pending_action = "BUY_FULL"
                pending_sig_w_pos = w_pos

    if shares > 0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trade_logs.append(
            {
                "entry_date": entry_date_str,
                "partial": tp1_info_str,
                "entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
            }
        )

    res = compute_metrics(equity, trade_logs, days_invested, idx[i_start : i_end + 1])
    tag = f"{filter_name} + {exit_name}" + ("_Split50" if use_split_50_50 else "")
    res.update({"System": tag, "trade_logs": trade_logs})
    return res


def format_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["System", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "Exposure", "WorstYear"]
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows])


def main() -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)

    soxl = build_study_data("SOXL", "SMH")
    tqqq = build_study_data("TQQQ", "QQQ")

    # Parity verification
    base_s12 = run_tp_backtest(soxl, "F0_Base", "B0_X1a_WTSL35", "2015-01-01", "2026-11-19")
    base_t12 = run_tp_backtest(tqqq, "F0_Base", "B0_X1a_WTSL35", "2015-01-01", "2026-11-19")
    assert abs(base_s12["Final"] - 38035375.82) < 1.0 and base_s12["Trades"] == 6, base_s12
    assert abs(base_t12["Final"] - 1217163.54) < 1.0 and base_t12["Trades"] == 6, base_t12
    print("Parity assertions PASSED (SOXL $38,035,375.82, TQQQ $1,217,163.54).\n")

    s1_s11 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2025-12-31")
    s1_s12 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2026-11-19")
    s1_t11 = run_strategy1_baseline("TQQQ", "QQQ", "2015-01-01", "2025-12-31")
    s1_t12 = run_strategy1_baseline("TQQQ", "QQQ", "2015-01-01", "2026-11-19")

    # =========================================================================
    # PART 1: TIME-OR-DEPTH BOTTOM FILTER
    # =========================================================================
    filter_names = [
        "F0_Base",
        "F1_Wks30_or_DD80",
        "F2_Wks26_or_DD80",
        "F3_Wks30_or_200W_m40",
        "F4_Wks26_or_Vol30",
        "F5_Wks30_or_Vol30",
        "F6_Wks28_or_Speed5",
    ]

    print("=" * 110)
    print("PART 1 TABLE A: TRIGGERED E2_PrevHigh ENTRY DATES (2014-2026)")
    print("=" * 110)
    for tk, ds in [("SOXL", soxl), ("TQQQ", tqqq)]:
        print(f"\n[{tk}]")
        for fn in filter_names:
            evs = list_entry_events(ds, f"{fn}_E2")
            print(f"  {fn:24s} ({len(evs)}): {', '.join(evs)}")

    print("\n" + "=" * 110)
    print("PART 1 TABLE B: BACKTEST IMPACT OF TIME-OR-DEPTH FILTER (Strategy 3 & Strategy 4)")
    print("=" * 110)
    for label, ds, s, e, s1_r in [
        ("SOXL 2015-2025 (11yr)", soxl, "2015-01-01", "2025-12-31", s1_s11),
        ("SOXL 2015-2026 (12yr)", soxl, "2015-01-01", "2026-11-19", s1_s12),
        ("TQQQ 2015-2025 (11yr)", tqqq, "2015-01-01", "2025-12-31", s1_t11),
        ("TQQQ 2015-2026 (12yr)", tqqq, "2015-01-01", "2026-11-19", s1_t12),
    ]:
        rows = [s1_r]
        for fn in filter_names:
            rows.append(run_tp_backtest(ds, fn, "B0_X1a_WTSL35", s, e, use_split_50_50=False))
        for fn in ("F0_Base", "F1_Wks30_or_DD80", "F4_Wks26_or_Vol30"):
            rows.append(run_tp_backtest(ds, fn, "B0_X1a_WTSL35", s, e, use_split_50_50=True))
        for fn in ("F0_Base", "F1_Wks30_or_DD80", "F4_Wks26_or_Vol30"):
            rows.append(
                run_hybrid_backtest(
                    ds,
                    f"H5_{fn}",
                    s,
                    e,
                    bot_entry_col=f"{fn}_E2",
                    use_sl_w25=True,
                    trend_mode="SMA200",
                    hv_mode="ADAPT_115_140",
                    hv_reentry=95.0,
                    enable_alt=True,
                )
            )
        print(f"\n=== {label} ===")
        print(format_df(rows).to_string(index=False))

    # =========================================================================
    # PART 2: PROACTIVE TAKE-PROFIT EXITS VS PASSIVE BREAKDOWN EXIT
    # =========================================================================
    tp_rules = [
        "B0_X1a_WTSL35",
        "TP1_Hold52w",
        "TP1_Hold65w",
        "TP1_Hold78w",
        "TP1_Hold90w",
        "TP2a_Pure_300%",
        "TP2b_Pure_500%",
        "TP2c_Pure_700%",
        "TP2d_300%_or_X1a",
        "TP2e_500%_or_X1a",
        "TP2f_700%_or_X1a",
        "TP3a_W40_M20_30W45",
        "TP3b_W52_M20_30W40",
        "TP3c_W40_M25",
        "TP3d_W52_M25",
        "TP3e_W52_RSI72",
        "TP3f_W45_M20_Str40_EMA5",
        "TP3g_W52_M20_Str35_EMA5",
        "TP3h_W45_M20_Str40_WFishDown",
        "TP3i_W52_M20_TightTSL20",
        "TP3j_W52_M20_TightTSL25",
        "TP4a_Half_W52_M20_30W40_Rest_X1a",
        "TP4b_Half_300%_Rest_X1a",
    ]

    print("\n" + "=" * 110)
    print("PART 2 TABLE A: PROACTIVE TAKE-PROFIT SCAN (on F0_Base, isolated from Part 1)")
    print("=" * 110)
    p2_map: dict[str, dict[str, dict]] = {}
    for label, ds, s, e, s1_r in [
        ("SOXL 2015-2025 (11yr)", soxl, "2015-01-01", "2025-12-31", s1_s11),
        ("SOXL 2015-2026 (12yr)", soxl, "2015-01-01", "2026-11-19", s1_s12),
        ("TQQQ 2015-2025 (11yr)", tqqq, "2015-01-01", "2025-12-31", s1_t11),
        ("TQQQ 2015-2026 (12yr)", tqqq, "2015-01-01", "2026-11-19", s1_t12),
    ]:
        rows = [s1_r]
        p2_map[label] = {}
        for ex in tp_rules:
            r = run_tp_backtest(ds, "F0_Base", ex, s, e)
            rows.append(r)
            p2_map[label][ex] = r
        print(f"\n=== {label} ===")
        print(format_df(rows).to_string(index=False))

    # Pick Top 5 TP rules by balanced SOXL + TQQQ Calmar & CAGR
    scored = []
    for ex in tp_rules:
        if ex == "B0_X1a_WTSL35":
            continue
        s11 = p2_map["SOXL 2015-2025 (11yr)"][ex]
        s12 = p2_map["SOXL 2015-2026 (12yr)"][ex]
        t12 = p2_map["TQQQ 2015-2026 (12yr)"][ex]
        sc = (s11["Calmar"] + s12["Calmar"] + t12["Calmar"]) * 10.0 + s11["CAGR"] * 0.2 + s12["CAGR"] * 0.2 + t12["CAGR"] * 0.3
        scored.append((sc, ex))
    scored.sort(reverse=True)
    top_tp = [x[1] for x in scored[:4]]
    for must in ("TP3b_W52_M20_30W40", "TP3f_W45_M20_Str40_EMA5", "TP3i_W52_M20_TightTSL20", "TP4a_Half_W52_M20_30W40_Rest_X1a"):
        if must not in top_tp:
            top_tp.append(must)
    top_tp = ["B0_X1a_WTSL35"] + top_tp[:5]

    print("\n" + "=" * 110)
    print("PART 2 TABLE B: COMBINED (Cleanest Part 1 Filter F1_Wks30_or_DD80 & F4_Wks26_or_Vol30) x TOP TP EXITS")
    print("=" * 110)
    for label, ds, s, e, s1_r in [
        ("SOXL 2015-2025 (11yr)", soxl, "2015-01-01", "2025-12-31", s1_s11),
        ("SOXL 2015-2026 (12yr)", soxl, "2015-01-01", "2026-11-19", s1_s12),
        ("TQQQ 2015-2025 (11yr)", tqqq, "2015-01-01", "2025-12-31", s1_t11),
        ("TQQQ 2015-2026 (12yr)", tqqq, "2015-01-01", "2026-11-19", s1_t12),
    ]:
        rows = [s1_r]
        for fn in ("F0_Base", "F1_Wks30_or_DD80", "F4_Wks26_or_Vol30"):
            for ex in top_tp:
                rows.append(run_tp_backtest(ds, fn, ex, s, e))
        print(f"\n=== {label} ===")
        print(format_df(rows).to_string(index=False))

    print("\n" + "=" * 110)
    print("TRADE-BY-TRADE LOGS: Baseline vs Best Filter + Best TP Candidates (2015-2026)")
    print("=" * 110)
    for tk, ds in [("SOXL", soxl), ("TQQQ", tqqq)]:
        for fn, ex in [
            ("F1_Wks30_or_DD80", "B0_X1a_WTSL35"),
            ("F4_Wks26_or_Vol30", "B0_X1a_WTSL35"),
            ("F1_Wks30_or_DD80", top_tp[1]),
            ("F1_Wks30_or_DD80", top_tp[2]),
            ("F0_Base", "TP4a_Half_W52_M20_30W40_Rest_X1a"),
        ]:
            r = run_tp_backtest(ds, fn, ex, "2015-01-01", "2026-11-19")
            print(
                f"\n--- {tk} | {r['System']} | Final=${r['Final']:,.2f} | CAGR={r['CAGR']}% | MaxDD={r['MaxDD']}% | Calmar={r['Calmar']} | Trades={r['Trades']} | Win={r['WinRate']}% ---"
            )
            print(pd.DataFrame(r["trade_logs"]).to_string(index=False))


if __name__ == "__main__":
    main()
