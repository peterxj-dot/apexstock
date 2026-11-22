"""Analyze early Wave-1 exits in SOXL F1+TP3h, test 6 structural exit upgrades, and run 5-part robustness verification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL
from study_major_exits_and_backtest import compute_metrics, run_buy_and_hold
from test_bottom_filter_and_tp_exits import build_study_data
from test_cycle_v2_and_hybrid import run_strategy1_baseline

UPGRADE_EXITS = [
    "B0_X1a_WTSL35",
    "TP3h_Base",
    "U1_MDiff30",
    "U2_MDiff30_or_EMA10_m6",
    "U3_InstantStr35_EMA10_m5",
    "U4_WFishDiv38",
    "U5_Scale50_Wave1_Wave2",
    "U6_Hold65_or_Str100",
]


def prepare_extended_data(ticker: str, bench_ticker: str) -> dict:
    """Build causal daily/weekly data with EMA10 distance and rolling 6w Fisher max."""
    data = build_study_data(ticker, bench_ticker)
    w = data["weekly"]
    w["Close_vs_EMA10%"] = (w["Close"] / w["W_EMA10"] - 1.0) * 100.0
    w["W_Fish_max6w"] = w["W_Fisher"].rolling(6, min_periods=1).max()
    return data


def build_entry_mask(data: dict, wks_th: int | None, dd_th: float | None) -> np.ndarray:
    """Return weekly boolean mask for causal E2 entry with optional (wks_th, dd_th) gate."""
    w = data["weekly"]
    pre = w["pre_c7_vol"].fillna(False).to_numpy(dtype=bool)
    prev_hi = (w["Close"] > w["High"].shift(1)).fillna(False).to_numpy(dtype=bool)
    if wks_th is None or dd_th is None:
        return pre & prev_hi
    wks = w["wks_52H"].fillna(-1.0).to_numpy(dtype=np.float64)
    dd = w["dd52_8wL%"].fillna(0.0).to_numpy(dtype=np.float64)
    return pre & prev_hi & ((wks >= float(wks_th)) | (dd <= float(dd_th)))


def _hit_x1a_arr(mf: float, vs30: float, wc: float, peak_w_close: float) -> tuple[bool, str]:
    if np.isnan(mf) or mf < 2.0:
        return False, ""
    if not np.isnan(vs30) and vs30 < 0.0:
        return True, "M>=2&<30W0"
    if peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
        return True, "M>=2&WTSL35%"
    return False, ""


def _eval_exit_fast(
    exit_name: str,
    wc: float,
    mf: float,
    mf_diff: float,
    vs30: float,
    vs_ema10: float,
    wf: float,
    wf_sig: float,
    wf_max6: float,
    hold_weeks: int,
    peak_w_close: float,
    high_zone_armed: bool,
    scaled_out: bool,
    re_armed_wave2: bool,
    hold_th: int,
    str_th: float,
    mf_th: float,
) -> tuple[str, str, bool, bool]:
    """Evaluate causal exit rule using only completed weekly bar scalars."""
    if exit_name == "B0_X1a_WTSL35":
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    wf_cross_down = (not np.isnan(wf)) and (not np.isnan(wf_sig)) and (wf < wf_sig)
    mf_ge = (not np.isnan(mf)) and (mf >= mf_th)
    vs30_ok = not np.isnan(vs30)
    mf_diff_ok = not np.isnan(mf_diff)
    ema10_ok = not np.isnan(vs_ema10)

    if exit_name == "TP3h_Base":
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= str_th:
            high_zone_armed = True
        if high_zone_armed and wf_cross_down:
            return "SELL", f"TP3h_W{hold_th}_S{int(str_th)}", high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    if exit_name == "U1_MDiff30":
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= str_th:
            high_zone_armed = True
        if high_zone_armed and wf_cross_down and mf_diff_ok and mf_diff <= 0.30:
            return "SELL", "U1_MDiff<=0.30", high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    if exit_name == "U2_MDiff30_or_EMA10_m6":
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= str_th:
            high_zone_armed = True
        cond_mdiff = mf_diff_ok and mf_diff <= 0.30
        cond_ema10 = ema10_ok and vs_ema10 <= -6.0
        if high_zone_armed and wf_cross_down and (cond_mdiff or cond_ema10):
            tag = "U2_MDiff<=0.30" if cond_mdiff else "U2_EMA10<=-6%"
            return "SELL", tag, high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    if exit_name == "U3_InstantStr35_EMA10_m5":
        # Arm at Str >= 35%, require actual trend break below EMA10 (-5%) on WFish cross
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= 35.0:
            high_zone_armed = True
        if high_zone_armed and wf_cross_down and ema10_ok and vs_ema10 <= -5.0:
            return "SELL", "U3_EMA10<=-5%", high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    if exit_name == "U4_WFishDiv38":
        # Skip the initial post-parabolic W_Fisher >= 3.8 thrust rollover; sell subsequent exhaustion cross
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= str_th:
            high_zone_armed = True
        if high_zone_armed and wf_cross_down and (not np.isnan(wf_max6)) and wf_max6 < 3.8:
            return "SELL", "U4_WFishMax6<3.8", high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    if exit_name == "U5_Scale50_Wave1_Wave2":
        if hold_weeks >= hold_th and mf_ge and vs30_ok and vs30 >= str_th:
            high_zone_armed = True
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        if hit:
            return "SELL", r, high_zone_armed, re_armed_wave2
        if not scaled_out:
            if high_zone_armed and wf_cross_down:
                return "SELL_HALF", "U5_W1_Half", high_zone_armed, False
            return "", "", high_zone_armed, re_armed_wave2
        # Stage 2 (holding remaining 50% after Wave-1 scale-out)
        if (not np.isnan(wf)) and (not np.isnan(wf_sig)) and (wf >= wf_sig):
            re_armed_wave2 = True
            return "", "", high_zone_armed, re_armed_wave2
        if re_armed_wave2 and wf_cross_down:
            return "SELL", "U5_W2_WFish<Sig", high_zone_armed, re_armed_wave2
        if ema10_ok and vs_ema10 <= -6.0 and wf_cross_down:
            return "SELL", "U5_EMA10<=-6%", high_zone_armed, re_armed_wave2
        return "", "", high_zone_armed, re_armed_wave2

    if exit_name == "U6_Hold65_or_Str100":
        if mf_ge and vs30_ok and ((hold_weeks >= 65 and vs30 >= str_th) or vs30 >= 100.0):
            high_zone_armed = True
        if high_zone_armed and wf_cross_down:
            return "SELL", "U6_W65_or_Str100", high_zone_armed, re_armed_wave2
        hit, r = _hit_x1a_arr(mf, vs30, wc, peak_w_close)
        return ("SELL", r, high_zone_armed, re_armed_wave2) if hit else ("", "", high_zone_armed, re_armed_wave2)

    raise ValueError(f"Unknown exit rule: {exit_name}")


def run_backtest_param(
    data: dict,
    entry_mask: np.ndarray,
    exit_name: str,
    start_date: str,
    end_date: str,
    hold_th: int = 45,
    str_th: float = 40.0,
    mf_th: float = 2.0,
    use_sl_w25: bool = True,
    fill_mode: str = "t+1_open",
    fee_rate: float = FEE_RATE,
    filter_tag: str = "F1",
) -> dict:
    """Fast strictly-causal weekly-signal simulator with configurable thresholds, fill delay, and fee."""
    daily_df = data["daily"]
    w_df = data["weekly"]
    week_end_daily_pos = data["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    wc_arr = w_df["Close"].to_numpy(dtype=np.float64)
    mf_arr = w_df["M_Fish_prev"].to_numpy(dtype=np.float64)
    mf_diff_arr = w_df["M_Fish_prev_diff"].to_numpy(dtype=np.float64)
    vs30_arr = w_df["Close_vs_30W%"].to_numpy(dtype=np.float64)
    vs_ema10_arr = w_df["Close_vs_EMA10%"].to_numpy(dtype=np.float64)
    wf_arr = w_df["W_Fisher"].to_numpy(dtype=np.float64)
    wf_sig_arr = w_df["W_Fisher_Sig"].to_numpy(dtype=np.float64)
    wf_max6_arr = w_df["W_Fish_max6w"].to_numpy(dtype=np.float64)

    delay_bars = 1 if fill_mode in ("t+1_open", "t+1_close") else 2
    close_fill = fill_mode == "t+1_close"

    cash = INITIAL_CAPITAL
    shares = 0.0
    avg_entry_px = 0.0
    entry_date_str = ""
    partial_str = "-"
    entry_fill_pos = -1
    entry_sig_w_pos = -1
    trade_cash_start = INITIAL_CAPITAL
    scaled_out = False
    high_zone_armed = False
    re_armed_wave2 = False

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_sig_w_pos = -1
    pending_exec_pos = -1
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trade_logs: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")

        just_exited = False
        held_before_close = shares > 0.0

        def _execute_pending(fill_px: float) -> None:
            nonlocal cash, shares, avg_entry_px, entry_date_str, partial_str
            nonlocal entry_fill_pos, entry_sig_w_pos, trade_cash_start
            nonlocal scaled_out, high_zone_armed, re_armed_wave2
            nonlocal peak_d_close, peak_w_close, cooldown_until_pos, just_exited
            nonlocal pending_action, pending_reason, pending_sig_w_pos, pending_exec_pos

            if pending_action == "BUY_FULL" and shares == 0.0:
                trade_cash_start = cash
                shares = (cash * (1.0 - fee_rate)) / fill_px
                cash = 0.0
                avg_entry_px = float(fill_px)
                entry_date_str = dt_str
                partial_str = "-"
                entry_fill_pos = pos
                entry_sig_w_pos = pending_sig_w_pos
                scaled_out = False
                high_zone_armed = False
                re_armed_wave2 = False
                peak_d_close = c
                peak_w_close = 0.0
            elif pending_action == "SELL_HALF" and shares > 0.0 and not scaled_out:
                if pos <= entry_fill_pos:
                    raise AssertionError(f"Same-day scaleout at {dt_str}")
                sell_sh = 0.5 * shares
                cash += sell_sh * fill_px * (1.0 - fee_rate)
                shares -= sell_sh
                scaled_out = True
                partial_str = f"TP1:{dt_str}(${fill_px:.2f},{pending_reason})"
            elif pending_action == "SELL" and shares > 0.0:
                if pos <= entry_fill_pos:
                    raise AssertionError(f"Same-day buy-sell at {dt_str}")
                cash += shares * fill_px * (1.0 - fee_rate)
                ret_pct = (cash / trade_cash_start - 1.0) * 100.0
                trade_logs.append(
                    {
                        "entry_date": entry_date_str,
                        "partial": partial_str,
                        "entry_px": round(avg_entry_px, 2),
                        "exit_date": dt_str,
                        "exit_px": round(float(fill_px), 2),
                        "ret_pct": round(float(ret_pct), 2),
                        "exit_reason": pending_reason,
                    }
                )
                if pending_reason.startswith("SL_"):
                    cooldown_until_pos = pos + 10
                shares = 0.0
                avg_entry_px = 0.0
                just_exited = True

            pending_action = ""
            pending_reason = ""
            pending_sig_w_pos = -1
            pending_exec_pos = -1

        if not close_fill and pending_action and pos == pending_exec_pos:
            _execute_pending(o)

        if shares > 0.0:
            if not close_fill:
                days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if close_fill and pending_action and pos == pending_exec_pos:
            if held_before_close:
                days_invested += 1
            _execute_pending(c)
        elif close_fill and held_before_close:
            days_invested += 1

        equity[step] = cash + shares * c

        if pos >= i_end or pos not in week_end_daily_pos or pending_action:
            continue

        w_pos = week_end_daily_pos[pos]
        wc = float(wc_arr[w_pos])
        mf = float(mf_arr[w_pos])
        mf_diff = float(mf_diff_arr[w_pos])
        vs30 = float(vs30_arr[w_pos])
        vs_ema10 = float(vs_ema10_arr[w_pos])
        wf = float(wf_arr[w_pos])
        wf_sig = float(wf_sig_arr[w_pos])
        wf_max6 = float(wf_max6_arr[w_pos])

        if shares > 0.0:
            if wc > peak_w_close:
                peak_w_close = wc

            unmatured = (np.isnan(mf) or mf < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if use_sl_w25 and unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
                pending_exec_pos = pos + delay_bars
            elif pos > entry_fill_pos:
                hold_weeks = w_pos - entry_sig_w_pos
                act, rsn, high_zone_armed, re_armed_wave2 = _eval_exit_fast(
                    exit_name,
                    wc,
                    mf,
                    mf_diff,
                    vs30,
                    vs_ema10,
                    wf,
                    wf_sig,
                    wf_max6,
                    hold_weeks,
                    peak_w_close,
                    high_zone_armed,
                    scaled_out,
                    re_armed_wave2,
                    hold_th,
                    str_th,
                    mf_th,
                )
                if act:
                    pending_action = act
                    pending_reason = rsn
                    pending_exec_pos = pos + delay_bars
        elif (not just_exited) and pos >= cooldown_until_pos and bool(entry_mask[w_pos]):
            pending_action = "BUY_FULL"
            pending_sig_w_pos = w_pos
            pending_exec_pos = pos + delay_bars

    if shares > 0.0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trade_logs.append(
            {
                "entry_date": entry_date_str,
                "partial": partial_str,
                "entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 2),
                "exit_reason": "OPEN_END",
            }
        )

    res = compute_metrics(equity, trade_logs, days_invested, idx[i_start : i_end + 1])
    res.update({"System": f"{filter_tag}+{exit_name}", "trade_logs": trade_logs})
    return res


def format_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["System", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "Exposure", "WorstYear"]
    return pd.DataFrame([{k: (r.get("System") or r.get("Entry") or "-") if k == "System" else r[k] for k in cols} for r in rows])


def compute_cycle_peaks(data: dict, f1_trades: list[dict]) -> list[dict]:
    """Compute fixed weekly-close and daily-close bull-cycle peaks between successive F1 entries."""
    w = data["weekly"]
    d = data["daily"]
    cycles = []
    for i, t in enumerate(f1_trades):
        ed = pd.Timestamp(t["entry_date"])
        nd = pd.Timestamp(f1_trades[i + 1]["entry_date"]) - pd.Timedelta(days=1) if i + 1 < len(f1_trades) else d.index[-1]
        w_sub = w[(w.index >= ed) & (w.index <= nd)]
        d_sub = d[(d.index >= ed) & (d.index <= nd)]
        w_pk_dt = w_sub["Close"].idxmax()
        d_pk_dt = d_sub["Close"].idxmax()
        cycles.append(
            {
                "cycle": i + 1,
                "entry_date": t["entry_date"],
                "entry_px": float(t["entry_px"]),
                "end_bound": nd.strftime("%Y-%m-%d"),
                "peak_w_date": w_pk_dt.strftime("%Y-%m-%d"),
                "peak_w_close": float(w_sub.loc[w_pk_dt, "Close"]),
                "peak_d_date": d_pk_dt.strftime("%Y-%m-%d"),
                "peak_d_close": float(d_sub.loc[d_pk_dt, "Close"]),
            }
        )
    return cycles


def _effective_exit_px(trade: dict) -> float:
    """Compute effective blended exit price (exact 50/50 blend when scaled out via TP1)."""
    p = trade.get("partial", "-")
    ex2 = float(trade["exit_px"])
    if p != "-" and "($" in p:
        px1 = float(p.split("($")[1].split(",")[0])
        return 0.5 * px1 + 0.5 * ex2
    return ex2


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 24)

    soxl = prepare_extended_data("SOXL", "SMH")
    tqqq = prepare_extended_data("TQQQ", "QQQ")

    f1_mask_s = build_entry_mask(soxl, 30, -80.0)
    f0_mask_s = build_entry_mask(soxl, None, None)
    f0_mask_t = build_entry_mask(tqqq, None, None)

    # =========================================================================
    # PARITY ASSERTIONS
    # =========================================================================
    p_s1 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2026-11-19")
    p_f1_tp3h = run_backtest_param(soxl, f1_mask_s, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F1")
    p_f0_tp3h = run_backtest_param(soxl, f0_mask_s, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F0")
    p_f0_x1a = run_backtest_param(soxl, f0_mask_s, "B0_X1a_WTSL35", "2015-01-01", "2026-11-19", filter_tag="F0")
    p_t_tp3h = run_backtest_param(tqqq, f0_mask_t, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F0")

    assert abs(p_s1["Final"] - 29633196.29) < 1.0, p_s1
    assert abs(p_f1_tp3h["Final"] - 90489657.17) < 1.0 and p_f1_tp3h["Trades"] == 5, p_f1_tp3h
    assert abs(p_f0_tp3h["Final"] - 60877424.74) < 1.0 and p_f0_tp3h["Trades"] == 6, p_f0_tp3h
    assert abs(p_f0_x1a["Final"] - 38035375.82) < 1.0 and p_f0_x1a["Trades"] == 6, p_f0_x1a
    assert abs(p_t_tp3h["Final"] - 2040358.14) < 1.0 and p_t_tp3h["Trades"] == 6, p_t_tp3h
    print(
        "Parity assertions PASSED: "
        f"S1=${p_s1['Final']:,.2f}, F1+TP3h=${p_f1_tp3h['Final']:,.2f}, "
        f"F0+TP3h=${p_f0_tp3h['Final']:,.2f}, TQQQ_F0+TP3h=${p_t_tp3h['Final']:,.2f}\n"
    )

    # =========================================================================
    # AUDIT 1: EXIT UPGRADES U1..U6 COMPARISON & 5-CYCLE PEAK CAPTURE
    # =========================================================================
    print("=" * 120)
    print("AUDIT 1A: SOXL & TQQQ COMPARISON OF EXIT UPGRADES U1..U6")
    print("=" * 120)

    specs = [
        ("SOXL F1 2015-2025 (11yr)", soxl, f1_mask_s, "F1", "2015-01-01", "2025-12-31", "SOXL", "SMH"),
        ("SOXL F1 2015-2026 (12yr)", soxl, f1_mask_s, "F1", "2015-01-01", "2026-11-19", "SOXL", "SMH"),
        ("SOXL F0 2015-2025 (11yr)", soxl, f0_mask_s, "F0", "2015-01-01", "2025-12-31", "SOXL", "SMH"),
        ("SOXL F0 2015-2026 (12yr)", soxl, f0_mask_s, "F0", "2015-01-01", "2026-11-19", "SOXL", "SMH"),
        ("TQQQ F0 2015-2025 (11yr)", tqqq, f0_mask_t, "F0", "2015-01-01", "2025-12-31", "TQQQ", "QQQ"),
        ("TQQQ F0 2015-2026 (12yr)", tqqq, f0_mask_t, "F0", "2015-01-01", "2026-11-19", "TQQQ", "QQQ"),
    ]
    results_12yr_f1: dict[str, dict] = {}
    results_12yr_f0_t: dict[str, dict] = {}
    for label, ds, mask, ftag, s, e, tk, bk in specs:
        rows = [
            run_buy_and_hold(ds, s, e),
            run_strategy1_baseline(tk, bk, s, e),
        ]
        for ex in UPGRADE_EXITS:
            r = run_backtest_param(ds, mask, ex, s, e, filter_tag=ftag)
            rows.append(r)
            if label == "SOXL F1 2015-2026 (12yr)":
                results_12yr_f1[ex] = r
            if label == "TQQQ F0 2015-2026 (12yr)":
                results_12yr_f0_t[ex] = r
        print(f"\n=== {label} ===")
        print(format_table(rows).to_string(index=False))

    print("\n" + "=" * 120)
    print("AUDIT 1B: SOXL F1 5-CYCLE TRADE LOGS & PEAK CAPTURE (%)")
    print("=" * 120)
    cycles = compute_cycle_peaks(soxl, results_12yr_f1["TP3h_Base"]["trade_logs"])
    print("Cycle Peak Benchmarks (Weekly Close & Daily Close within [entry_date, next_entry - 1d]):")
    for cy in cycles:
        w_mult = cy["peak_w_close"] / cy["entry_px"]
        print(
            f"  Cycle {cy['cycle']}: Entry {cy['entry_date']} @ ${cy['entry_px']:.2f} | "
            f"Peak_WClose {cy['peak_w_date']} @ ${cy['peak_w_close']:.2f} ({w_mult:.2f}x) | "
            f"Peak_DClose {cy['peak_d_date']} @ ${cy['peak_d_close']:.2f}"
        )

    print("\nCycle-by-Cycle Exit Comparison (Effective Exit Px / % of Peak Weekly Close Captured):")
    cyc_rows = []
    for ex in UPGRADE_EXITS:
        tlogs = results_12yr_f1[ex]["trade_logs"]
        row = {"Exit": ex, "Final($M)": round(results_12yr_f1[ex]["Final"] / 1e6, 2), "MaxDD%": results_12yr_f1[ex]["MaxDD"]}
        caps = []
        for idx_c, cy in enumerate(cycles):
            if idx_c < len(tlogs):
                t = tlogs[idx_c]
                eff_px = _effective_exit_px(t)
                cap_pct = eff_px / cy["peak_w_close"] * 100.0
                caps.append(cap_pct)
                row[f"C{cy['cycle']}_Exit"] = f"{t['exit_date']} @${eff_px:.2f} ({t['ret_pct']:+.0f}% | {cap_pct:.1f}%pk)"
            else:
                row[f"C{cy['cycle']}_Exit"] = "-"
        row["Avg_PeakCap%"] = round(float(np.mean(caps)), 1)
        cyc_rows.append(row)
    print(pd.DataFrame(cyc_rows).to_string(index=False))

    # =========================================================================
    # AUDIT 2: 64-COMBO 3D EXIT PARAMETER GRID (HOLD x STR x MF)
    # =========================================================================
    print("\n" + "=" * 120)
    print("AUDIT 2: 64-COMBO 3D EXIT PARAMETER GRID (hold=[40,45,50,55] x str=[30,35,40,45] x mf=[1.8,1.9,2.0,2.1])")
    print("=" * 120)

    s1_s12_final = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2026-11-19")["Final"]
    s1_t12_final = run_strategy1_baseline("TQQQ", "QQQ", "2015-01-01", "2026-11-19")["Final"]
    s1_s11_final = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2025-12-31")["Final"]

    grid_exits = ["TP3h_Base", "U1_MDiff30", "U2_MDiff30_or_EMA10_m6", "U5_Scale50_Wave1_Wave2"]
    grid_contexts = [
        ("SOXL_F1_12yr", soxl, f1_mask_s, "F1", "2015-01-01", "2026-11-19", s1_s12_final),
        ("SOXL_F0_12yr", soxl, f0_mask_s, "F0", "2015-01-01", "2026-11-19", s1_s12_final),
        ("SOXL_F0_11yr", soxl, f0_mask_s, "F0", "2015-01-01", "2025-12-31", s1_s11_final),
        ("TQQQ_F0_12yr", tqqq, f0_mask_t, "F0", "2015-01-01", "2026-11-19", s1_t12_final),
    ]

    grid_records: list[dict] = []
    summary_rows: list[dict] = []
    for ctx_name, ds, mask, ftag, s, e, s1_bench in grid_contexts:
        for ex in grid_exits:
            sub = []
            for h_th in (40, 45, 50, 55):
                for st_th in (30.0, 35.0, 40.0, 45.0):
                    for mf_th in (1.8, 1.9, 2.0, 2.1):
                        r = run_backtest_param(
                            ds, mask, ex, s, e, hold_th=h_th, str_th=st_th, mf_th=mf_th, filter_tag=ftag
                        )
                        rec = {
                            "Context": ctx_name,
                            "Exit": ex,
                            "hold": h_th,
                            "str": st_th,
                            "mf": mf_th,
                            "Final": r["Final"],
                            "CAGR": r["CAGR"],
                            "MaxDD": r["MaxDD"],
                            "Calmar": r["Calmar"],
                            "WinRate": r["WinRate"],
                        }
                        sub.append(rec)
                        grid_records.append(rec)

            df_sub = pd.DataFrame(sub)
            beat_pct = float((df_sub["Final"] > s1_bench).mean() * 100.0)
            summary_rows.append(
                {
                    "Context": ctx_name,
                    "Exit": ex,
                    "N": len(df_sub),
                    "Beat_S1%": round(beat_pct, 1),
                    "Min_Final": round(float(df_sub["Final"].min()), 0),
                    "P25_Final": round(float(df_sub["Final"].quantile(0.25)), 0),
                    "Med_Final": round(float(df_sub["Final"].median()), 0),
                    "Mean_Final": round(float(df_sub["Final"].mean()), 0),
                    "Max_Final": round(float(df_sub["Final"].max()), 0),
                    "Med_CAGR%": round(float(df_sub["CAGR"].median()), 2),
                    "Worst_MaxDD%": round(float(df_sub["MaxDD"].min()), 2),
                    "Min_Calmar": round(float(df_sub["Calmar"].min()), 3),
                    "Med_Calmar": round(float(df_sub["Calmar"].median()), 3),
                }
            )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    df_grid = pd.DataFrame(grid_records)
    print("\n--- Breakdown by mf_th (1.8, 1.9, 2.0, 2.1) to expose 2024-03 M_Fish boundary cliff ---")
    for ctx_target in ("SOXL_F1_12yr", "TQQQ_F0_12yr"):
        mf_rows = []
        for ex in grid_exits:
            for mf_val in (1.8, 1.9, 2.0, 2.1):
                part = df_grid[(df_grid["Context"] == ctx_target) & (df_grid["Exit"] == ex) & (df_grid["mf"] == mf_val)]
                mf_rows.append(
                    {
                        "Context": ctx_target,
                        "Exit": ex,
                        "mf_th": mf_val,
                        "N": len(part),
                        "Min_Final": round(float(part["Final"].min()), 0),
                        "Med_Final": round(float(part["Final"].median()), 0),
                        "Max_Final": round(float(part["Final"].max()), 0),
                        "Med_CAGR%": round(float(part["CAGR"].median()), 2),
                        "Worst_DD%": round(float(part["MaxDD"].min()), 2),
                        "Med_Calmar": round(float(part["Calmar"].median()), 3),
                    }
                )
        print(f"\n[{ctx_target}]")
        print(pd.DataFrame(mf_rows).to_string(index=False))

    # =========================================================================
    # AUDIT 3: 18-COMBO F1 ENTRY FILTER GRID (WKS_TH x DD_TH)
    # =========================================================================
    print("\n" + "=" * 120)
    print("AUDIT 3: 18-COMBO F1 ENTRY FILTER GRID ON SOXL (2015-2026)")
    print("=" * 120)
    entry_rows = []
    for wks_th in (24, 26, 28, 30, 32, 35):
        for dd_th in (-78.0, -80.0, -82.0):
            m = build_entry_mask(soxl, wks_th, dd_th)
            r_b = run_backtest_param(soxl, m, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="G")
            r_u1 = run_backtest_param(soxl, m, "U1_MDiff30", "2015-01-01", "2026-11-19", filter_tag="G")
            r_u2 = run_backtest_param(soxl, m, "U2_MDiff30_or_EMA10_m6", "2015-01-01", "2026-11-19", filter_tag="G")
            r_u5 = run_backtest_param(soxl, m, "U5_Scale50_Wave1_Wave2", "2015-01-01", "2026-11-19", filter_tag="G")
            entry_rows.append(
                {
                    "wks_th": wks_th,
                    "dd_th": dd_th,
                    "Trades": r_b["Trades"],
                    "Base_Final": round(r_b["Final"], 0),
                    "Base_DD%": r_b["MaxDD"],
                    "U1_Final": round(r_u1["Final"], 0),
                    "U2_Final": round(r_u2["Final"], 0),
                    "U2_DD%": r_u2["MaxDD"],
                    "U5_Final": round(r_u5["Final"], 0),
                    "U5_DD%": r_u5["MaxDD"],
                }
            )
    df_entry = pd.DataFrame(entry_rows)
    print(df_entry.to_string(index=False))

    # =========================================================================
    # AUDIT 4: EXECUTION DELAY & 5x FEE STRESS TEST
    # =========================================================================
    print("\n" + "=" * 120)
    print("AUDIT 4: EXECUTION DELAY (t+1 Open / t+1 Close / t+2 Open) x FEE STRESS TEST (0.05% vs 0.25%)")
    print("=" * 120)
    stress_rows = []
    stress_targets = [
        ("SOXL F1 + TP3h_Base", soxl, f1_mask_s, "F1", "TP3h_Base"),
        ("SOXL F1 + U1_MDiff30", soxl, f1_mask_s, "F1", "U1_MDiff30"),
        ("SOXL F1 + U2_MDiff30_or_EMA10_m6", soxl, f1_mask_s, "F1", "U2_MDiff30_or_EMA10_m6"),
        ("SOXL F1 + U5_Scale50_Wave1_Wave2", soxl, f1_mask_s, "F1", "U5_Scale50_Wave1_Wave2"),
        ("SOXL F0 + U2_MDiff30_or_EMA10_m6", soxl, f0_mask_s, "F0", "U2_MDiff30_or_EMA10_m6"),
        ("TQQQ F0 + U2_MDiff30_or_EMA10_m6", tqqq, f0_mask_t, "F0", "U2_MDiff30_or_EMA10_m6"),
    ]
    for label, ds, mask, ftag, ex in stress_targets:
        for fm in ("t+1_open", "t+1_close", "t+2_open"):
            for fr in (0.0005, 0.0025):
                r = run_backtest_param(ds, mask, ex, "2015-01-01", "2026-11-19", fill_mode=fm, fee_rate=fr, filter_tag=ftag)
                stress_rows.append(
                    {
                        "Strategy": label,
                        "Fill": fm,
                        "Fee": f"{fr*100:.2f}%",
                        "Final": round(r["Final"], 2),
                        "CAGR": r["CAGR"],
                        "MaxDD": r["MaxDD"],
                        "Calmar": r["Calmar"],
                        "Trades": r["Trades"],
                    }
                )
    print(pd.DataFrame(stress_rows).to_string(index=False))

    # =========================================================================
    # AUDIT 5: EX-2026 ATTRIBUTION (4 COMPLETED CYCLES PRE-CYCLE5 & 2015-2025)
    # =========================================================================
    print("\n" + "=" * 120)
    print("AUDIT 5: EX-2026 CYCLE ATTRIBUTION (PRE-CYCLE-5: 2015-01-01 ~ 2025-04-25 & 11yr: 2015-01-01 ~ 2025-12-31)")
    print("=" * 120)
    s1_pre5 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2025-04-25")
    bh_pre5 = run_buy_and_hold(soxl, "2015-01-01", "2025-04-25")
    attr_rows = [
        {
            "Strategy": "Buy & Hold (SOXL)",
            "Pre5_Final": round(bh_pre5["Final"], 2),
            "Pre5_CAGR": bh_pre5["CAGR"],
            "Pre5_MaxDD": bh_pre5["MaxDD"],
            "4Cyc_GeoMean%": "-",
            "11yr_Final": round(run_buy_and_hold(soxl, "2015-01-01", "2025-12-31")["Final"], 2),
            "12yr_Final": round(run_buy_and_hold(soxl, "2015-01-01", "2026-11-19")["Final"], 2),
        },
        {
            "Strategy": "Strategy 1 (SMA200_hv95)",
            "Pre5_Final": round(s1_pre5["Final"], 2),
            "Pre5_CAGR": s1_pre5["CAGR"],
            "Pre5_MaxDD": s1_pre5["MaxDD"],
            "4Cyc_GeoMean%": "-",
            "11yr_Final": round(run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2025-12-31")["Final"], 2),
            "12yr_Final": round(s1_s12_final, 2),
        },
    ]
    for ex in UPGRADE_EXITS:
        r_pre5 = run_backtest_param(soxl, f1_mask_s, ex, "2015-01-01", "2025-04-25", filter_tag="F1")
        r_11 = run_backtest_param(soxl, f1_mask_s, ex, "2015-01-01", "2025-12-31", filter_tag="F1")
        r_12 = results_12yr_f1[ex]
        # Geometric mean of the 4 completed cycles (C1..C4)
        c4_mults = [1.0 + t["ret_pct"] / 100.0 for t in r_12["trade_logs"][:4]]
        geo_c4 = (float(np.prod(c4_mults)) ** (1.0 / 4.0) - 1.0) * 100.0
        attr_rows.append(
            {
                "Strategy": f"F1+{ex}",
                "Pre5_Final": round(r_pre5["Final"], 2),
                "Pre5_CAGR": r_pre5["CAGR"],
                "Pre5_MaxDD": r_pre5["MaxDD"],
                "4Cyc_GeoMean%": round(geo_c4, 1),
                "11yr_Final": round(r_11["Final"], 2),
                "12yr_Final": round(r_12["Final"], 2),
            }
        )
    print(pd.DataFrame(attr_rows).to_string(index=False))

    # Full detailed trade logs for top candidate upgrades on SOXL & TQQQ
    print("\n" + "=" * 120)
    print("TRADE LOGS: SOXL (F1) & TQQQ (F0) FOR TP3h_Base vs U1 / U2 / U5")
    print("=" * 120)
    for ds_name, rmap in [("SOXL F1 (12yr)", results_12yr_f1), ("TQQQ F0 (12yr)", results_12yr_f0_t)]:
        for ex in ("TP3h_Base", "U1_MDiff30", "U2_MDiff30_or_EMA10_m6", "U5_Scale50_Wave1_Wave2"):
            r = rmap[ex]
            print(f"\n[{ds_name} | {ex}] Final=${r['Final']:,.2f} CAGR={r['CAGR']}% MaxDD={r['MaxDD']}% Calmar={r['Calmar']}")
            print(pd.DataFrame(r["trade_logs"]).to_string(index=False))


if __name__ == "__main__":
    main()
