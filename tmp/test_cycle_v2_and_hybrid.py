"""Evaluate Direction 1 (refined pure major-cycle exits) and Direction 2 (Hybrid E2 + Strategy 1 defense)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, historical_volatility, sma
from scan_exit_conditioned_reentry import simulate_exit_conditioned
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL, compute_asset_features, load_adjusted
from study_major_exits_and_backtest import compute_metrics, prepare_backtest_data, run_buy_and_hold


def extend_causal_data(ticker: str, bench_ticker: str) -> dict:
    """Extend prepare_backtest_data with causal HV20, benchmark SMAs/EMAs, and 12W peak drawdown."""
    data = prepare_backtest_data(ticker, bench_ticker)
    d = data["daily"]
    w = data["weekly"]

    d["HV20"] = historical_volatility(d["Close"], period=20, annualization_factor=252)
    bench_df = load_adjusted(bench_ticker).reindex(d.index, method="ffill")
    d["Bench_Close"] = bench_df["Close"]
    d["Bench_SMA200"] = sma(bench_df["Close"], 200)
    d["Bench_EMA150"] = ema(bench_df["Close"], 150)
    d["is_week_end"] = d.index.isin(w.index)

    w["HV20"] = d["HV20"].reindex(w.index)
    w["W_Peak12w"] = w["Close"].rolling(12, min_periods=1).max()
    w["DD_12w%"] = (w["Close"] / w["W_Peak12w"] - 1.0) * 100.0
    w["ALT_Buy_S1"] = (w["W_Fisher"] < -1.8) & ((w["dW_Fish"] > 0) | (w["W_Vol_R"] > 1.3))
    return data


def run_strategy1_baseline(
    target_ticker: str,
    bench_ticker: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Exact Strategy 1 (Paired_SMA200_hv95) benchmark matching AGENTS.md."""
    target = load_adjusted(target_ticker)
    bench = load_adjusted(bench_ticker)

    full_idx = target.index
    tf = compute_asset_features(target, full_idx)
    bf = compute_asset_features(bench, full_idx)

    mask = (full_idx >= pd.Timestamp(start_date)) & (full_idx <= pd.Timestamp(end_date))
    bt_idx = full_idx[mask]

    def sl(obj) -> np.ndarray:
        return obj.to_numpy()[mask]

    opens = sl(tf["daily"]["Open"]).astype(np.float64)
    closes = sl(tf["daily"]["Close"]).astype(np.float64)
    is_w_end = sl(tf["is_week_end"]).astype(bool)
    t_hv20 = sl(tf["D_HV20"]).astype(np.float64)
    t_wf = sl(tf["W_Fisher"]).astype(np.float64)
    t_dwf = sl(tf["dW_Fisher"]).astype(np.float64)
    t_wvr = sl(tf["W_Vol_R"]).astype(np.float64)
    b_dc = sl(bf["daily"]["Close"]).astype(np.float64)
    b_sma200 = sl(bf["D_SMA200"]).astype(np.float64)

    vcap = 115.0
    vol_exit = t_hv20 > vcap
    alt_buy = is_w_end & (t_wf < -1.8) & ((t_dwf > 0) | (t_wvr > 1.3))
    tb_raw = b_dc > b_sma200 * 1.01
    tb = tb_raw & (t_hv20 <= vcap * 0.92)
    mb = b_dc < b_sma200 * 0.99

    yrs = bt_idx.year.to_numpy()
    yei = [(int(y), int(np.where(yrs == y)[0][-1])) for y in sorted(set(yrs))]

    r = simulate_exit_conditioned(
        opens,
        closes,
        tb,
        vol_exit,
        mb,
        alt_buy,
        yei,
        bt_idx,
        trend_tsl=None,
        alt_fsl=0.20,
        alt_tp=0.25,
        alt_max_days=40,
        default_acd=10,
        after_tsl_spec=("default", tb, alt_buy, 0),
        after_vol_spec=("hv_drop_95", tb_raw & (t_hv20 <= 95.0), alt_buy, 0),
        after_tb_spec=("default", tb, alt_buy, 0),
        after_fsl_spec=("default", tb, alt_buy, -1),
        after_tp_spec=("default", tb, alt_buy, -1, -1, None),
        record_details=True,
    )
    return {
        "System": "S1_Paired_SMA200_hv95",
        "Final": r["Final"],
        "Ret%": round((r["Final"] / INITIAL_CAPITAL - 1.0) * 100.0, 1),
        "CAGR": r["CAGR"],
        "MaxDD": r["MaxDD"],
        "Calmar": r["Calmar"],
        "Sharpe": r["Sharpe"],
        "Trades": r["Trades"],
        "WinRate": r["WinRate"],
        "Exposure": r["Exposure"],
        "WorstYear": r["WorstYear"],
        "yearly_rets": r["yearly_rets"],
        "trade_logs": r["trade_logs"],
    }


def check_dir1_exit(rule: str, w_row: pd.Series, peak_w_close: float) -> tuple[bool, str]:
    mf = w_row["M_Fish_prev"]
    if pd.isna(mf) or float(mf) < 2.0:
        return False, ""
    vs30 = w_row["Close_vs_30W%"]
    b_vs30 = w_row["Bench_vs_30W%"]
    dd12 = w_row["DD_12w%"]
    hv20 = w_row["HV20"]
    wc = float(w_row["Close"])
    w_ema10 = w_row["W_EMA10"]
    tsl35 = peak_w_close > 0.0 and wc <= peak_w_close * 0.65

    if rule == "D1_Base_X1a":
        if pd.notna(vs30) and float(vs30) < 0.0:
            return True, "M>=2&<30W0"
    elif rule == "D1_Base_X3b":
        if pd.notna(vs30) and float(vs30) < -6.0:
            return True, "M>=2&<30W-6%"
        if tsl35:
            return True, "M>=2&WTSL35%"
    elif rule in ("D1_X1a_WTSL35", "D1_M20_30W0_WTSL35_Reclaim"):
        if pd.notna(vs30) and float(vs30) < 0.0:
            return True, "M>=2&<30W0"
        if tsl35:
            return True, "M>=2&WTSL35%"
    elif rule == "D1_M20_30W0_DD30":
        if pd.notna(vs30) and pd.notna(dd12) and float(vs30) < 0.0 and float(dd12) <= -30.0:
            return True, "M>=2&<30W0&DD30"
        if tsl35:
            return True, "M>=2&WTSL35%"
    elif rule == "D1_M20_30W6_or_HV115DD30":
        if pd.notna(vs30) and float(vs30) < -6.0:
            return True, "M>=2&<30W-6%"
        if pd.notna(hv20) and pd.notna(dd12) and float(hv20) > 115.0 and float(dd12) <= -30.0:
            return True, "M>=2&HV115&DD30"
        if tsl35:
            return True, "M>=2&WTSL35%"
    elif rule == "D1_M20_Bench30W_or_HV115DD30":
        if pd.notna(b_vs30) and float(b_vs30) < 0.0:
            return True, "M>=2&Bench<30W"
        if pd.notna(hv20) and pd.notna(dd12) and float(hv20) > 115.0 and float(dd12) <= -30.0:
            return True, "M>=2&HV115&DD30"
        if tsl35:
            return True, "M>=2&WTSL35%"
    elif rule == "D1_M20_30W0_or_HV120WEMA10":
        if pd.notna(vs30) and float(vs30) < 0.0:
            return True, "M>=2&<30W0"
        if pd.notna(hv20) and pd.notna(w_ema10) and float(hv20) > 120.0 and wc < float(w_ema10):
            return True, "M>=2&HV120&<EMA10"
        if tsl35:
            return True, "M>=2&WTSL35%"
    else:
        raise ValueError(f"Unknown Direction 1 rule: {rule}")
    return False, ""


def run_dir1_backtest(
    data: dict,
    exit_rule: str,
    start_date: str,
    end_date: str,
    entry_col: str = "E2_PrevHigh",
    use_split_50_50: bool = False,
    sl_cooldown_days: int = 10,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> dict:
    """Strictly causal simulation for Direction 1 refined major-cycle exits."""
    daily_df = data["daily"]
    w_df = data["weekly"]
    week_end_daily_pos = data["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = float(initial_capital)
    shares = 0.0
    tranches = 0
    avg_entry_px = 0.0
    entry_date_str = ""
    tranche2_date_str = "-"
    entry_fill_pos = -1
    trade_cash_start = float(initial_capital)
    trade_mode = ""

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_entry_mode = ""
    cooldown_until_pos = -1
    last_top_exit_w_pos = -100000

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
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            tranches = 2
            avg_entry_px = o
            entry_date_str = dt_str
            tranche2_date_str = "-"
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_HALF" and tranches == 0:
            trade_cash_start = cash
            cost = 0.5 * cash
            shares = (cost * (1.0 - fee_rate)) / o
            cash -= cost
            tranches = 1
            avg_entry_px = o
            entry_date_str = dt_str
            tranche2_date_str = "-"
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_TOPUP" and tranches == 1 and shares > 0:
            new_shares = (cash * (1.0 - fee_rate)) / o
            tot = shares + new_shares
            avg_entry_px = (shares * avg_entry_px + new_shares * o) / tot
            shares = tot
            cash = 0.0
            tranches = 2
            tranche2_date_str = dt_str
            if c > peak_d_close:
                peak_d_close = c
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            proceeds = shares * o * (1.0 - fee_rate)
            cash += proceeds
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trade_logs.append(
                {
                    "mode": trade_mode,
                    "entry_date": entry_date_str,
                    "tranche2_date": tranche2_date_str,
                    "exit_date": dt_str,
                    "entry_px": round(avg_entry_px, 2),
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                }
            )
            if pending_reason.startswith("SL_"):
                cooldown_until_pos = pos + sl_cooldown_days
            else:
                last_top_exit_w_pos = int(w_df.index.searchsorted(idx[pos], side="left"))
            shares = 0.0
            tranches = 0
            avg_entry_px = 0.0
            just_exited = True
        pending_action = ""
        pending_reason = ""
        pending_entry_mode = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end:
            continue

        is_w_end = pos in week_end_daily_pos
        if not is_w_end:
            continue
        w_pos = week_end_daily_pos[pos]
        w_row = w_df.iloc[w_pos]

        if shares > 0:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            m_fish_val = w_row["M_Fish_prev"]
            unmatured = (pd.isna(m_fish_val) or float(m_fish_val) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
            else:
                hit_b, reason_b = check_dir1_exit(exit_rule, w_row, peak_w_close) if pos > entry_fill_pos else (False, "")
                if hit_b:
                    pending_action = "SELL"
                    pending_reason = reason_b
                elif use_split_50_50 and tranches == 1 and bool(w_row["E1_C7Vol_EMA10"]):
                    pending_action = "BUY_TOPUP"

        elif not just_exited and pos >= cooldown_until_pos:
            if use_split_50_50:
                if bool(w_row["E1_C7Vol_EMA10"]):
                    pending_action = "BUY_FULL"
                    pending_entry_mode = "BOT_E1"
                elif bool(w_row["E2_PrevHigh"]):
                    pending_action = "BUY_HALF"
                    pending_entry_mode = "BOT_SPLIT"
            elif bool(w_row[entry_col]):
                pending_action = "BUY_FULL"
                pending_entry_mode = "BOT_E2"

            if pending_action == "" and exit_rule == "D1_M20_30W0_WTSL35_Reclaim":
                if 1 <= (w_pos - last_top_exit_w_pos) <= 12:
                    vs30 = w_row["Close_vs_30W%"]
                    b_vs30 = w_row["Bench_vs_30W%"]
                    ema10 = w_row["W_EMA10"]
                    if (
                        pd.notna(vs30)
                        and pd.notna(b_vs30)
                        and pd.notna(ema10)
                        and float(vs30) > 5.0
                        and float(b_vs30) > 3.0
                        and float(w_row["Close"]) > float(ema10)
                    ):
                        pending_action = "BUY_FULL"
                        pending_entry_mode = "RECLAIM"
                        last_top_exit_w_pos = -100000  # consume reclaim window

    if shares > 0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trade_logs.append(
            {
                "mode": trade_mode,
                "entry_date": entry_date_str,
                "tranche2_date": tranche2_date_str,
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "entry_px": round(avg_entry_px, 2),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
            }
        )

    res = compute_metrics(equity, trade_logs, days_invested, idx[i_start : i_end + 1], initial_capital)
    tag = f"{exit_rule}" + ("_Split50" if use_split_50_50 else "")
    res.update({"System": tag, "trade_logs": trade_logs})
    return res


def run_hybrid_backtest(
    data: dict,
    name: str,
    start_date: str,
    end_date: str,
    bot_entry_col: str = "E2_PrevHigh",
    use_sl_w25: bool = True,
    trend_mode: str = "SMA200",  # "SMA200" or "EMA150"
    hv_mode: str = "FIXED_115",  # "FIXED_115", "FIXED_130", "ADAPT_115_140", "NONE"
    hv_reentry: float = 95.0,
    trend_tsl: float | None = None,
    use_m20_wtsl35: bool = False,
    enable_alt: bool = True,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> dict:
    """Strictly causal Hybrid Dual-Engine simulator."""
    daily_df = data["daily"]
    w_df = data["weekly"]
    week_end_daily_pos = data["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)
    hv20 = daily_df["HV20"].to_numpy(dtype=np.float64)
    b_close = daily_df["Bench_Close"].to_numpy(dtype=np.float64)
    b_sma200 = daily_df["Bench_SMA200"].to_numpy(dtype=np.float64)
    b_ema150 = daily_df["Bench_EMA150"].to_numpy(dtype=np.float64)
    m_fish_ffill = daily_df["M_Fish_prev_ffill"].to_numpy(dtype=np.float64)

    cash = float(initial_capital)
    shares = 0.0
    pos_mode = 0  # 0=FLAT, 1=TREND, 2=ALT, 3=CYCLE_BOT
    initial_mode_str = ""
    promoted_in_trade = False
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_reason = "INIT"
    peak_d_close = 0.0
    peak_w_close = 0.0

    pending = 0  # 0=none, 1=TREND, 2=ALT, 3=CYCLE_BOT, -1=SELL
    pending_reason = ""

    equity = np.empty(n_bt, dtype=np.float64)
    trade_logs: list[dict] = []
    days_invested = 0

    for step, i in enumerate(range(i_start, i_end + 1)):
        o = opens[i]
        c = closes[i]

        just_exited = False
        if pending in (1, 2, 3) and pos_mode == 0:
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            pos_mode = pending
            initial_mode_str = {1: "TREND", 2: "ALT", 3: "BOT"}[pending]
            promoted_in_trade = False
            entry_px = o
            entry_idx = i
            peak_d_close = c
            peak_w_close = 0.0
        elif pending == -1 and pos_mode != 0:
            if i <= entry_idx:
                raise AssertionError(f"Same-day buy-sell on {idx[i]}")
            proceeds = shares * o * (1.0 - fee_rate)
            ret = (o * (1.0 - fee_rate)) / (entry_px * (1.0 + fee_rate)) - 1.0
            mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
            trade_logs.append(
                {
                    "mode": mode_label,
                    "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(ret * 100.0, 1),
                    "exit_reason": pending_reason,
                }
            )
            cash = proceeds
            shares = 0.0
            pos_mode = 0
            exit_idx = i
            last_exit_reason = pending_reason
            just_exited = True
        pending = 0
        pending_reason = ""

        equity[step] = cash + shares * c
        if pos_mode != 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if i >= i_end:
            continue

        is_w_end = i in week_end_daily_pos
        w_row = w_df.iloc[week_end_daily_pos[i]] if is_w_end else None

        # Pre-compute trend entry / exit / vol conditions at day i
        if trend_mode == "SMA200":
            tb_raw = (not np.isnan(b_sma200[i])) and (b_close[i] > b_sma200[i] * 1.01)
            ma_break = (not np.isnan(b_sma200[i])) and (b_close[i] < b_sma200[i] * 0.99)
        else:
            tb_raw = (not np.isnan(b_ema150[i])) and (b_close[i] > b_ema150[i] * 1.02)
            ma_break = (not np.isnan(b_ema150[i])) and (b_close[i] < b_ema150[i] * 0.98)

        hv = hv20[i]
        mf = m_fish_ffill[i]
        if hv_mode == "FIXED_115":
            vol_exit = (not np.isnan(hv)) and (hv > 115.0)
            tb_def = tb_raw and (not np.isnan(hv)) and (hv <= 115.0 * 0.92)
        elif hv_mode == "FIXED_130":
            vol_exit = (not np.isnan(hv)) and (hv > 130.0)
            tb_def = tb_raw and (not np.isnan(hv)) and (hv <= 130.0 * 0.92)
        elif hv_mode == "ADAPT_115_140":
            high_mf = (not np.isnan(mf)) and (mf >= 1.5)
            vol_exit = (not np.isnan(hv)) and ((high_mf and hv > 115.0) or (hv > 140.0))
            tb_def = tb_raw and (not np.isnan(hv)) and (hv <= 115.0 * 0.92)
        elif hv_mode == "NONE":
            vol_exit = False
            tb_def = tb_raw
        else:
            raise ValueError(f"Unknown hv_mode: {hv_mode}")

        tb_after_vol = tb_raw and (not np.isnan(hv)) and (hv <= hv_reentry) and (not vol_exit)

        if pos_mode == 3:
            # CYCLE_BOT mode: update weekly peak, check SL_W25 first, else promote seamlessly to TREND when healthy
            if is_w_end and w_row is not None:
                wc = float(w_row["Close"])
                if wc > peak_w_close:
                    peak_w_close = wc

            hit_sl = False
            if use_sl_w25 and is_w_end and w_row is not None:
                m_w = w_row["M_Fish_prev"]
                unmatured = (pd.isna(m_w) or float(m_w) < 1.5) and (peak_d_close < entry_px * 1.50)
                if unmatured and float(w_row["Close"]) < entry_px * 0.75:
                    hit_sl = True

            if hit_sl:
                pending = -1
                pending_reason = "SL_W25"
            elif use_m20_wtsl35 and is_w_end and w_row is not None:
                m_w = w_row["M_Fish_prev"]
                if pd.notna(m_w) and float(m_w) >= 2.0 and peak_w_close > 0.0 and float(w_row["Close"]) <= peak_w_close * 0.65:
                    pending = -1
                    pending_reason = "M>=2&WTSL35"

            # Seamless promotion to TREND (only if not exiting and Bench > threshold and not vol_exit)
            if pending == 0 and tb_raw and (not vol_exit) and (not ma_break):
                pos_mode = 1
                promoted_in_trade = True

        elif pos_mode == 1:
            if is_w_end and w_row is not None:
                wc = float(w_row["Close"])
                if wc > peak_w_close:
                    peak_w_close = wc

            if trend_tsl is not None and c <= peak_d_close * (1.0 - trend_tsl):
                pending = -1
                pending_reason = "TREND_TSL"
            elif vol_exit:
                pending = -1
                pending_reason = "VOL_SPIKE"
            elif ma_break:
                pending = -1
                pending_reason = "TREND_BREAK"
            elif use_m20_wtsl35 and is_w_end and w_row is not None and i > entry_idx:
                m_w = w_row["M_Fish_prev"]
                if pd.notna(m_w) and float(m_w) >= 2.0 and peak_w_close > 0.0 and float(w_row["Close"]) <= peak_w_close * 0.65:
                    pending = -1
                    pending_reason = "M>=2&WTSL35"

        elif pos_mode == 2:
            if c <= entry_px * 0.80:
                pending = -1
                pending_reason = "ALT_FSL"
            elif c >= entry_px * 1.25 or (i - entry_idx) >= 40:
                pending = -1
                pending_reason = "ALT_TP"

        else:
            # FLAT entry checks (strictly do not trigger on the exact exit fill day to prevent same-day re-eval whipsaw)
            if not just_exited:
                elapsed = i - exit_idx
                bot_sig = bool(is_w_end and w_row is not None and bool(w_row[bot_entry_col]))
                alt_sig = bool(enable_alt and is_w_end and w_row is not None and bool(w_row["ALT_Buy_S1"]))

                if last_exit_reason == "SL_W25":
                    if elapsed >= 10 and bot_sig:
                        pending = 3
                else:
                    # Major bottom CYCLE_BOT is allowed whenever elapsed >= 10 after ALT exits, or immediately after VOL_SPIKE/TREND exits
                    bot_allowed = (elapsed >= 10) if last_exit_reason in ("ALT_FSL", "ALT_TP") else True
                    if last_exit_reason in ("INIT", "TREND_BREAK", "TREND_TSL", "M>=2&WTSL35"):
                        if tb_def:
                            pending = 1
                        elif bot_allowed and bot_sig:
                            pending = 3
                        elif alt_sig:
                            pending = 2
                    elif last_exit_reason == "VOL_SPIKE":
                        if tb_after_vol:
                            pending = 1
                        elif bot_allowed and bot_sig:
                            pending = 3
                        elif alt_sig:
                            pending = 2
                    elif last_exit_reason in ("ALT_FSL", "ALT_TP"):
                        if elapsed >= 10:
                            if tb_def:
                                pending = 1
                            elif bot_sig:
                                pending = 3
                            elif alt_sig:
                                pending = 2

    if pos_mode != 0:
        ret = closes[i_end] / (entry_px * (1.0 + fee_rate)) - 1.0
        mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
        trade_logs.append(
            {
                "mode": mode_label,
                "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "entry_px": round(entry_px, 2),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(ret * 100.0, 1),
                "exit_reason": "OPEN_END",
            }
        )

    res = compute_metrics(equity, trade_logs, days_invested, idx[i_start : i_end + 1], initial_capital)
    res.update({"System": name, "trade_logs": trade_logs})
    return res


def format_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["System", "Final", "Ret%", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "Exposure", "WorstYear"]
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows])


def main() -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 20)

    soxl = extend_causal_data("SOXL", "SMH")
    tqqq = extend_causal_data("TQQQ", "QQQ")

    # =========================================================================
    # 0. BASELINE PARITY CHECK VS AGENTS.md
    # =========================================================================
    s1_soxl_11 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2025-12-31")
    s1_soxl_12 = run_strategy1_baseline("SOXL", "SMH", "2015-01-01", "2026-11-19")
    s1_soxl_26 = run_strategy1_baseline("SOXL", "SMH", "2026-01-02", "2026-11-19")
    s1_soxl_early = run_strategy1_baseline("SOXL", "SMH", "2011-01-03", "2014-12-31")
    s1_tqqq_11 = run_strategy1_baseline("TQQQ", "QQQ", "2015-01-01", "2025-12-31")
    s1_tqqq_12 = run_strategy1_baseline("TQQQ", "QQQ", "2015-01-01", "2026-11-19")

    print("=== STRATEGY 1 BASELINE PARITY CHECK (VS AGENTS.md) ===")
    print("SOXL 2015-2025:", s1_soxl_11["Final"], "Trades:", s1_soxl_11["Trades"])
    print("SOXL 2015-2026:", s1_soxl_12["Final"], "Trades:", s1_soxl_12["Trades"])
    print("SOXL 2026 OOS :", s1_soxl_26["Final"], "Trades:", s1_soxl_26["Trades"])
    print("SOXL 2011-2014:", s1_soxl_early["Final"], "Trades:", s1_soxl_early["Trades"])
    print("TQQQ 2015-2025:", s1_tqqq_11["Final"], "Trades:", s1_tqqq_11["Trades"])
    print("TQQQ 2015-2026:", s1_tqqq_12["Final"], "Trades:", s1_tqqq_12["Trades"])

    assert abs(s1_soxl_11["Final"] - 14141371.30) < 1.0, "S1 11yr parity mismatch"
    assert abs(s1_soxl_12["Final"] - 29633196.29) < 1.0, "S1 12yr parity mismatch"
    assert abs(s1_tqqq_11["Final"] - 788796.44) < 1.0, "S1 TQQQ 11yr parity mismatch"
    print("AGENTS.md parity assertions PASSED.\n")

    # =========================================================================
    # 1. DIRECTION 1: REFINED PURE MAJOR-CYCLE LOW-FREQUENCY EXITS
    # =========================================================================
    dir1_specs = [
        ("D1_Base_X1a", False),
        ("D1_Base_X3b", False),
        ("D1_X1a_WTSL35", False),
        ("D1_M20_30W0_DD30", False),
        ("D1_M20_30W6_or_HV115DD30", False),
        ("D1_M20_Bench30W_or_HV115DD30", False),
        ("D1_M20_30W0_or_HV120WEMA10", False),
        ("D1_M20_30W0_WTSL35_Reclaim", False),
        ("D1_X1a_WTSL35", True),
        ("D1_M20_30W6_or_HV115DD30", True),
        ("D1_M20_Bench30W_or_HV115DD30", True),
    ]

    windows_d1 = [
        ("SOXL 2015-2025 (11yr)", "SOXL", "SMH", soxl, "2015-01-01", "2025-12-31", s1_soxl_11),
        ("SOXL 2015-2026 (12yr)", "SOXL", "SMH", soxl, "2015-01-01", "2026-11-19", s1_soxl_12),
        ("SOXL 2026 OOS", "SOXL", "SMH", soxl, "2026-01-02", "2026-11-19", s1_soxl_26),
        ("TQQQ 2015-2025 (11yr)", "TQQQ", "QQQ", tqqq, "2015-01-01", "2025-12-31", s1_tqqq_11),
        ("TQQQ 2015-2026 (12yr)", "TQQQ", "QQQ", tqqq, "2015-01-01", "2026-11-19", s1_tqqq_12),
    ]

    print("=" * 110)
    print("DIRECTION 1: REFINED PURE MAJOR-CYCLE EXITS (E2_PrevHigh + SL_W25)")
    print("=" * 110)
    d1_res_map: dict[str, dict[str, dict]] = {}
    for label, tk, bk, ds, s, e, s1_r in windows_d1:
        bh = run_buy_and_hold(ds, s, e)
        bh["System"] = f"Buy&Hold ({tk})"
        rows = [bh, s1_r]
        d1_res_map[label] = {}
        for rule, split_flag in dir1_specs:
            r = run_dir1_backtest(ds, rule, s, e, entry_col="E2_PrevHigh", use_split_50_50=split_flag)
            rows.append(r)
            d1_res_map[label][r["System"]] = r
        print(f"\n=== {label} ===")
        print(format_table(rows).to_string(index=False))

    # =========================================================================
    # 2. DIRECTION 2: HYBRID DUAL-ENGINE (E2 BOTTOM + STRATEGY 1 DEFENSE)
    # =========================================================================
    hybrid_configs = [
        ("H1_E2_Promote_S1Full", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="FIXED_115", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=True)),
        ("H2_E2_Promote_NoALT", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="FIXED_115", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=False)),
        ("H3_E1_Promote_S1Full", dict(bot_entry_col="E1_C7Vol_EMA10", use_sl_w25=False, trend_mode="SMA200", hv_mode="FIXED_115", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=True)),
        ("H4_E2_Promote_S1_HV130", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="FIXED_130", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=True)),
        ("H5_E2_Promote_S1_AdaptHV", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="ADAPT_115_140", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=True)),
        ("H6_E2_Promote_S1_AdaptHV_NoALT", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="ADAPT_115_140", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=False, enable_alt=False)),
        ("H7_E2_Promote_EMA150_TSL30", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="EMA150", hv_mode="FIXED_115", hv_reentry=95.0, trend_tsl=0.30, use_m20_wtsl35=False, enable_alt=True)),
        ("H8_E2_Promote_NoHV_M20Exit", dict(bot_entry_col="E2_PrevHigh", use_sl_w25=True, trend_mode="SMA200", hv_mode="NONE", hv_reentry=95.0, trend_tsl=None, use_m20_wtsl35=True, enable_alt=False)),
    ]

    windows_d2 = windows_d1 + [("SOXL 2011-2014 (Early OOS)", "SOXL", "SMH", soxl, "2011-01-03", "2014-12-31", s1_soxl_early)]

    print("\n" + "=" * 110)
    print("DIRECTION 2: HYBRID DUAL-ENGINE (Major Bottom E2 -> Promote to TREND + Defense + ALT)")
    print("=" * 110)
    h_res_map: dict[str, dict[str, dict]] = {}
    for label, tk, bk, ds, s, e, s1_r in windows_d2:
        bh = run_buy_and_hold(ds, s, e)
        bh["System"] = f"Buy&Hold ({tk})"
        rows = [bh, s1_r]
        h_res_map[label] = {}
        for hname, kwargs in hybrid_configs:
            r = run_hybrid_backtest(ds, hname, s, e, **kwargs)
            rows.append(r)
            h_res_map[label][hname] = r
        print(f"\n=== {label} ===")
        print(format_table(rows).to_string(index=False))

    # Pick Top Direction 1 & Top Direction 2 systems for Yearly Comparison & Trade Logs
    print("\n" + "=" * 110)
    print("TOP WINNERS: YEARLY RETURNS & FULL TRADE LOGS (2015-2026)")
    print("=" * 110)

    # Select best D1 by combined Calmar + CAGR across SOXL 11yr, SOXL 12yr, TQQQ 12yr
    d1_names = [f"{r}{'_Split50' if sp else ''}" for r, sp in dir1_specs]
    d1_scored = []
    for nm in d1_names:
        s11 = d1_res_map["SOXL 2015-2025 (11yr)"][nm]
        s12 = d1_res_map["SOXL 2015-2026 (12yr)"][nm]
        t12 = d1_res_map["TQQQ 2015-2026 (12yr)"][nm]
        sc = (s11["Calmar"] + s12["Calmar"] + t12["Calmar"]) * 10.0 + s11["CAGR"] * 0.2 + s12["CAGR"] * 0.2 + t12["CAGR"] * 0.3
        d1_scored.append((sc, nm))
    d1_scored.sort(reverse=True)
    best_d1 = d1_scored[0][1]
    second_d1 = d1_scored[1][1]

    h_scored = []
    for hname, _ in hybrid_configs:
        s11 = h_res_map["SOXL 2015-2025 (11yr)"][hname]
        s12 = h_res_map["SOXL 2015-2026 (12yr)"][hname]
        t12 = h_res_map["TQQQ 2015-2026 (12yr)"][hname]
        sc = (s11["Calmar"] + s12["Calmar"] + t12["Calmar"]) * 10.0 + s11["CAGR"] * 0.2 + s12["CAGR"] * 0.2 + t12["CAGR"] * 0.3
        h_scored.append((sc, hname))
    h_scored.sort(reverse=True)
    best_h = h_scored[0][1]
    second_h = h_scored[1][1]

    print(f"Top Direction 1: {best_d1}, {second_d1}")
    print(f"Top Direction 2: {best_h}, {second_h}, H1_E2_Promote_S1Full")

    for asset_lbl, d1_key, h_key, s1_12 in [
        ("SOXL 2015-2026", "SOXL 2015-2026 (12yr)", "SOXL 2015-2026 (12yr)", s1_soxl_12),
        ("TQQQ 2015-2026", "TQQQ 2015-2026 (12yr)", "TQQQ 2015-2026 (12yr)", s1_tqqq_12),
    ]:
        picks = [
            s1_12,
            d1_res_map[d1_key][best_d1],
            d1_res_map[d1_key][second_d1],
            h_res_map[h_key][best_h],
            h_res_map[h_key]["H1_E2_Promote_S1Full"],
        ]
        if second_h not in (best_h, "H1_E2_Promote_S1Full"):
            picks.append(h_res_map[h_key][second_h])

        yr_df = pd.DataFrame({p["System"]: {y: round(v, 1) for y, v in p["yearly_rets"].items()} for p in picks}).T
        print(f"\n=== {asset_lbl} Yearly Returns (%) ===")
        print(yr_df.to_string())

        for p in picks[1:]:
            print(
                f"\n--- {asset_lbl} | {p['System']} | Final=${p['Final']:,.2f} | CAGR={p['CAGR']}% | MaxDD={p['MaxDD']}% | Calmar={p['Calmar']} | Trades={p['Trades']} | WinRate={p['WinRate']}% ---"
            )
            print(pd.DataFrame(p["trade_logs"]).to_string(index=False))


if __name__ == "__main__":
    main()
