"""Stage 2 scanner targeting MaxDD < 45% / < 40% / < 35% with high CAGR on SOXL (2015-2025)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, historical_volatility, rsi, sma
from scan_soxl_strategies import (
    END_DATE,
    FEE_RATE,
    INITIAL_CAPITAL,
    START_DATE,
    compute_asset_features,
    load_adjusted,
)


def simulate_enhanced(
    opens: np.ndarray,
    closes: np.ndarray,
    buy_sig: np.ndarray,
    sell_sig: np.ndarray,
    year_end_indices: list[tuple[int, int]],
    dates_index: pd.DatetimeIndex,
    fixed_sl: float | None = None,
    trail_sl: float | None = None,
    profit_trigger: float | None = None,
    profit_trail_sl: float | None = None,
    emergency_sell: np.ndarray | None = None,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
    stop_cooldown_days: int | None = None,
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
    record_details: bool = False,
) -> dict:
    """Strictly causal daily simulator with profit-activated trailing stop & emergency brake."""
    n = len(closes)
    cash = initial_capital
    shares = 0.0
    in_pos = False
    pending = 0  # 0=none, 1=BUY, -1=SELL
    pending_was_stop = False
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_was_stop = False
    peak_close = 0.0

    equity = np.empty(n, dtype=np.float64)
    trade_rets: list[float] = []
    trade_logs: list[dict] = []
    days_invested = 0

    for i in range(n):
        o = opens[i]
        c = closes[i]

        # 1. Execute pending action at today's Open
        if pending == 1 and not in_pos:
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            in_pos = True
            entry_px = o
            entry_idx = i
            peak_close = c
        elif pending == -1 and in_pos:
            proceeds = shares * o * (1.0 - fee_rate)
            ret = (o * (1.0 - fee_rate)) / (entry_px * (1.0 + fee_rate)) - 1.0
            trade_rets.append(ret)
            if record_details:
                trade_logs.append(
                    {
                        "entry_date": dates_index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": dates_index[i].strftime("%Y-%m-%d"),
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(o, 2),
                        "ret_pct": round(ret * 100.0, 2),
                        "reason": "STOP/BRAKE" if pending_was_stop else "SIGNAL",
                    }
                )
            cash = proceeds
            shares = 0.0
            in_pos = False
            exit_idx = i
            last_exit_was_stop = pending_was_stop
        pending = 0
        pending_was_stop = False

        # 2. Mark-to-market at Close
        if in_pos:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            stop_hit = False
            if fixed_sl is not None and c <= entry_px * (1.0 - fixed_sl):
                stop_hit = True
            elif trail_sl is not None and c <= peak_close * (1.0 - trail_sl):
                stop_hit = True
            elif (
                profit_trigger is not None
                and profit_trail_sl is not None
                and peak_close >= entry_px * (1.0 + profit_trigger)
                and c <= peak_close * (1.0 - profit_trail_sl)
            ):
                stop_hit = True
            elif emergency_sell is not None and emergency_sell[i]:
                stop_hit = True

            if stop_hit:
                pending = -1
                pending_was_stop = True
            elif (i - entry_idx) >= min_hold_days and sell_sig[i]:
                pending = -1
                pending_was_stop = False
        else:
            equity[i] = cash
            req_cd = stop_cooldown_days if (last_exit_was_stop and stop_cooldown_days is not None) else cooldown_days
            if (i - exit_idx) >= req_cd and buy_sig[i]:
                pending = 1

    if in_pos:
        ret = closes[-1] / (entry_px * (1.0 + fee_rate)) - 1.0
        trade_rets.append(ret)
        if record_details:
            trade_logs.append(
                {
                    "entry_date": dates_index[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": dates_index[-1].strftime("%Y-%m-%d") + " (Open)",
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(float(closes[-1]), 2),
                    "ret_pct": round(ret * 100.0, 2),
                    "reason": "OPEN",
                }
            )

    final_val = float(equity[-1])
    years_span = n / 252.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0
    cum_max = np.maximum.accumulate(equity)
    max_dd = float(((equity / cum_max) - 1.0).min() * 100.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    daily_ret = np.diff(equity) / equity[:-1]
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
    downside = np.minimum(daily_ret, 0.0)
    d_std = float(np.sqrt(np.mean(downside * downside)))
    sortino = float(daily_ret.mean() / d_std * np.sqrt(252.0)) if d_std > 1e-12 else 0.0

    trades_cnt = len(trade_rets)
    win_rate = (sum(1 for r in trade_rets if r > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = days_invested / n * 100.0

    yearly_rets: dict[int, float] = {}
    prev_eq = initial_capital
    for yr, end_idx in year_end_indices:
        yr_eq = float(equity[end_idx])
        yearly_rets[yr] = (yr_eq / prev_eq - 1.0) * 100.0
        prev_eq = yr_eq
    worst_year = min(yearly_rets.values()) if yearly_rets else 0.0

    res = {
        "Final": round(final_val, 2),
        "CAGR": round(cagr, 2),
        "MaxDD": round(max_dd, 2),
        "Calmar": round(calmar, 3),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Trades": trades_cnt,
        "WinRate": round(win_rate, 1),
        "Exposure": round(exposure, 1),
        "WorstYear": round(worst_year, 1),
    }
    if record_details:
        res["yearly_rets"] = yearly_rets
        res["trade_logs"] = trade_logs
    return res


def main() -> None:
    t0 = time.perf_counter()
    soxl = load_adjusted("SOXL")
    smh = load_adjusted("SMH")

    full_idx = soxl.index
    soxl_f = compute_asset_features(soxl, full_idx)
    smh_f = compute_asset_features(smh, full_idx)

    # Extra daily indicators for Stage 2
    soxl_d = soxl_f["daily"]
    smh_d = smh_f["daily"]
    soxl_f["D_EMA5"] = ema(soxl_d["Close"], 5)
    soxl_f["D_HV10"] = historical_volatility(soxl_d["Close"], period=10, annualization_factor=252)
    soxl_f["D_RSI14"] = rsi(soxl_d["Close"], 14)
    soxl_f["ret5d"] = soxl_d["Close"].pct_change(5)
    soxl_f["ret10d"] = soxl_d["Close"].pct_change(10)
    smh_f["D_EMA5"] = ema(smh_d["Close"], 5)
    smh_f["D_HV10"] = historical_volatility(smh_d["Close"], period=10, annualization_factor=252)

    bt_mask = (full_idx >= START_DATE) & (full_idx <= END_DATE)
    bt_idx = full_idx[bt_mask]
    n_bt = len(bt_idx)

    def sl(obj) -> np.ndarray:
        return obj.to_numpy()[bt_mask]

    opens = sl(soxl_d["Open"]).astype(np.float64)
    closes = sl(soxl_d["Close"]).astype(np.float64)
    is_w_end = sl(soxl_f["is_week_end"]).astype(bool)

    yr_series = pd.Series(np.arange(n_bt), index=bt_idx)
    year_end_indices = [(int(yr), int(grp.iloc[-1])) for yr, grp in yr_series.groupby(bt_idx.year)]

    soxl_wc = sl(soxl_f["W_Close"])
    soxl_wf = sl(soxl_f["W_Fisher"])
    soxl_wfs = sl(soxl_f["W_Fisher_Sig"])
    soxl_dwf = sl(soxl_f["dW_Fisher"])
    soxl_wvr = sl(soxl_f["W_Vol_R"])
    soxl_mf = sl(soxl_f["M_Fisher"])
    soxl_hv10 = sl(soxl_f["D_HV10"])
    soxl_hv20 = sl(soxl_f["D_HV20"])
    soxl_ret5d = sl(soxl_f["ret5d"])
    soxl_d_ema20 = sl(soxl_f["D_EMA20"])

    smh_dc = sl(smh_d["Close"])
    smh_d_ema20 = sl(smh_f["D_EMA20"])
    smh_d_ema30 = sl(smh_f["D_EMA30"])
    smh_d_ema50 = sl(smh_f["D_EMA50"])

    results: list[dict] = []
    replay_specs: dict[str, tuple] = {}

    def record(family: str, name: str, params_str: str, buy_sig: np.ndarray, sell_sig: np.ndarray, sim_kw: dict) -> None:
        m = simulate_enhanced(opens, closes, buy_sig, sell_sig, year_end_indices, bt_idx, **sim_kw)
        row = {"Family": family, "Name": name, "Params": params_str, **m}
        results.append(row)
        replay_specs[f"{family}|{name}|{params_str}"] = (buy_sig, sell_sig, sim_kw)

    # -------------------------------------------------------------------------
    # E1: E1_F1_ProfitLock (F1_FisherVol + Profit-Activated Trailing Stop + Finer Cooldown)
    # -------------------------------------------------------------------------
    e1_bases = [
        (30, True, 99.0, 1.00),
        (30, True, 99.0, 0.95),
        (30, True, 2.2, 1.00),
        (30, False, 99.0, 1.00),
        (20, True, 99.0, 0.95),
        (20, True, 2.2, 0.95),
    ]
    pt_grid = [(None, None), (0.30, 0.15), (0.30, 0.20), (0.50, 0.15), (0.50, 0.20), (0.50, 0.25), (0.80, 0.20), (0.80, 0.25)]
    for sma_w, allow_bot, mf_sell, s3_buf in e1_bases:
        w_sma = sl(soxl_f[f"W_SMA{sma_w}"])
        b_bot = (soxl_wf < -1.5) & ((soxl_dwf > 0) | (soxl_wvr > 1.3)) if allow_bot else np.zeros(n_bt, dtype=bool)
        b_sig = is_w_end & (b_bot | ((soxl_wc > w_sma) & (soxl_wf > soxl_wfs)))
        s_top = ((soxl_mf > 2.0) & (soxl_dwf < 0) & (soxl_wc < w_sma * 1.05)) | ((soxl_mf > mf_sell) & (soxl_dwf < -0.25))
        s_sig = is_w_end & (s_top | ((soxl_wc < w_sma * s3_buf) & (soxl_wf < soxl_wfs) & (soxl_wf < -0.3)))

        for fsl in (None, 0.12, 0.15, 0.20):
            for pt, ptsl in pt_grid:
                for cd in (15, 20, 25, 30, 40, 50):
                    for mh in (0, 10):
                        pstr = f"W_SMA{sma_w},bot={allow_bot},mf={mf_sell},buf={s3_buf},fsl={fsl},pt=({pt},{ptsl}),mh={mh},cd={cd}"
                        record("E1_F1_ProfitLock", "F1_ProfitLock", pstr, b_sig, s_sig, {
                            "fixed_sl": fsl, "profit_trigger": pt, "profit_trail_sl": ptsl,
                            "min_hold_days": mh, "cooldown_days": cd,
                        })

    # -------------------------------------------------------------------------
    # E2: E2_F1_DailyBrake (F1_FisherVol + Fast Daily Crash / Vol / SMH Emergency Brake)
    # -------------------------------------------------------------------------
    emergency_brakes = {
        "none": None,
        "smh_ema20_0.98": smh_dc < smh_d_ema20 * 0.98,
        "smh_ema50_0.98": smh_dc < smh_d_ema50 * 0.98,
        "soxl_ema20_0.95": closes < soxl_d_ema20 * 0.95,
        "soxl_drop5d_15": soxl_ret5d < -0.15,
        "soxl_drop5d_20": soxl_ret5d < -0.20,
        "hv10_120": soxl_hv10 > 120,
        "hv20_110": soxl_hv20 > 110,
    }
    w_sma30 = sl(soxl_f["W_SMA30"])
    for allow_bot in (True, False):
        b_bot = (soxl_wf < -1.5) & ((soxl_dwf > 0) | (soxl_wvr > 1.3)) if allow_bot else np.zeros(n_bt, dtype=bool)
        b_sig = is_w_end & (b_bot | ((soxl_wc > w_sma30) & (soxl_wf > soxl_wfs)))
        for mf_sell in (2.2, 99.0):
            s_top = ((soxl_mf > 2.0) & (soxl_dwf < 0) & (soxl_wc < w_sma30 * 1.05)) | ((soxl_mf > mf_sell) & (soxl_dwf < -0.25))
            s_sig = is_w_end & (s_top | ((soxl_wc < w_sma30) & (soxl_wf < soxl_wfs) & (soxl_wf < -0.3)))
            for em_name, em_arr in emergency_brakes.items():
                for fsl in (None, 0.15, 0.20):
                    for pt, ptsl in ((None, None), (0.40, 0.18), (0.60, 0.20)):
                        for cd in (10, 20, 30, 40):
                            for mh in (0, 10):
                                pstr = f"W_SMA30,bot={allow_bot},mf={mf_sell},em={em_name},fsl={fsl},pt=({pt},{ptsl}),mh={mh},cd={cd}"
                                record("E2_F1_DailyBrake", "F1_DailyBrake", pstr, b_sig, s_sig, {
                                    "fixed_sl": fsl, "profit_trigger": pt, "profit_trail_sl": ptsl,
                                    "emergency_sell": em_arr, "min_hold_days": mh, "cooldown_days": cd,
                                })

    # -------------------------------------------------------------------------
    # E3: E3_FastDailyTrend_Cooldown (Fast Daily EMA Trend + Volatility Cap + Stops)
    # -------------------------------------------------------------------------
    for src_name, src_f, src_c in (("SMH", smh_f, smh_dc), ("SOXL", soxl_f, closes)):
        for fast_n, slow_n in ((10, 30), (10, 50), (20, 50), (20, 100)):
            f_arr = sl(src_f[f"D_EMA{fast_n}"])
            s_arr = sl(src_f[f"D_EMA{slow_n}"])
            b_base = (src_c > f_arr) & (f_arr > s_arr)
            for exit_n in (10, 20):
                ex_arr = sl(src_f[f"D_EMA{exit_n}"])
                for buf in (0.97, 1.00):
                    s_base = src_c < ex_arr * buf
                    for vcap in (None, 100):
                        if vcap is None:
                            b_sig, s_sig = b_base, s_base
                        else:
                            b_sig = b_base & (soxl_hv20 <= vcap * 0.92)
                            s_sig = s_base | (soxl_hv20 > vcap)
                        for fsl, tsl in ((None, None), (0.12, None), (0.15, None), (None, 0.15), (None, 0.20), (None, 0.25)):
                            for mh, cd in ((0, 10), (5, 10), (5, 20), (10, 20), (10, 30)):
                                pstr = f"{src_name}_EMA({fast_n},{slow_n}),ex={exit_n}*{buf},vcap={vcap},fsl={fsl},tsl={tsl},mh={mh},cd={cd}"
                                record("E3_FastDailyTrend", "FastDailyTrend", pstr, b_sig, s_sig, {
                                    "fixed_sl": fsl, "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd,
                                })

    # -------------------------------------------------------------------------
    # E4: E4_WeeklySMH_DailyExit (Clean Weekly SMH Entry + Fast Daily Risk-Off)
    # -------------------------------------------------------------------------
    dip_buy = is_w_end & (soxl_wf < -1.5) & (soxl_dwf > 0)
    smh_wc = sl(smh_f["W_Close"])
    for w_n in (15, 20, 30):
        smh_ws = sl(smh_f[f"W_SMA{w_n}"])
        b_base = is_w_end & (smh_wc > smh_ws)
        for dip in (False, True):
            b_sig = (b_base | dip_buy) if dip else b_base
            for exit_m, ex_arr in ((20, smh_d_ema20), (30, smh_d_ema30), (50, smh_d_ema50)):
                for buf in (0.97, 0.99):
                    s_sig = smh_dc < ex_arr * buf
                    for fsl, tsl in ((None, None), (0.15, None), (None, 0.18), (None, 0.22)):
                        for mh in (0, 10):
                            for cd in (10, 20, 30, 40):
                                pstr = f"SMH_W_SMA{w_n},dip={dip},ex_EMA{exit_m}*{buf},fsl={fsl},tsl={tsl},mh={mh},cd={cd}"
                                record("E4_WeeklySMH_DailyExit", "WeeklySMH_DailyExit", pstr, b_sig, s_sig, {
                                    "fixed_sl": fsl, "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd,
                                })

    elapsed = time.perf_counter() - t0
    df_v2 = pd.DataFrame(results).sort_values(["Calmar", "CAGR"], ascending=[False, False]).reset_index(drop=True)
    out_v2 = ROOT / "tmp" / "scan_results_v2.csv"
    df_v2.to_csv(out_v2, index=False)

    # Combine Stage 1 & Stage 2 for unified Pareto reporting
    s1_csv = ROOT / "tmp" / "scan_results.csv"
    if s1_csv.exists():
        df_s1 = pd.read_csv(s1_csv)
        df_all = pd.concat([df_v2, df_s1], ignore_index=True).sort_values(["Calmar", "CAGR"], ascending=[False, False]).reset_index(drop=True)
    else:
        df_all = df_v2

    print(f"Stage 2 scanned {len(df_v2):,} combinations in {elapsed:.2f}s -> saved {out_v2}")
    print(f"Combined Stage 1 + Stage 2 total: {len(df_all):,} combinations")

    cols = ["Family", "Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Sortino", "Trades", "WinRate", "Exposure", "WorstYear"]

    # 1. Top 15 Overall by Calmar Ratio
    top15 = df_v2[(df_v2["Trades"] >= 5) & (df_v2["CAGR"] >= 30.0)].head(15)
    print("\n=== Stage 2 Top 15 Overall by Calmar Ratio (Trades >= 5, CAGR >= 30%) ===")
    print(top15[cols].to_string(index=False))

    # 2. Complete Pareto Frontier by MaxDD Tier across Combined Stage 1 + Stage 2
    print("\n=== Complete Pareto Frontier by MaxDD Tier (Combined S1+S2, Top 3 Highest CAGR per Tier, Trades >= 5) ===")
    valid_all = df_all[df_all["Trades"] >= 5]
    for tier_label, dd_floor in [
        ("MaxDD > -30%", -30.0),
        ("MaxDD > -35%", -35.0),
        ("MaxDD > -40%", -40.0),
        ("MaxDD > -45%", -45.0),
        ("MaxDD > -50%", -50.0),
    ]:
        sub = valid_all[valid_all["MaxDD"] > dd_floor].sort_values(["CAGR", "Calmar"], ascending=[False, False]).head(3)
        print(f"\n-- {tier_label} --")
        if sub.empty:
            print("  (No strategies in this tier)")
        else:
            print(sub[cols].to_string(index=False))

    # 3. Best Strategy per Stage-2 Family
    print("\n=== Best Strategy per Stage-2 Family (by Calmar, Trades >= 5) ===")
    best_fam = df_v2[df_v2["Trades"] >= 5].groupby("Family", as_index=False).first().sort_values("Calmar", ascending=False)
    print(best_fam[cols].to_string(index=False))

    # 4. Detailed Breakdown + Parameter Plateau of Overall Stage-2 #1 Calmar Winner & Best Low-DD Winner (<40% or <45%)
    v2_valid = df_v2[df_v2["Trades"] >= 5]
    top_calmar = v2_valid.iloc[0]

    low_dd_pool = v2_valid[(v2_valid["MaxDD"] > -40.0) & (v2_valid["CAGR"] >= 25.0)]
    if low_dd_pool.empty:
        low_dd_pool = v2_valid[v2_valid["MaxDD"] > -45.0]
    if low_dd_pool.empty:
        low_dd_pool = v2_valid
    top_low_dd = low_dd_pool.sort_values(["Calmar", "CAGR"], ascending=[False, False]).iloc[0]

    for label, row in [
        ("#1 Stage-2 Calmar Winner", top_calmar),
        ("#1 Low-DD Balanced Winner (MaxDD > -40% / -45%, Highest Calmar)", top_low_dd),
    ]:
        key = f"{row['Family']}|{row['Name']}|{row['Params']}"
        b_sig, s_sig, sim_kw = replay_specs[key]
        detail = simulate_enhanced(opens, closes, b_sig, s_sig, year_end_indices, bt_idx, record_details=True, **sim_kw)
        print(f"\n=== {label} ===")
        print(f"Family: {row['Family']} | Params: {row['Params']}")
        print(
            f"Final: ${detail['Final']:,.2f} | CAGR: {detail['CAGR']}% | MaxDD: {detail['MaxDD']}% | "
            f"Calmar: {detail['Calmar']} | Sharpe: {detail['Sharpe']} | Sortino: {detail['Sortino']} | "
            f"Trades: {detail['Trades']} | WinRate: {detail['WinRate']}% | Exposure: {detail['Exposure']}%"
        )
        yr_str = " | ".join(f"{yr}: {val:+.1f}%" for yr, val in detail["yearly_rets"].items())
        print(f"Yearly Returns: {yr_str}")
        print("Trade Log:")
        print(pd.DataFrame(detail["trade_logs"]).to_string(index=False))

    # 5. Parameter Plateau Check around Stage-2 #1 Calmar Winner
    print("\n=== Parameter Plateau Check around Stage-2 #1 Calmar Winner ===")
    top_key = f"{top_calmar['Family']}|{top_calmar['Name']}|{top_calmar['Params']}"
    tb_sig, ts_sig, t_kw = replay_specs[top_key]
    base_cd = t_kw.get("cooldown_days", 0)
    base_fsl = t_kw.get("fixed_sl")
    base_ptsl = t_kw.get("profit_trail_sl")
    plateau_rows = []
    for cd_delta in (-10, -5, 0, 5, 10):
        for fsl_val in ([None] if base_fsl is None else [round(max(0.05, base_fsl + d), 2) for d in (-0.05, -0.02, 0.0, 0.02, 0.05)]):
            kw_mod = dict(t_kw)
            kw_mod["cooldown_days"] = max(0, base_cd + cd_delta)
            kw_mod["fixed_sl"] = fsl_val
            if base_ptsl is not None and fsl_val == base_fsl:
                for ptsl_d in (-0.05, 0.0, 0.05):
                    kw_p = dict(kw_mod)
                    kw_p["profit_trail_sl"] = round(max(0.05, base_ptsl + ptsl_d), 2)
                    pm = simulate_enhanced(opens, closes, tb_sig, ts_sig, year_end_indices, bt_idx, **kw_p)
                    plateau_rows.append({"cd": kw_p["cooldown_days"], "fsl": fsl_val, "ptsl": kw_p["profit_trail_sl"], **pm})
            else:
                pm = simulate_enhanced(opens, closes, tb_sig, ts_sig, year_end_indices, bt_idx, **kw_mod)
                plateau_rows.append({"cd": kw_mod["cooldown_days"], "fsl": fsl_val, "ptsl": base_ptsl, **pm})
    print(pd.DataFrame(plateau_rows)[["cd", "fsl", "ptsl", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
