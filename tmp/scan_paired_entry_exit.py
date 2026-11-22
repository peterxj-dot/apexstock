"""Test paired entry-exit architectures (distinct exits per entry mode + upgrade) on SOXL (2015-2025)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import rsi
from scan_soxl_strategies import (
    END_DATE,
    FEE_RATE,
    INITIAL_CAPITAL,
    START_DATE,
    compute_asset_features,
    load_adjusted,
)


def simulate_paired(
    opens: np.ndarray,
    closes: np.ndarray,
    trend_buy: np.ndarray,
    trend_sell: np.ndarray,
    alt_buy: np.ndarray | None,
    alt_sell: np.ndarray | None,
    upgrade_cond: np.ndarray | None,
    year_end_indices: list[tuple[int, int]],
    dates_index: pd.DatetimeIndex,
    trend_fsl: float | None = None,
    trend_tsl: float | None = None,
    alt_fsl: float | None = None,
    alt_tsl: float | None = None,
    alt_tp: float | None = None,
    alt_max_days: int | None = None,
    allow_upgrade: bool = False,
    trend_min_hold: int = 0,
    trend_cooldown: int = 0,
    alt_cooldown: int = 0,
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
    record_details: bool = False,
) -> dict:
    """State-machine causal simulator: mode 0=FLAT, 1=TREND, 2=ALT (Bottom/Pullback)."""
    n = len(closes)
    cash = initial_capital
    shares = 0.0
    mode = 0  # 0=FLAT, 1=TREND, 2=ALT
    initial_mode = 0
    pending = 0  # 0=none, 1=BUY TREND, 2=BUY ALT, -1=SELL
    pending_reason = ""
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_mode = 0
    peak_close = 0.0
    upgraded_flag = False

    equity = np.empty(n, dtype=np.float64)
    trade_rets: list[float] = []
    trade_logs: list[dict] = []
    days_invested = 0

    for i in range(n):
        o = opens[i]
        c = closes[i]

        # 1. Execute pending order at today's Open
        if pending in (1, 2) and mode == 0:
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            mode = pending
            initial_mode = pending
            entry_px = o
            entry_idx = i
            peak_close = c
            upgraded_flag = False
        elif pending == -1 and mode in (1, 2):
            proceeds = shares * o * (1.0 - fee_rate)
            ret = (o * (1.0 - fee_rate)) / (entry_px * (1.0 + fee_rate)) - 1.0
            trade_rets.append(ret)
            if record_details:
                ttype = "TREND" if initial_mode == 1 else ("ALT->TREND" if upgraded_flag else "ALT_ONLY")
                trade_logs.append(
                    {
                        "type": ttype,
                        "entry_date": dates_index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": dates_index[i].strftime("%Y-%m-%d"),
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(o, 2),
                        "ret_pct": round(ret * 100.0, 2),
                        "reason": pending_reason,
                    }
                )
            cash = proceeds
            shares = 0.0
            last_exit_mode = mode
            mode = 0
            exit_idx = i
        pending = 0
        pending_reason = ""

        # 2. Mark-to-market & evaluate state transitions
        if mode == 1:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            if trend_fsl is not None and c <= entry_px * (1.0 - trend_fsl):
                pending = -1
                pending_reason = "TREND_FSL"
            elif trend_tsl is not None and c <= peak_close * (1.0 - trend_tsl):
                pending = -1
                pending_reason = "TREND_TSL"
            elif (i - entry_idx) >= trend_min_hold and trend_sell[i]:
                pending = -1
                pending_reason = "TREND_SIG"

        elif mode == 2:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            if alt_fsl is not None and c <= entry_px * (1.0 - alt_fsl):
                pending = -1
                pending_reason = "ALT_FSL"
            elif alt_tsl is not None and c <= peak_close * (1.0 - alt_tsl):
                pending = -1
                pending_reason = "ALT_TSL"
            elif allow_upgrade and upgrade_cond is not None and upgrade_cond[i]:
                # Promote ALT position seamlessly to TREND at zero cost
                mode = 1
                upgraded_flag = True
                peak_close = c
            elif alt_sell is not None and alt_sell[i]:
                pending = -1
                pending_reason = "ALT_SIG"
            elif alt_tp is not None and c >= entry_px * (1.0 + alt_tp):
                pending = -1
                pending_reason = "ALT_TP"
            elif alt_max_days is not None and (i - entry_idx) >= alt_max_days:
                pending = -1
                pending_reason = "ALT_TIME"

        else:
            equity[i] = cash
            req_cd = trend_cooldown if last_exit_mode == 1 else (alt_cooldown if last_exit_mode == 2 else 0)
            if (i - exit_idx) >= req_cd:
                if trend_buy[i]:
                    pending = 1
                elif alt_buy is not None and alt_buy[i]:
                    pending = 2

    if mode != 0:
        ret = closes[-1] / (entry_px * (1.0 + fee_rate)) - 1.0
        trade_rets.append(ret)
        if record_details:
            ttype = "TREND" if initial_mode == 1 else ("ALT->TREND" if upgraded_flag else "ALT_ONLY")
            trade_logs.append(
                {
                    "type": ttype,
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
    soxl_f["D_RSI14"] = rsi(soxl_f["daily"]["Close"], 14)

    bt_mask = (full_idx >= START_DATE) & (full_idx <= END_DATE)
    bt_idx = full_idx[bt_mask]
    n_bt = len(bt_idx)

    def sl(obj) -> np.ndarray:
        return obj.to_numpy()[bt_mask]

    opens = sl(soxl_f["daily"]["Open"]).astype(np.float64)
    closes = sl(soxl_f["daily"]["Close"]).astype(np.float64)
    is_w_end = sl(soxl_f["is_week_end"]).astype(bool)

    yr_series = pd.Series(np.arange(n_bt), index=bt_idx)
    year_end_indices = [(int(yr), int(grp.iloc[-1])) for yr, grp in yr_series.groupby(bt_idx.year)]

    soxl_wc = sl(soxl_f["W_Close"])
    soxl_wf = sl(soxl_f["W_Fisher"])
    soxl_wfs = sl(soxl_f["W_Fisher_Sig"])
    soxl_dwf = sl(soxl_f["dW_Fisher"])
    soxl_wvr = sl(soxl_f["W_Vol_R"])
    soxl_mf = sl(soxl_f["M_Fisher"])
    soxl_hv20 = sl(soxl_f["D_HV20"])
    soxl_d_rsi = sl(soxl_f["D_RSI14"])
    soxl_ws30 = sl(soxl_f["W_SMA30"])

    smh_dc = sl(smh_f["daily"]["Close"])
    smh_ema150 = sl(smh_f["D_EMA150"])
    smh_sma200 = sl(smh_f["D_SMA200"])

    results: list[dict] = []
    replay_specs: dict[str, tuple] = {}

    def record(arch: str, name: str, params_str: str, args: tuple, kwargs: dict) -> None:
        m = simulate_paired(*args, **kwargs)
        results.append({"Arch": arch, "Name": name, "Params": params_str, **m})
        replay_specs[f"{arch}|{name}|{params_str}"] = (args, kwargs)

    # -------------------------------------------------------------------------
    # 0. Baselines (Unified Exits)
    # -------------------------------------------------------------------------
    vcap = 115.0
    b_ema150 = (smh_dc > smh_ema150 * 1.02) & (soxl_hv20 <= vcap * 0.92)
    s_ema150 = (smh_dc < smh_ema150 * 0.98) | (soxl_hv20 > vcap)
    record("Baseline", "Base_F2_TrendOnly", "SMH_EMA150_buf0.02_tsl0.20",
           (opens, closes, b_ema150, s_ema150, None, None, None, year_end_indices, bt_idx),
           {"trend_tsl": 0.20})

    b_sma200 = (smh_dc > smh_sma200 * 1.01) & (soxl_hv20 <= vcap * 0.92)
    s_sma200 = (smh_dc < smh_sma200 * 0.99) | (soxl_hv20 > vcap)
    record("Baseline", "Base_F2_SMA200", "SMH_SMA200_buf0.01_tslNone",
           (opens, closes, b_sma200, s_sma200, None, None, None, year_end_indices, bt_idx),
           {})

    f1_bot = (soxl_wf < -1.5) & ((soxl_dwf > 0) | (soxl_wvr > 1.3))
    f1_b = is_w_end & (f1_bot | ((soxl_wc > soxl_ws30) & (soxl_wf > soxl_wfs)))
    f1_s = is_w_end & (
        ((soxl_mf > 2.0) & (soxl_dwf < 0) & (soxl_wc < soxl_ws30 * 1.05))
        | ((soxl_mf > 2.2) & (soxl_dwf < -0.25))
        | ((soxl_wc < soxl_ws30 * 0.95) & (soxl_wf < soxl_wfs) & (soxl_wf < -0.3))
    )
    for cd in (0, 10):
        record("Baseline", "Base_F1_Unified", f"W_SMA30_botTrue_mf2.2_cd{cd}",
               (opens, closes, f1_b, f1_s, None, None, None, year_end_indices, bt_idx),
               {"trend_cooldown": cd})

    # -------------------------------------------------------------------------
    # 1. P1_SMHTrend_Plus_PanicBottom
    # -------------------------------------------------------------------------
    smh_configs = [
        ("EMA150", smh_ema150, 0.02),
        ("SMA200", smh_sma200, 0.01),
    ]
    p1_exit_modes = [
        ("rev_w0.0", is_w_end & (soxl_wf > 0.0), None, None, False),
        ("rev_w0.5", is_w_end & (soxl_wf > 0.5), None, None, False),
        ("rev_w1.0", is_w_end & (soxl_wf > 1.0), None, None, False),
        ("tp0.25_40d", None, 0.25, 40, False),
        ("tp0.40_40d", None, 0.40, 40, False),
        ("tp0.40_65d", None, 0.40, 65, False),
        ("tp0.60_65d", None, 0.60, 65, False),
        ("upg_40d", None, None, 40, True),
        ("upg_65d", None, None, 65, True),
        ("upg_120d", None, None, 120, True),
    ]
    for ma_name, ma_arr, buf in smh_configs:
        tb = (smh_dc > ma_arr * (1.0 + buf)) & (soxl_hv20 <= vcap * 0.92)
        ts = (smh_dc < ma_arr * (1.0 - buf)) | (soxl_hv20 > vcap)
        for wf_th in (-1.5, -1.8):
            ab = is_w_end & (soxl_wf < wf_th) & ((soxl_dwf > 0) | (soxl_wvr > 1.3))
            for ttsl in (None, 0.20, 0.25, 0.30):
                for afsl in (0.12, 0.15, 0.20):
                    for atsl in (None, 0.20):
                        for ex_lbl, a_sell, a_tp, a_md, upg in p1_exit_modes:
                            for acd in (0, 10):
                                pstr = f"{ma_name}(buf={buf}),ttsl={ttsl},wf<{wf_th},afsl={afsl},atsl={atsl},{ex_lbl},acd={acd}"
                                record(
                                    "P1_SMHTrend_PanicBot",
                                    "P1_Paired",
                                    pstr,
                                    (opens, closes, tb, ts, ab, a_sell, tb if upg else None, year_end_indices, bt_idx),
                                    {
                                        "trend_tsl": ttsl,
                                        "alt_fsl": afsl,
                                        "alt_tsl": atsl,
                                        "alt_tp": a_tp,
                                        "alt_max_days": a_md,
                                        "allow_upgrade": upg,
                                        "alt_cooldown": acd,
                                    },
                                )

    # -------------------------------------------------------------------------
    # 2. P2_WeeklyFisher_SplitExit (Splitting F1 Trend vs Panic Bottom Exits)
    # -------------------------------------------------------------------------
    p2_tb = is_w_end & (soxl_wc > soxl_ws30) & (soxl_wf > soxl_wfs)
    p2_upg = is_w_end & (soxl_wc > soxl_ws30)
    p2_exit_modes = [
        ("rev_w0.5", is_w_end & (soxl_wf > 0.5), None, None, False),
        ("tp0.30_30d", None, 0.30, 30, False),
        ("tp0.30_60d", None, 0.30, 60, False),
        ("tp0.50_60d", None, 0.50, 60, False),
        ("upg_30d", None, None, 30, True),
        ("upg_60d", None, None, 60, True),
        ("upg_120d", None, None, 120, True),
    ]
    for mf_top in (2.2, 99.0):
        p2_ts = is_w_end & (
            ((soxl_mf > 2.0) & (soxl_dwf < 0) & (soxl_wc < soxl_ws30 * 1.05))
            | ((soxl_mf > mf_top) & (soxl_dwf < -0.25))
            | ((soxl_wc < soxl_ws30 * 0.95) & (soxl_wf < soxl_wfs) & (soxl_wf < -0.3))
        )
        for tfsl, ttsl in ((None, None), (0.20, None), (None, 0.25), (None, 0.30)):
            for wf_th in (-1.5, -1.8):
                ab = is_w_end & (soxl_wf < wf_th) & ((soxl_dwf > 0) | (soxl_wvr > 1.3))
                for afsl in (0.12, 0.15, 0.18):
                    for atsl in (None, 0.18):
                        for ex_lbl, a_sell, a_tp, a_md, upg in p2_exit_modes:
                            for acd in (0, 10):
                                pstr = f"W_SMA30,mf={mf_top},tfsl={tfsl},ttsl={ttsl},wf<{wf_th},afsl={afsl},atsl={atsl},{ex_lbl},acd={acd}"
                                record(
                                    "P2_WeeklyFisher_Split",
                                    "P2_Paired",
                                    pstr,
                                    (opens, closes, p2_tb, p2_ts, ab, a_sell, p2_upg if upg else None, year_end_indices, bt_idx),
                                    {
                                        "trend_fsl": tfsl,
                                        "trend_tsl": ttsl,
                                        "alt_fsl": afsl,
                                        "alt_tsl": atsl,
                                        "alt_tp": a_tp,
                                        "alt_max_days": a_md,
                                        "allow_upgrade": upg,
                                        "alt_cooldown": acd,
                                    },
                                )

    # -------------------------------------------------------------------------
    # 3. P3_SMHTrend_Plus_BullPullback (Daily RSI Oversold Pullbacks)
    # -------------------------------------------------------------------------
    for ma_name, ma_arr, buf in smh_configs:
        tb = (smh_dc > ma_arr * (1.0 + buf)) & (soxl_hv20 <= vcap * 0.92)
        ts = (smh_dc < ma_arr * (1.0 - buf)) | (soxl_hv20 > vcap)
        for ttsl in (None, 0.20, 0.25):
            for rsi_in in (30, 35):
                ab = (smh_dc > smh_sma200) & (soxl_d_rsi < rsi_in)
                for rsi_out in (60, 70):
                    a_sell = (soxl_d_rsi > rsi_out) | (smh_dc < smh_sma200 * 0.98)
                    for afsl in (0.12, 0.15):
                        for upg in (False, True):
                            for acd in (0, 10):
                                pstr = f"{ma_name},ttsl={ttsl},rsi<{rsi_in}->rsi>{rsi_out},afsl={afsl},upg={upg},acd={acd}"
                                record(
                                    "P3_SMH_BullPullback",
                                    "P3_Paired",
                                    pstr,
                                    (opens, closes, tb, ts, ab, a_sell, tb if upg else None, year_end_indices, bt_idx),
                                    {
                                        "trend_tsl": ttsl,
                                        "alt_fsl": afsl,
                                        "allow_upgrade": upg,
                                        "alt_cooldown": acd,
                                    },
                                )

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(results).sort_values(["Calmar", "CAGR"], ascending=[False, False]).reset_index(drop=True)
    out_csv = ROOT / "tmp" / "scan_paired_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"Scanned {len(df):,} paired + baseline configurations in {elapsed:.2f}s -> saved {out_csv}\n")

    cols = ["Arch", "Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Sortino", "Trades", "WinRate", "Exposure", "WorstYear"]

    print("=== 1. Unified-Exit Baselines ===")
    print(df[df["Arch"] == "Baseline"][cols].to_string(index=False))

    print("\n=== 2. Top 15 Paired Strategies by Calmar Ratio (Trades >= 8, CAGR >= 40%) ===")
    top15 = df[(df["Arch"] != "Baseline") & (df["Trades"] >= 8) & (df["CAGR"] >= 40.0)].head(15)
    print(top15[cols].to_string(index=False))

    print("\n=== 3. Best Strategy per Architecture (by Calmar, Trades >= 8) ===")
    best_arch = df[(df["Arch"] != "Baseline") & (df["Trades"] >= 8)].groupby("Arch", as_index=False).first().sort_values("Calmar", ascending=False)
    print(best_arch[cols].to_string(index=False))

    print("\n=== 4. Best Paired Strategies Under MaxDD > -45% and MaxDD > -50% ===")
    for lbl, fl in [("MaxDD > -45%", -45.0), ("MaxDD > -50%", -50.0)]:
        sub = df[(df["Arch"] != "Baseline") & (df["Trades"] >= 8) & (df["MaxDD"] > fl)].head(3)
        print(f"-- {lbl} --")
        print(sub[cols].to_string(index=False) if not sub.empty else "  (None)")

    # 5. Apples-to-Apples Ablation on #1 P1_SMHTrend_PanicBot
    p1_best = df[df["Arch"] == "P1_SMHTrend_PanicBot"].iloc[0]
    p1_key = f"{p1_best['Arch']}|{p1_best['Name']}|{p1_best['Params']}"
    p1_args, p1_kw = replay_specs[p1_key]
    opens_a, closes_a, tb_a, ts_a, ab_a, a_sell_a, upg_a, yei_a, bt_idx_a = p1_args

    # Same TREND parent only
    parent_only = simulate_paired(
        opens_a, closes_a, tb_a, ts_a, None, None, None, yei_a, bt_idx_a,
        trend_tsl=p1_kw.get("trend_tsl"),
    )
    # Naive unified exit (TREND | BOTTOM sharing TREND exit)
    naive_union = simulate_paired(
        opens_a, closes_a, tb_a | ab_a, ts_a, None, None, None, yei_a, bt_idx_a,
        trend_tsl=p1_kw.get("trend_tsl"),
    )
    # Paired winner with details
    p1_detail = simulate_paired(*p1_args, record_details=True, **p1_kw)

    print("\n=== 5. Apples-to-Apples Ablation on #1 P1 Paired Winner ===")
    ablation_df = pd.DataFrame([
        {"Variant": "1) TREND Only (Unified Parent)", **parent_only},
        {"Variant": "2) Naive Union (TREND|BOT -> Same Trend Exit)", **naive_union},
        {"Variant": "3) Paired Entry-Exit (P1 Winner)", **{k: p1_detail[k] for k in parent_only}},
    ])
    print(ablation_df.to_string(index=False))

    print("\n=== 6. Entry-Mode Attribution & Yearly Returns (#1 P1 Paired Winner) ===")
    print(f"Params: {p1_best['Params']}")
    yr_str = " | ".join(f"{yr}: {val:+.1f}%" for yr, val in p1_detail["yearly_rets"].items())
    print(f"Yearly Returns: {yr_str}")
    tlog = pd.DataFrame(p1_detail["trade_logs"])
    attr = tlog.groupby("type")["ret_pct"].agg(
        Trades="count",
        WinRate_pct=lambda s: round((s > 0).mean() * 100, 1),
        AvgRet_pct=lambda s: round(s.mean(), 1),
        MedRet_pct=lambda s: round(s.median(), 1),
        MinRet_pct=lambda s: round(s.min(), 1),
        MaxRet_pct=lambda s: round(s.max(), 1),
    ).reset_index()
    print(attr.to_string(index=False))
    print("\nALT / ALT->TREND Trades Detail:")
    print(tlog[tlog["type"] != "TREND"].to_string(index=False))

    # 7. Parameter Plateau Check around #1 P1 Paired Winner
    print("\n=== 7. Parameter Plateau Check around #1 P1 Paired Winner ===")
    # Extract key pattern from p1_best["Params"]
    p1_sub = df[df["Arch"] == "P1_SMHTrend_PanicBot"]
    # Compare across both SMH MAs, trend_tsl, wf_th, alt_fsl, and exit mode
    ex_token = [tok for tok in p1_best["Params"].split(",") if tok.startswith(("rev_", "tp", "upg_"))][0]
    plateau = p1_sub[
        p1_sub["Params"].str.contains("wf<-1.5")
        & p1_sub["Params"].str.contains(ex_token)
        & p1_sub["Params"].str.contains("acd=0")
        & p1_sub["Params"].str.contains("atsl=None")
    ][["Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate"]]
    print(plateau.sort_values("Calmar", ascending=False).head(16).to_string(index=False))


if __name__ == "__main__":
    main()
