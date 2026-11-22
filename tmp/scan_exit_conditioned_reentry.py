"""Test exit-conditioned re-entry strategies on SOXL (2015-2025)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, fisher_transform, sma
from scan_soxl_strategies import (
    END_DATE,
    FEE_RATE,
    INITIAL_CAPITAL,
    START_DATE,
    compute_asset_features,
    load_adjusted,
)


def consecutive_true(cond: pd.Series, days: int) -> pd.Series:
    return cond.rolling(days, min_periods=days).sum() == days


def simulate_exit_conditioned(
    opens: np.ndarray,
    closes: np.ndarray,
    default_trend_buy: np.ndarray,
    vol_exit: np.ndarray,
    ma_break: np.ndarray,
    default_alt_buy: np.ndarray | None,
    year_end_indices: list[tuple[int, int]],
    dates_index: pd.DatetimeIndex,
    trend_tsl: float | None = None,
    alt_fsl: float | None = 0.20,
    alt_tp: float | None = 0.25,
    alt_max_days: int | None = 40,
    default_acd: int = 10,
    # Exit-conditioned overrides: dicts mapping mode_name -> precomputed arrays/params
    after_tsl_spec: tuple = ("default", None, None, 0),
    after_vol_spec: tuple = ("default", None, None, 0),
    after_tb_spec: tuple = ("default", None, None, 0),
    after_fsl_spec: tuple = ("default", None, None, -1),
    after_tp_spec: tuple = ("default", None, None, -1, None, None),
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
    record_details: bool = False,
) -> dict:
    """Causal simulator tracking last_exit_reason to select re-entry strategy."""
    n = len(closes)
    cash = initial_capital
    shares = 0.0
    mode = 0  # 0=FLAT, 1=TREND, 2=ALT
    initial_mode = 0
    pending = 0  # 0=none, 1=BUY TREND, 2=BUY ALT, -1=SELL
    pending_reason = ""
    pending_fast_bridge = False
    in_fast_bridge = False

    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_px = 0.0
    last_exit_reason = "INIT"
    peak_close = 0.0

    equity = np.empty(n, dtype=np.float64)
    trade_rets: list[float] = []
    trade_logs: list[dict] = []
    days_invested = 0

    tsl_name, tsl_tb, tsl_ab, tsl_cd = after_tsl_spec
    vol_name, vol_tb, vol_ab, vol_cd = after_vol_spec
    tb_name, tb_tb, tb_ab, tb_cd = after_tb_spec
    fsl_name, fsl_tb, fsl_ab, fsl_cd = after_fsl_spec
    tp_name, tp_tb, tp_ab, tp_tcd, tp_acd, tp_bridge_exit = after_tp_spec

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
            in_fast_bridge = pending_fast_bridge
        elif pending == -1 and mode in (1, 2):
            proceeds = shares * o * (1.0 - fee_rate)
            ret = (o * (1.0 - fee_rate)) / (entry_px * (1.0 + fee_rate)) - 1.0
            trade_rets.append(ret)
            if record_details:
                trade_logs.append(
                    {
                        "mode": "TREND" if initial_mode == 1 else "ALT",
                        "prev_exit": last_exit_reason,
                        "entry_date": dates_index[entry_idx].strftime("%Y-%m-%d"),
                        "exit_date": dates_index[i].strftime("%Y-%m-%d"),
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(o, 2),
                        "ret_pct": round(ret * 100.0, 2),
                        "exit_reason": pending_reason,
                    }
                )
            cash = proceeds
            shares = 0.0
            mode = 0
            in_fast_bridge = False
            exit_idx = i
            last_exit_px = o
            last_exit_reason = pending_reason
        pending = 0
        pending_reason = ""
        pending_fast_bridge = False

        # 2. Mark-to-market & evaluate state transitions
        if mode == 1:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            if in_fast_bridge and default_trend_buy[i]:
                in_fast_bridge = False

            if trend_tsl is not None and c <= peak_close * (1.0 - trend_tsl):
                pending = -1
                pending_reason = "TREND_TSL"
            elif vol_exit[i]:
                pending = -1
                pending_reason = "VOL_SPIKE"
            elif (tp_bridge_exit[i] if (in_fast_bridge and tp_bridge_exit is not None) else ma_break[i]):
                pending = -1
                pending_reason = "TREND_BREAK"

        elif mode == 2:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            if alt_fsl is not None and c <= entry_px * (1.0 - alt_fsl):
                pending = -1
                pending_reason = "ALT_FSL"
            elif alt_tp is not None and c >= entry_px * (1.0 + alt_tp):
                pending = -1
                pending_reason = "ALT_TP"
            elif alt_max_days is not None and (i - entry_idx) >= alt_max_days:
                pending = -1
                pending_reason = "ALT_TP"

        else:
            equity[i] = cash
            elapsed = i - exit_idx

            if last_exit_reason == "INIT":
                if default_trend_buy[i]:
                    pending = 1
                elif default_alt_buy is not None and default_alt_buy[i]:
                    pending = 2
            elif last_exit_reason == "TREND_TSL":
                if elapsed >= tsl_cd:
                    if tsl_tb[i]:
                        pending = 1
                    elif tsl_ab is not None and tsl_ab[i]:
                        pending = 2
            elif last_exit_reason == "VOL_SPIKE":
                if elapsed >= vol_cd:
                    if vol_tb[i]:
                        pending = 1
                    elif vol_ab is not None and vol_ab[i]:
                        pending = 2
            elif last_exit_reason == "TREND_BREAK":
                if elapsed >= tb_cd:
                    if tb_tb[i]:
                        pending = 1
                    elif tb_ab is not None and tb_ab[i]:
                        pending = 2
            elif last_exit_reason == "ALT_FSL":
                req_cd = default_acd if fsl_cd < 0 else fsl_cd
                if elapsed >= req_cd:
                    if fsl_tb[i]:
                        pending = 1
                    elif fsl_ab is not None and fsl_ab[i]:
                        pending = 2
            elif last_exit_reason == "ALT_TP":
                req_tcd = default_acd if tp_tcd < 0 else tp_tcd
                req_acd = default_acd if tp_acd < 0 else tp_acd
                if elapsed >= req_tcd and tp_tb[i]:
                    pending = 1
                    if tp_bridge_exit is not None and not default_trend_buy[i]:
                        pending_fast_bridge = True
                elif tp_ab is not None and elapsed >= req_acd:
                    if tp_name == "pullback_10pct":
                        pb_hit = (c <= last_exit_px * 0.90) and bool(tp_ab[i])
                        if (default_alt_buy is not None and default_alt_buy[i]) or pb_hit:
                            pending = 2
                    elif tp_ab[i]:
                        pending = 2

    if mode != 0:
        ret = closes[-1] / (entry_px * (1.0 + fee_rate)) - 1.0
        trade_rets.append(ret)
        if record_details:
            trade_logs.append(
                {
                    "mode": "TREND" if initial_mode == 1 else "ALT",
                    "prev_exit": last_exit_reason,
                    "entry_date": dates_index[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": dates_index[-1].strftime("%Y-%m-%d") + " (Open)",
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(float(closes[-1]), 2),
                    "ret_pct": round(ret * 100.0, 2),
                    "exit_reason": "OPEN",
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

    soxl_d = soxl_f["daily"]
    smh_d = smh_f["daily"]
    soxl_hv20_fall = soxl_f["D_HV20"] < soxl_f["D_HV20"].shift(5)

    # Extra EMAs
    soxl_ema10_s = ema(soxl_d["Close"], 10)
    soxl_ema20_s = soxl_f["D_EMA20"]
    smh_ema10_s = ema(smh_d["Close"], 10)
    smh_ema20_s = smh_f["D_EMA20"]
    smh_ema50_s = smh_f["D_EMA50"]

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

    soxl_ema10 = sl(soxl_ema10_s)
    soxl_ema20 = sl(soxl_ema20_s)
    soxl_df = sl(soxl_f["D_Fisher"])
    soxl_dfs = sl(soxl_f["D_Fisher_Sig"])
    soxl_wc = sl(soxl_f["W_Close"])
    soxl_ws20 = sl(soxl_f["W_SMA20"])
    soxl_wf = sl(soxl_f["W_Fisher"])
    soxl_wfs = sl(soxl_f["W_Fisher_Sig"])
    soxl_dwf = sl(soxl_f["dW_Fisher"])
    soxl_wvr = sl(soxl_f["W_Vol_R"])
    soxl_hv20 = sl(soxl_f["D_HV20"])
    hv_falling = sl(soxl_hv20_fall).astype(bool)

    smh_dc = sl(smh_d["Close"])
    smh_ema10 = sl(smh_ema10_s)
    smh_ema20 = sl(smh_ema20_s)
    smh_ema50 = sl(smh_ema50_s)
    smh_wf = sl(smh_f["W_Fisher"])
    smh_wfs = sl(smh_f["W_Fisher_Sig"])

    vcap = 115.0
    vol_exit = soxl_hv20 > vcap
    alt_buy_18 = is_w_end & (soxl_wf < -1.8) & ((soxl_dwf > 0) | (soxl_wvr > 1.3))
    alt_buy_22 = is_w_end & (soxl_wf < -2.2) & ((soxl_dwf > 0) | (soxl_wvr > 1.3))

    backbones = {
        "SMA200": (smh_f["D_SMA200"], 0.01),
        "EMA150": (smh_f["D_EMA150"], 0.02),
    }

    results: list[dict] = []
    replay_specs: dict[str, tuple] = {}

    def build_specs(ma_s: pd.Series, buf: float, alt_buy: np.ndarray | None):
        ma_arr = sl(ma_s)
        tb_raw = smh_dc > ma_arr * (1.0 + buf)
        tb_def = tb_raw & (soxl_hv20 <= vcap * 0.92)
        ma_brk = smh_dc < ma_arr * (1.0 - buf)
        above_3d = sl(consecutive_true(smh_d["Close"] > ma_s * (1.0 + buf), 3)).astype(bool)
        above_5d = sl(consecutive_true(smh_d["Close"] > ma_s * (1.0 + buf), 5)).astype(bool)

        tsl_opts = {
            "default": ("default", tb_def, alt_buy, 0),
            "soxl_ema10": ("soxl_ema10", tb_def & (closes > soxl_ema10), alt_buy, 0),
            "soxl_ema20": ("soxl_ema20", tb_def & (closes > soxl_ema20), alt_buy, 0),
            "smh_ema10": ("smh_ema10", tb_def & (smh_dc > smh_ema10), alt_buy, 0),
            "smh_ema20": ("smh_ema20", tb_def & (smh_dc > smh_ema20), alt_buy, 0),
            "d_fish_up": ("d_fish_up", tb_def & (soxl_df > soxl_dfs), alt_buy, 0),
            "w_fish_up": ("w_fish_up", tb_def & (soxl_wf > soxl_wfs), alt_buy, 0),
            "cd5": ("cd5", tb_def, alt_buy, 5),
            "cd10": ("cd10", tb_def, alt_buy, 10),
            "cd15": ("cd15", tb_def, alt_buy, 15),
        }
        vol_opts = {
            "default": ("default", tb_def, alt_buy, 0),
            "hv_drop_95": ("hv_drop_95", tb_raw & (soxl_hv20 <= 95.0), alt_buy, 0),
            "hv_drop_100": ("hv_drop_100", tb_raw & (soxl_hv20 <= 100.0), alt_buy, 0),
            "hv_cool_soxl20": ("hv_cool_soxl20", tb_def & hv_falling & (closes > soxl_ema20), alt_buy, 0),
            "hv_cool_smh20": ("hv_cool_smh20", tb_def & hv_falling & (smh_dc > smh_ema20), alt_buy, 0),
            "w_fish_up": ("w_fish_up", tb_def & (soxl_wf > soxl_wfs), alt_buy, 0),
        }
        tb_opts = {
            "default": ("default", tb_def, alt_buy, 0),
            "confirm_3d": ("confirm_3d", tb_def & above_3d, alt_buy, 0),
            "confirm_5d": ("confirm_5d", tb_def & above_5d, alt_buy, 0),
            "smh_w_fish_up": ("smh_w_fish_up", tb_def & (smh_wf > smh_wfs), alt_buy, 0),
        }
        fsl_opts = {
            "default": ("default", tb_def, alt_buy, -1),
            "cd15": ("cd15", tb_def, alt_buy, 15),
            "cd20": ("cd20", tb_def, alt_buy, 20),
            "cd30": ("cd30", tb_def, alt_buy, 30),
            "deeper_wf_-2.2": ("deeper_wf_-2.2", tb_def, alt_buy_22 if alt_buy is not None else None, 0),
            "ban_alt_until_trend": ("ban_alt_until_trend", tb_def, None, 0),
        }
        tp_opts = {
            "default": ("default", tb_def, alt_buy, -1, -1, None),
            "trend_cd0_alt_cd10": ("trend_cd0_alt_cd10", tb_def, alt_buy, 0, 10, None),
            "fast_trend_smh_ema50": (
                "fast_trend_smh_ema50",
                tb_def | ((smh_dc > smh_ema50) & (soxl_hv20 <= vcap * 0.92)),
                alt_buy,
                0,
                10,
                smh_dc < smh_ema50 * 0.98,
            ),
            "fast_trend_soxl_w20": (
                "fast_trend_soxl_w20",
                tb_def | (is_w_end & (soxl_wc > soxl_ws20) & (soxl_hv20 <= vcap * 0.92)),
                alt_buy,
                0,
                10,
                is_w_end & (soxl_wc < soxl_ws20 * 0.98),
            ),
            "alt_cd5": ("alt_cd5", tb_def, alt_buy, 5, 5, None),
            "alt_cd15": ("alt_cd15", tb_def, alt_buy, 15, 15, None),
            "pullback_10pct": ("pullback_10pct", tb_def, (soxl_df > soxl_dfs), 0, 5, None),
        }
        return tb_def, ma_brk, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts

    def record(exp: str, bb: str, sweep: str, ttsl, acd: int, k_tsl: str, k_vol: str, k_tb: str, k_fsl: str, k_tp: str,
               tb_def, ma_brk, alt_buy, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts) -> None:
        args = (opens, closes, tb_def, vol_exit, ma_brk, alt_buy, year_end_indices, bt_idx)
        kw = {
            "trend_tsl": ttsl,
            "alt_fsl": 0.20 if alt_buy is not None else None,
            "alt_tp": 0.25 if alt_buy is not None else None,
            "alt_max_days": 40 if alt_buy is not None else None,
            "default_acd": acd,
            "after_tsl_spec": tsl_opts[k_tsl],
            "after_vol_spec": vol_opts[k_vol],
            "after_tb_spec": tb_opts[k_tb],
            "after_fsl_spec": fsl_opts[k_fsl],
            "after_tp_spec": tp_opts[k_tp],
        }
        m = simulate_exit_conditioned(*args, **kw)
        pstr = f"{bb},ttsl={ttsl},acd={acd}|tsl:{k_tsl}|vol:{k_vol}|tb:{k_tb}|fsl:{k_fsl}|tp:{k_tp}"
        results.append({"Exp": exp, "Backbone": bb, "Sweep": sweep, "Params": pstr,
                        "k_tsl": k_tsl, "k_vol": k_vol, "k_tb": k_tb, "k_fsl": k_fsl, "k_tp": k_tp,
                        "ttsl": str(ttsl), "acd": acd, **m})
        replay_specs[f"{exp}|{pstr}"] = (args, kw)

    # =========================================================================
    # Experiment A: Pure Trend (F2 without bottom buys)
    # =========================================================================
    for bb_name, (ma_s, buf) in backbones.items():
        tb_def, ma_brk, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts = build_specs(ma_s, buf, None)
        for ttsl in (0.20, 0.25, 0.30, None):
            # Baseline
            record("ExpA_PureTrend", bb_name, "baseline", ttsl, 0, "default", "default", "default", "default", "default",
                   tb_def, ma_brk, None, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            # Single-factor
            if ttsl is not None:
                for k in tsl_opts:
                    if k != "default":
                        record("ExpA_PureTrend", bb_name, "single_tsl", ttsl, 0, k, "default", "default", "default", "default",
                               tb_def, ma_brk, None, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            for k in vol_opts:
                if k != "default":
                    record("ExpA_PureTrend", bb_name, "single_vol", ttsl, 0, "default", k, "default", "default", "default",
                           tb_def, ma_brk, None, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            for k in tb_opts:
                if k != "default":
                    record("ExpA_PureTrend", bb_name, "single_tb", ttsl, 0, "default", "default", k, "default", "default",
                           tb_def, ma_brk, None, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
        # Combined Exp A
        for ttsl in (0.20, 0.25):
            for k_tsl in ("default", "soxl_ema10", "soxl_ema20", "smh_ema20", "w_fish_up", "cd10"):
                for k_vol in ("default", "hv_drop_95", "hv_cool_smh20", "w_fish_up"):
                    for k_tb in ("default", "confirm_3d", "smh_w_fish_up"):
                        if (k_tsl, k_vol, k_tb) == ("default", "default", "default"):
                            continue
                        record("ExpA_PureTrend", bb_name, "combined", ttsl, 0, k_tsl, k_vol, k_tb, "default", "default",
                               tb_def, ma_brk, None, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)

    # =========================================================================
    # Experiment B: Paired Entry-Exit (P1)
    # =========================================================================
    for bb_name, (ma_s, buf) in backbones.items():
        tb_def, ma_brk, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts = build_specs(ma_s, buf, alt_buy_18)
        for acd in (10, 0):
            for ttsl in (None, 0.20, 0.25, 0.30):
                record("ExpB_P1", bb_name, "baseline", ttsl, acd, "default", "default", "default", "default", "default",
                       tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)

        # Single-factor sweeps at acd=10 (with ttsl=0.25 for after_tsl, and ttsl=None for vol/tb/fsl/tp)
        for ttsl in (0.20, 0.25, 0.30):
            for k in tsl_opts:
                if k != "default":
                    record("ExpB_P1", bb_name, "single_tsl", ttsl, 10, k, "default", "default", "default", "default",
                           tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)

        for ttsl in (None, 0.25):
            for k in vol_opts:
                if k != "default":
                    record("ExpB_P1", bb_name, "single_vol", ttsl, 10, "default", k, "default", "default", "default",
                           tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            for k in tb_opts:
                if k != "default":
                    record("ExpB_P1", bb_name, "single_tb", ttsl, 10, "default", "default", k, "default", "default",
                           tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            for k in fsl_opts:
                if k != "default":
                    record("ExpB_P1", bb_name, "single_fsl", ttsl, 10, "default", "default", "default", k, "default",
                           tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)
            for k in tp_opts:
                if k != "default":
                    record("ExpB_P1", bb_name, "single_tp", ttsl, 10, "default", "default", "default", "default", k,
                           tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)

        # Combined Exp B across top candidate options on both backbones
        for ttsl in (None, 0.25, 0.30):
            tsl_list = ("default",) if ttsl is None else ("default", "soxl_ema20", "smh_ema20", "w_fish_up", "cd10")
            for k_tsl in tsl_list:
                for k_vol in ("default", "hv_drop_95", "hv_cool_smh20"):
                    for k_tb in ("default", "confirm_3d"):
                        for k_fsl in ("default", "cd20", "deeper_wf_-2.2", "ban_alt_until_trend"):
                            for k_tp in ("default", "trend_cd0_alt_cd10", "fast_trend_smh_ema50", "pullback_10pct"):
                                if (k_tsl, k_vol, k_tb, k_fsl, k_tp) == ("default", "default", "default", "default", "default"):
                                    continue
                                record("ExpB_P1", bb_name, "combined", ttsl, 10, k_tsl, k_vol, k_tb, k_fsl, k_tp,
                                       tb_def, ma_brk, alt_buy_18, tsl_opts, vol_opts, tb_opts, fsl_opts, tp_opts)

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(results).sort_values(["Calmar", "CAGR"], ascending=[False, False]).reset_index(drop=True)
    out_csv = ROOT / "tmp" / "scan_exit_reentry_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"Scanned {len(df):,} exit-conditioned configs in {elapsed:.2f}s -> {out_csv}\n")

    cols = ["Backbone", "Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "WorstYear"]

    # 1. Baseline verification
    print("=== 1. Baseline Verification ===")
    b_rows = df[df["Sweep"] == "baseline"]
    print(b_rows[["Exp"] + cols].to_string(index=False))

    # 2. Experiment A (Pure Trend)
    print("\n=== 2. Experiment A (Pure Trend F2): Top 10 by Calmar + Best MaxDD > -48% ===")
    exp_a = df[df["Exp"] == "ExpA_PureTrend"]
    print(exp_a.head(10)[cols].to_string(index=False))
    exp_a_lowdd = exp_a[exp_a["MaxDD"] > -48.0].head(5)
    print("\n-- ExpA MaxDD > -48% --")
    print(exp_a_lowdd[cols].to_string(index=False) if not exp_a_lowdd.empty else "  (None)")

    # 3. Experiment B Single-Factor Effect Table on BOTH Backbones
    print("\n=== 3. Experiment B (P1): Single-Factor Effect vs Exact Baseline (acd=10) ===")
    exp_b = df[df["Exp"] == "ExpB_P1"]
    for sweep_tag, rule_col, ttsl_ref in [
        ("single_tsl", "k_tsl", "0.25"),
        ("single_vol", "k_vol", "None"),
        ("single_tb", "k_tb", "None"),
        ("single_fsl", "k_fsl", "None"),
        ("single_tp", "k_tp", "None"),
    ]:
        print(f"\n-- Factor: {sweep_tag} (ttsl={ttsl_ref}) --")
        base_sub = exp_b[(exp_b["Sweep"] == "baseline") & (exp_b["acd"] == 10) & (exp_b["ttsl"] == ttsl_ref)]
        fac_sub = exp_b[(exp_b["Sweep"] == sweep_tag) & (exp_b["ttsl"] == ttsl_ref)]
        merged = pd.concat([base_sub, fac_sub], ignore_index=True)
        piv = merged.pivot_table(
            index=rule_col,
            columns="Backbone",
            values=["CAGR", "MaxDD", "Calmar"],
            aggfunc="first",
        )
        print(piv.round(3).to_string())

    # 4. Experiment B Top 15 Combined & Low-DD
    print("\n=== 4. Experiment B (P1): Top 15 Overall by Calmar ===")
    print(exp_b.head(15)[cols].to_string(index=False))

    print("\n=== 5. Experiment B (P1): Best under MaxDD > -50% and > -45% ===")
    for lbl, fl in [("MaxDD > -50%", -50.0), ("MaxDD > -45%", -45.0)]:
        sub = exp_b[exp_b["MaxDD"] > fl].head(5)
        print(f"-- {lbl} --")
        print(sub[cols].to_string(index=False) if not sub.empty else "  (None)")

    # 6. Detailed Breakdown of Best Exp A and Best Exp B (both overall and cross-backbone robust)
    best_a = exp_a.iloc[0]
    best_b = exp_b.iloc[0]
    for title, row in [("Best Exp A (Pure Trend + Exit-Conditioned)", best_a),
                       ("Best Exp B (Paired P1 + Exit-Conditioned)", best_b)]:
        key = f"{row['Exp']}|{row['Params']}"
        args, kw = replay_specs[key]
        det = simulate_exit_conditioned(*args, record_details=True, **kw)
        print(f"\n=== {title} ===")
        print(f"Params: {row['Params']}")
        print(f"Final: ${det['Final']:,.2f} | CAGR: {det['CAGR']}% | MaxDD: {det['MaxDD']}% | Calmar: {det['Calmar']} | Sharpe: {det['Sharpe']} | Sortino: {det['Sortino']} | Trades: {det['Trades']} | WinRate: {det['WinRate']}%")
        print("Yearly:", " | ".join(f"{yr}: {v:+.1f}%" for yr, v in det["yearly_rets"].items()))
        tdf = pd.DataFrame(det["trade_logs"])
        if not tdf.empty:
            print("Attribution by previous exit:")
            print(tdf.groupby(["prev_exit", "mode"])["ret_pct"].agg(
                Trades="count",
                WinRate_pct=lambda s: round((s > 0).mean() * 100, 1),
                AvgRet_pct=lambda s: round(s.mean(), 1),
            ).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
