"""Blind out-of-sample (2011-2014, 2026) & cross-asset (TQQQ/QQQ) validation of frozen strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from scan_exit_conditioned_reentry import simulate_exit_conditioned
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL, compute_asset_features, load_adjusted


def run_window(
    target_ticker: str,
    bench_ticker: str,
    start_date: str,
    end_date: str,
    label: str,
    print_2026_logs: bool = False,
    print_tqqq_attr: bool = False,
) -> None:
    target = load_adjusted(target_ticker)
    bench = load_adjusted(bench_ticker)

    full_idx = target.index
    tf = compute_asset_features(target, full_idx)
    bf = compute_asset_features(bench, full_idx)

    mask = (full_idx >= start_date) & (full_idx <= end_date)
    bt_idx = full_idx[mask]
    n_bt = len(bt_idx)

    def sl(obj) -> np.ndarray:
        return obj.to_numpy()[mask]

    opens = sl(tf["daily"]["Open"]).astype(np.float64)
    closes = sl(tf["daily"]["Close"]).astype(np.float64)
    is_w_end = sl(tf["is_week_end"]).astype(bool)

    yr_series = pd.Series(np.arange(n_bt), index=bt_idx)
    year_end_indices = [(int(yr), int(grp.iloc[-1])) for yr, grp in yr_series.groupby(bt_idx.year)]

    t_wf = sl(tf["W_Fisher"])
    t_dwf = sl(tf["dW_Fisher"])
    t_wvr = sl(tf["W_Vol_R"])
    t_hv20 = sl(tf["D_HV20"])

    b_dc = sl(bf["daily"]["Close"])
    b_sma200 = sl(bf["D_SMA200"])
    b_ema150 = sl(bf["D_EMA150"])

    # Warm-up verification
    if np.isnan(b_sma200[0]) or np.isnan(t_wf[0]) or np.isnan(t_hv20[0]):
        raise RuntimeError(f"Insufficient warm-up at {start_date} for {target_ticker}/{bench_ticker}")

    vcap = 115.0
    vol_exit = t_hv20 > vcap
    alt_buy = is_w_end & (t_wf < -1.8) & ((t_dwf > 0) | (t_wvr > 1.3))

    # Buy & Hold
    bh_eq = INITIAL_CAPITAL * (closes / closes[0])
    years_span = n_bt / 252.0
    bh_ret = (bh_eq[-1] / INITIAL_CAPITAL - 1.0) * 100.0
    bh_cagr = ((bh_eq[-1] / INITIAL_CAPITAL) ** (1.0 / years_span) - 1.0) * 100.0
    bh_dd = ((bh_eq / np.maximum.accumulate(bh_eq)) - 1.0).min() * 100.0
    bh_calmar = bh_cagr / abs(bh_dd) if bh_dd < -1e-9 else 0.0
    bh_dr = np.diff(bh_eq) / bh_eq[:-1]
    bh_sharpe = float(bh_dr.mean() / bh_dr.std() * np.sqrt(252.0)) if bh_dr.std() > 0 else 0.0

    rows = [
        {
            "Strategy": f"Buy & Hold ({target_ticker})",
            "Final": round(float(bh_eq[-1]), 2),
            "Return%": round(bh_ret, 1),
            "CAGR%": round(bh_cagr, 2),
            "MaxDD%": round(float(bh_dd), 2),
            "Calmar": round(float(bh_calmar), 3),
            "Sharpe": round(bh_sharpe, 2),
            "Trades": 1,
            "WinRate%": 100.0 if bh_ret > 0 else 0.0,
            "Exposure%": 100.0,
        }
    ]

    # 1. PureTrend_SMA200_hv95
    tb_s200_raw = b_dc > b_sma200 * 1.01
    tb_s200 = tb_s200_raw & (t_hv20 <= vcap * 0.92)
    mb_s200 = b_dc < b_sma200 * 0.99

    # 2. EMA150
    tb_e150_raw = b_dc > b_ema150 * 1.02
    tb_e150 = tb_e150_raw & (t_hv20 <= vcap * 0.92)
    mb_e150 = b_dc < b_ema150 * 0.98

    configs = [
        (
            "PureTrend_SMA200_hv95",
            (opens, closes, tb_s200, vol_exit, mb_s200, None, year_end_indices, bt_idx),
            {
                "trend_tsl": None,
                "alt_fsl": None,
                "alt_tp": None,
                "alt_max_days": None,
                "default_acd": 0,
                "after_tsl_spec": ("default", tb_s200, None, 0),
                "after_vol_spec": ("hv_drop_95", tb_s200_raw & (t_hv20 <= 95.0), None, 0),
                "after_tb_spec": ("default", tb_s200, None, 0),
                "after_fsl_spec": ("default", tb_s200, None, -1),
                "after_tp_spec": ("default", tb_s200, None, -1, -1, None),
            },
        ),
        (
            "Paired_SMA200_hv95",
            (opens, closes, tb_s200, vol_exit, mb_s200, alt_buy, year_end_indices, bt_idx),
            {
                "trend_tsl": None,
                "alt_fsl": 0.20,
                "alt_tp": 0.25,
                "alt_max_days": 40,
                "default_acd": 10,
                "after_tsl_spec": ("default", tb_s200, alt_buy, 0),
                "after_vol_spec": ("hv_drop_95", tb_s200_raw & (t_hv20 <= 95.0), alt_buy, 0),
                "after_tb_spec": ("default", tb_s200, alt_buy, 0),
                "after_fsl_spec": ("default", tb_s200, alt_buy, -1),
                "after_tp_spec": ("default", tb_s200, alt_buy, -1, -1, None),
            },
        ),
        (
            "Paired_EMA150_tsl30_hv95",
            (opens, closes, tb_e150, vol_exit, mb_e150, alt_buy, year_end_indices, bt_idx),
            {
                "trend_tsl": 0.30,
                "alt_fsl": 0.20,
                "alt_tp": 0.25,
                "alt_max_days": 40,
                "default_acd": 10,
                "after_tsl_spec": ("default", tb_e150, alt_buy, 0),
                "after_vol_spec": ("hv_drop_95", tb_e150_raw & (t_hv20 <= 95.0), alt_buy, 0),
                "after_tb_spec": ("default", tb_e150, alt_buy, 0),
                "after_fsl_spec": ("default", tb_e150, alt_buy, -1),
                "after_tp_spec": ("default", tb_e150, alt_buy, -1, -1, None),
            },
        ),
        (
            "Paired_SMA200_tsl20_cd5",
            (opens, closes, tb_s200, vol_exit, mb_s200, alt_buy, year_end_indices, bt_idx),
            {
                "trend_tsl": 0.20,
                "alt_fsl": 0.20,
                "alt_tp": 0.25,
                "alt_max_days": 40,
                "default_acd": 10,
                "after_tsl_spec": ("cd5", tb_s200, alt_buy, 5),
                "after_vol_spec": ("default", tb_s200, alt_buy, 0),
                "after_tb_spec": ("default", tb_s200, alt_buy, 0),
                "after_fsl_spec": ("default", tb_s200, alt_buy, -1),
                "after_tp_spec": ("default", tb_s200, alt_buy, -1, -1, None),
            },
        ),
    ]

    details_map = {}
    for name, args, kw in configs:
        det = simulate_exit_conditioned(*args, record_details=True, **kw)
        details_map[name] = det
        tot_ret = (det["Final"] / INITIAL_CAPITAL - 1.0) * 100.0
        rows.append(
            {
                "Strategy": name,
                "Final": det["Final"],
                "Return%": round(tot_ret, 1),
                "CAGR%": det["CAGR"],
                "MaxDD%": det["MaxDD"],
                "Calmar": det["Calmar"],
                "Sharpe": det["Sharpe"],
                "Trades": det["Trades"],
                "WinRate%": det["WinRate"],
                "Exposure%": det["Exposure"],
            }
        )

    print(f"\n=== {label} ({target_ticker} driven by {bench_ticker}, {bt_idx[0].strftime('%Y-%m-%d')} ~ {bt_idx[-1].strftime('%Y-%m-%d')}, n={n_bt} days) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    if print_2026_logs:
        for sname in ("Paired_SMA200_hv95", "Paired_EMA150_tsl30_hv95"):
            print(f"\n-- {sname} 2026 Trade Log --")
            tdf = pd.DataFrame(details_map[sname]["trade_logs"])
            print(tdf.to_string(index=False) if not tdf.empty else "  (No trades)")

    if print_tqqq_attr:
        for sname in ("Paired_SMA200_hv95", "Paired_EMA150_tsl30_hv95"):
            tdf = pd.DataFrame(details_map[sname]["trade_logs"])
            print(f"\n-- {sname} ({target_ticker}) Mode Attribution --")
            if not tdf.empty:
                print(
                    tdf.groupby("mode")["ret_pct"]
                    .agg(
                        Trades="count",
                        WinRate_pct=lambda s: round((s > 0).mean() * 100, 1),
                        AvgRet_pct=lambda s: round(s.mean(), 1),
                    )
                    .reset_index()
                    .to_string(index=False)
                )


def main() -> None:
    # 1. SOXL 2026 OOS
    run_window("SOXL", "SMH", "2026-01-01", "2026-11-19", "1. SOXL 2026 Out-of-Sample", print_2026_logs=True)

    # 2. SOXL Continuous 2015 ~ 2026-11-19
    run_window("SOXL", "SMH", "2015-01-01", "2026-11-19", "2. SOXL Continuous 2015 ~ 2026-11-19")

    # 3. SOXL Early-History OOS (2011 ~ 2014)
    run_window("SOXL", "SMH", "2011-01-01", "2014-12-31", "3. SOXL Early-History OOS (2011-2014)")

    # 4. Cross-Asset Blind Test: TQQQ driven by QQQ (2015-2025)
    run_window("TQQQ", "QQQ", "2015-01-01", "2025-12-31", "4. Cross-Asset Blind Test: TQQQ (2015-2025)", print_tqqq_attr=True)

    # 5. Cross-Asset Blind Test: TQQQ 2026 OOS
    run_window("TQQQ", "QQQ", "2026-01-01", "2026-11-19", "5. Cross-Asset Blind Test: TQQQ 2026 OOS")


if __name__ == "__main__":
    main()
