"""Head-to-head comparison: trading TQQQ vs TECL using SOXL's signals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import (
    DATA_DIR,
    FEE_RATE,
    INITIAL_CAPITAL,
    _compute_stats,
    build_causal_features,
    load_daily_adjusted,
    simulate_d1_x1a_wtsl35,
    simulate_h5_adapt_hv,
    simulate_tp3h,
)
from tools.build_tqqq_zones import simulate_tqqq_from_soxl_signal


def buy_and_hold(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    d = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    shares = (INITIAL_CAPITAL * (1.0 - FEE_RATE)) / opens[0]
    eq = shares * closes
    ret = (eq[-1] / INITIAL_CAPITAL - 1.0) * 100.0
    trades = [{"ret_pct": round(float(ret), 1)}]
    return _compute_stats(eq, trades, len(d), pd.DatetimeIndex(d["Date"]))


def run_window(label: str, start_date: str, end_date: str, feats: dict, df_tqqq: pd.DataFrame, df_tecl: pd.DataFrame) -> None:
    print(f"\n=== {label} ({start_date} ~ {end_date}) ===")
    bh_t = buy_and_hold(df_tqqq, start_date, end_date)
    bh_e = buy_and_hold(df_tecl, start_date, end_date)
    print(f"  Buy&Hold TQQQ | Final=${bh_t['final']:12,.2f} | CAGR={bh_t['cagr']:6.2f}% | MaxDD={bh_t['max_dd']:6.2f}% | Calmar={bh_t['calmar']:5.3f} | Sharpe={bh_t['sharpe']:4.2f}")
    print(f"  Buy&Hold TECL | Final=${bh_e['final']:12,.2f} | CAGR={bh_e['cagr']:6.2f}% | MaxDD={bh_e['max_dd']:6.2f}% | Calmar={bh_e['calmar']:5.3f} | Sharpe={bh_e['sharpe']:4.2f}")

    soxl_strats = [
        (
            "H5_AdaptHV",
            simulate_h5_adapt_hv(feats, start_date=start_date, end_date=end_date),
        ),
        (
            "C2_Deep85HL_TP3h",
            simulate_tp3h(
                feats,
                use_f1_filter=True,
                start_date=start_date,
                end_date=end_date,
                entry_col_override="C2_Deep85HL",
                mode_tag_override="BOT_C2",
                strategy_id="C2_Deep85HL",
                tp_split_lottery_pct=0.10,
            ),
        ),
        (
            "F0_TP3h",
            simulate_tp3h(feats, use_f1_filter=False, start_date=start_date, end_date=end_date),
        ),
        (
            "D1_X1a_WTSL35",
            simulate_d1_x1a_wtsl35(feats, use_split_50_50=False, start_date=start_date, end_date=end_date),
        ),
    ]

    for sname, s_res in soxl_strats:
        st_s = s_res["stats"]
        r_t = simulate_tqqq_from_soxl_signal(df_tqqq, s_res, sname, sname, sname, "", start_date=start_date, end_date=end_date)["stats"]
        r_e = simulate_tqqq_from_soxl_signal(df_tecl, s_res, sname, sname, sname, "", start_date=start_date, end_date=end_date)["stats"]
        print(f"\n  [{sname}]")
        print(f"    -> SOXL (原标的) | Final=${st_s['final']:12,.2f} | CAGR={st_s['cagr']:6.2f}% | MaxDD={st_s['max_dd']:6.2f}% | Calmar={st_s['calmar']:5.3f} | Sharpe={st_s['sharpe']:4.2f} | Trades={st_s['trades']:2d} | WR={st_s['win_rate']:5.1f}%")
        print(f"    -> TQQQ          | Final=${r_t['final']:12,.2f} | CAGR={r_t['cagr']:6.2f}% | MaxDD={r_t['max_dd']:6.2f}% | Calmar={r_t['calmar']:5.3f} | Sharpe={r_t['sharpe']:4.2f} | Trades={r_t['trades']:2d} | WR={r_t['win_rate']:5.1f}%")
        print(f"    -> TECL          | Final=${r_e['final']:12,.2f} | CAGR={r_e['cagr']:6.2f}% | MaxDD={r_e['max_dd']:6.2f}% | Calmar={r_e['calmar']:5.3f} | Sharpe={r_e['sharpe']:4.2f} | Trades={r_e['trades']:2d} | WR={r_e['win_rate']:5.1f}%")


def main() -> None:
    df_soxl = load_daily_adjusted(DATA_DIR / "SOXL.csv")
    df_smh = load_daily_adjusted(DATA_DIR / "SMH.csv")
    df_tqqq = load_daily_adjusted(DATA_DIR / "TQQQ.csv")
    df_tecl = load_daily_adjusted(DATA_DIR / "TECL.csv")

    latest_common = str(min(df_soxl["Date"].max(), df_tqqq["Date"].max(), df_tecl["Date"].max()).date())
    feats = build_causal_features(df_soxl, df_smh)

    run_window("12年全周期", "2015-01-02", latest_common, feats, df_tqqq, df_tecl)
    run_window("2015-2025 主回测", "2015-01-02", "2025-12-31", feats, df_tqqq, df_tecl)
    run_window("早期样本外 (2011-2014)", "2011-01-03", "2014-12-31", feats, df_tqqq, df_tecl)
    run_window("2026 样本外 YTD", "2026-01-02", latest_common, feats, df_tqqq, df_tecl)


if __name__ == "__main__":
    main()
