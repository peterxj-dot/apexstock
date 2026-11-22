"""Verify share-volume and dollar-volume capitulation spikes at major bottoms (SOXL & TQQQ)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from scan_soxl_strategies import load_adjusted
from study_major_bottoms import build_weekly_dataset, evaluate_bottom_rule
from test_time_since_top import build_base_rules, get_troughs


def add_volume_features(w: pd.DataFrame, daily_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal share-volume, dollar-volume, and benchmark-volume metrics."""
    w = w.copy()

    # Share volume ratios (W_Vol_R already in w)
    w["max4w_Vol_R"] = w["W_Vol_R"].rolling(4, min_periods=1).max()
    w["max6w_Vol_R"] = w["W_Vol_R"].rolling(6, min_periods=1).max()

    # Dollar volume (split-adjusted Close * Volume)
    dol_vol = w["Close"] * w["Volume"]
    dol_sma20 = dol_vol.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    w["W_DolVol_R"] = dol_vol / dol_sma20
    w["max4w_DolVol_R"] = w["W_DolVol_R"].rolling(4, min_periods=1).max()
    w["max6w_DolVol_R"] = w["W_DolVol_R"].rolling(6, min_periods=1).max()

    # Causal completed monthly volume ratio
    m = (
        daily_df.assign(_dt=daily_df.index)
        .groupby(pd.Grouper(freq="ME"))
        .agg({"Volume": "sum", "_dt": "max"})
        .dropna(subset=["_dt"])
    )
    m.index = pd.DatetimeIndex(m["_dt"])
    m["M_Vol_R"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean().replace(0.0, np.nan)
    w["M_Vol_R"] = m["M_Vol_R"].reindex(w.index, method="ffill")

    # Benchmark weekly volume ratio (SMH / QQQ)
    bw = (
        bench_df.assign(_dt=bench_df.index)
        .groupby(pd.Grouper(freq="W-FRI"))
        .agg({"Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["_dt"])
    )
    bw.index = pd.DatetimeIndex(bw["_dt"])
    bw["Bench_Vol_R"] = bw["Volume"] / bw["Volume"].rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    b_dol = bw["Close"] * bw["Volume"]
    bw["Bench_DolVol_R"] = b_dol / b_dol.rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    bw_al = bw[["Bench_Vol_R", "Bench_DolVol_R"]].reindex(w.index, method="ffill")
    w["Bench_Vol_R"] = bw_al["Bench_Vol_R"]
    w["Bench_DolVol_R"] = bw_al["Bench_DolVol_R"]
    w["max4w_Bench_Vol_R"] = w["Bench_Vol_R"].rolling(4, min_periods=1).max()
    return w


def main() -> None:
    soxl = load_adjusted("SOXL")
    smh = load_adjusted("SMH")
    w_soxl, d_soxl = build_weekly_dataset(soxl, smh)
    w_soxl = add_volume_features(w_soxl, d_soxl, smh)
    soxl_troughs = get_troughs(w_soxl, ["2016-02", "2018-12", "2020-03", "2022-10", "2025-04"])

    tqqq = load_adjusted("TQQQ")
    qqq = load_adjusted("QQQ")
    w_tqqq, d_tqqq = build_weekly_dataset(tqqq, qqq)
    w_tqqq = add_volume_features(w_tqqq, d_tqqq, qqq)
    tqqq_troughs = get_troughs(w_tqqq, ["2016-02", "2018-12", "2020-03", "2022-12", "2025-04"])

    v_cols = [
        "W_Vol_R", "max4w_Vol_R", "max6w_Vol_R",
        "W_DolVol_R", "max4w_DolVol_R", "max6w_DolVol_R",
        "M_Vol_R", "Bench_Vol_R", "max4w_Bench_Vol_R",
    ]

    # =========================================================================
    # PART 1: Volume Anatomy at Major Bottoms vs Ordinary Weeks
    # =========================================================================
    print("========================================================================================")
    print("PART 1: VOLUME ANATOMY AT 5 MAJOR BOTTOMS VS ORDINARY WEEKS (SOXL & TQQQ 2014-2026)")
    print("========================================================================================")

    for asset_lbl, wf_df, troughs in [("SOXL", w_soxl, soxl_troughs), ("TQQQ", w_tqqq, tqqq_troughs)]:
        sub_14 = wf_df.loc["2014-01-01":]
        q = sub_14[v_cols].quantile([0.50, 0.75, 0.90]).round(2)
        q.index = ["P50 (Median)", "P75", "P90"]
        print(f"\n--- {asset_lbl} All Weeks Distribution (2014-2026, n={len(sub_14)}) ---")
        print(q.to_string())

        t_rows = []
        for bm, pos in troughs.items():
            # Also check +/- 1w peak around T0
            sl = wf_df.iloc[max(0, pos - 1) : min(len(wf_df), pos + 2)]
            row = wf_df.iloc[pos][v_cols].to_dict()
            row["Bottom"] = bm
            row["T0_Week"] = wf_df.index[pos].strftime("%Y-%m-%d")
            row["Peak_pm1w_VolR"] = float(sl["W_Vol_R"].max())
            row["Peak_pm1w_DolR"] = float(sl["W_DolVol_R"].max())
            t_rows.append(row)
        print(f"\n--- {asset_lbl} Exact 5 Major Bottom Trough Weeks (T0) & +/-1w Peak ---")
        tdf = pd.DataFrame(t_rows)[["Bottom", "T0_Week", "W_Vol_R", "Peak_pm1w_VolR", "max4w_Vol_R", "W_DolVol_R", "Peak_pm1w_DolR", "max4w_DolVol_R", "M_Vol_R", "Bench_Vol_R", "max4w_Bench_Vol_R"]]
        print(tdf.round(2).to_string(index=False))

    print("\n--- SOXL Live Context (2026-10-30 ~ 2026-11-19) ---")
    live = w_soxl.iloc[-4:][["Low", "Close", "W_Fish_Log", "dW_Fish_Log"] + v_cols].copy()
    live.insert(0, "WeekEnd", live.index.strftime("%Y-%m-%d"))
    print(live.round(2).to_string(index=False))

    # Event-level volume comparison on B9_Log and C7 (SOXL)
    b9_s, c7_s = build_base_rules(w_soxl)
    b9_t, c7_t = build_base_rules(w_tqqq)

    for rname, cond in [("B9_Log", b9_s), ("C7", c7_s)]:
        _, evs = evaluate_bottom_rule(rname, cond, w_soxl, d_soxl, soxl_troughs)
        rows = []
        for ev in evs:
            vals = w_soxl.iloc[ev["w_pos"]][v_cols].to_dict()
            rows.append({
                "sig_week": ev["sig_week"],
                "NearestBottom": ev["NearestBottom"],
                "InWin": ev["InBottomWin"],
                "MAE_6M%": ev["MAE_6M%"],
                "Fwd_6M%": ev["Fwd_6M%"],
                **vals,
            })
        print(f"\n--- SOXL {rname} Events: True Bottoms vs False Alarms (Volume Detail) ---")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    # =========================================================================
    # PART 2: Test Volume-Surge Filters on B9_Log & C7 (SOXL & TQQQ)
    # =========================================================================
    print("\n========================================================================================")
    print("PART 2: TESTING CAUSAL VOLUME FILTERS ON B9_Log AND C7 (SOXL & TQQQ 2014-2026)")
    print("========================================================================================")

    def make_vol_filters(wf_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
        return [
            ("Baseline (No Vol Filter)", pd.Series(True, index=wf_df.index)),
            ("SignalWeek: W_Vol_R >= 1.2", wf_df["W_Vol_R"] >= 1.2),
            ("SignalWeek: W_Vol_R >= 1.3", wf_df["W_Vol_R"] >= 1.3),
            ("SignalWeek: W_Vol_R >= 1.5", wf_df["W_Vol_R"] >= 1.5),
            ("Peak4w: max4w_Vol_R >= 1.5", wf_df["max4w_Vol_R"] >= 1.5),
            ("Peak4w: max4w_Vol_R >= 1.6", wf_df["max4w_Vol_R"] >= 1.6),
            ("Peak4w: max4w_Vol_R >= 2.0", wf_df["max4w_Vol_R"] >= 2.0),
            ("Peak6w: max6w_Vol_R >= 1.5", wf_df["max6w_Vol_R"] >= 1.5),
            ("Peak6w: max6w_Vol_R >= 1.6", wf_df["max6w_Vol_R"] >= 1.6),
            ("Peak6w: max6w_Vol_R >= 2.0", wf_df["max6w_Vol_R"] >= 2.0),
            ("DolPeak4w: max4w_DolVol_R >= 1.2", wf_df["max4w_DolVol_R"] >= 1.2),
            ("DolPeak4w: max4w_DolVol_R >= 1.3", wf_df["max4w_DolVol_R"] >= 1.3),
            ("DolPeak4w: max4w_DolVol_R >= 1.5", wf_df["max4w_DolVol_R"] >= 1.5),
            ("DolPeak6w: max6w_DolVol_R >= 1.2", wf_df["max6w_DolVol_R"] >= 1.2),
            ("DolPeak6w: max6w_DolVol_R >= 1.3", wf_df["max6w_DolVol_R"] >= 1.3),
            ("DolPeak6w: max6w_DolVol_R >= 1.5", wf_df["max6w_DolVol_R"] >= 1.5),
            ("BenchPeak4w: max4w_Bench_Vol_R >= 1.5", wf_df["max4w_Bench_Vol_R"] >= 1.5),
            ("BenchPeak4w: max4w_Bench_Vol_R >= 1.8", wf_df["max4w_Bench_Vol_R"] >= 1.8),
        ]

    vf_s = make_vol_filters(w_soxl)
    vf_t = make_vol_filters(w_tqqq)

    for bname, bs, bt in [("B9_Log", b9_s, b9_t), ("C7", c7_s, c7_t)]:
        print(f"\n=== SOXL: {bname} + Volume Filters ===")
        sr = [evaluate_bottom_rule(fn, bs & fc, w_soxl, d_soxl, soxl_troughs)[0] for fn, fc in vf_s]
        print(pd.DataFrame(sr).to_string(index=False))

        print(f"\n=== TQQQ Cross-Check: {bname} + Volume Filters ===")
        tr = [evaluate_bottom_rule(fn, bt & fc, w_tqqq, d_tqqq, tqqq_troughs)[0] for fn, fc in vf_t]
        print(pd.DataFrame(tr).to_string(index=False))


if __name__ == "__main__":
    main()
