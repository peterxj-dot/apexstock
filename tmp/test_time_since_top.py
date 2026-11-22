"""Test adding causal time-since-major-top filters to B9_Log and C7 bottom entry rules."""

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


def add_time_since_top_features(w: pd.DataFrame) -> pd.DataFrame:
    """Add strictly causal time-since-top, 8w-min-drawdown, and crash-speed features."""
    w = w.copy()
    highs = w["High"].to_numpy(dtype=np.float64)
    lows = w["Low"].to_numpy(dtype=np.float64)
    n = len(w)

    wks_52w = np.zeros(n, dtype=int)
    wks_104w = np.zeros(n, dtype=int)
    wks_conf_top = np.zeros(n, dtype=int)

    # Causal confirmed-top state machine (45% drop confirms peak; +100% rally resets UP leg)
    state = "UP"
    run_hi = highs[0]
    run_hi_idx = 0
    run_lo = lows[0]
    conf_top_idx = 0

    for i in range(n):
        s52 = max(0, i - 51)
        p52 = s52 + int(np.argmax(highs[s52 : i + 1]))
        wks_52w[i] = i - p52

        s104 = max(0, i - 103)
        p104 = s104 + int(np.argmax(highs[s104 : i + 1]))
        wks_104w[i] = i - p104

        h, l = highs[i], lows[i]
        if state == "UP":
            if h >= run_hi:
                run_hi = h
                run_hi_idx = i
            if l <= run_hi * 0.55:
                conf_top_idx = run_hi_idx
                state = "DOWN"
                run_lo = l
        else:
            if l < run_lo:
                run_lo = l
            if h >= run_lo * 2.0 or h > run_hi:
                state = "UP"
                run_hi = h
                run_hi_idx = i

        wks_conf_top[i] = 0 if (state == "UP" and h >= run_hi) else (i - conf_top_idx)

    w["wks_52w"] = wks_52w
    w["wks_104w"] = wks_104w
    w["wks_conf_top"] = wks_conf_top
    w["min8w_dd%"] = w["Low_DD_52W%"].rolling(8, min_periods=1).min()
    w["crash_speed"] = w["min8w_dd%"] / np.maximum(1, w["wks_52w"])
    return w


def build_base_rules(w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    b9_log = (w["Low"] <= w["W_SMA200"] * 1.10) & (w["W_Fish_Log"] < -1.5) & (w["dW_Fish_Log"] > 0)
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    min_low_200_8w = (w["Low"] / w["W_SMA200"]).rolling(8, min_periods=1).min()
    c7 = (min_low_200_8w <= 1.10) & (w["Close"] > w["W_EMA10"]) & (min_wf_4w < -1.5)
    return b9_log, c7


def get_troughs(w: pd.DataFrame, months: list[str]) -> dict[str, int]:
    ym = w.index.strftime("%Y-%m")
    out = {}
    for bm in months:
        sub = w[ym == bm]
        if not sub.empty:
            out[bm] = int(w.index.get_loc(sub["Low"].idxmin()))
    return out


def main() -> None:
    soxl = load_adjusted("SOXL")
    smh = load_adjusted("SMH")
    w_soxl, d_soxl = build_weekly_dataset(soxl, smh)
    w_soxl = add_time_since_top_features(w_soxl)
    soxl_troughs = get_troughs(w_soxl, ["2016-02", "2018-12", "2020-03", "2022-10", "2025-04"])

    tqqq = load_adjusted("TQQQ")
    qqq = load_adjusted("QQQ")
    w_tqqq, d_tqqq = build_weekly_dataset(tqqq, qqq)
    w_tqqq = add_time_since_top_features(w_tqqq)
    tqqq_troughs = get_troughs(w_tqqq, ["2016-02", "2018-12", "2020-03", "2022-12", "2025-04"])

    b9_s, c7_s = build_base_rules(w_soxl)
    b9_t, c7_t = build_base_rules(w_tqqq)

    # =========================================================================
    # PART 1: Full Diagnostic Table of Every B9_Log & C7 Event on SOXL
    # =========================================================================
    print("========================================================================================")
    print("PART 1: FULL DIAGNOSTIC TABLE OF B9_Log & C7 EVENTS (SOXL 2014-2026) + LIVE 2026-11-19")
    print("========================================================================================")

    diag_cols = ["wks_52w", "wks_104w", "wks_conf_top", "min8w_dd%", "crash_speed", "Close_vs_200W%", "SMH_vs_200W%", "W_Vol_R"]

    for rname, cond in [("B9_Log (Baseline)", b9_s), ("C7 (Baseline)", c7_s)]:
        s_sum, evs = evaluate_bottom_rule(rname, cond, w_soxl, d_soxl, soxl_troughs)
        print(f"\n--- {rname}: Events={s_sum['Events']}, Caught={s_sum['Caught']}, FalseAlarms={s_sum['FalseAlarms']} ---")
        rows = []
        for ev in evs:
            wp = ev["w_pos"]
            feat_vals = w_soxl.iloc[wp][diag_cols].to_dict()
            rows.append({
                "sig_week": ev["sig_week"],
                "entry_date": ev["entry_date"],
                "NearestBottom": ev["NearestBottom"],
                "InWin": ev["InBottomWin"],
                "MAE_6M%": ev["MAE_6M%"],
                "Fwd_6M%": ev["Fwd_6M%"],
                **feat_vals,
            })
        print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n--- Live Latest Weeks (2026-10-30 ~ 2026-11-19) ---")
    live = w_soxl.iloc[-4:][["Low", "Close", "W_Fish_Log", "dW_Fish_Log"] + diag_cols].copy()
    live.insert(0, "WeekEnd", live.index.strftime("%Y-%m-%d"))
    print(live.round(2).to_string(index=False))

    # =========================================================================
    # PART 2: Test Time-Since-Top & Time-OR-Depth/Speed Filters on B9_Log & C7
    # =========================================================================
    print("\n========================================================================================")
    print("PART 2: TESTING TIME-SINCE-TOP FILTERS ON B9_Log AND C7 (SOXL & TQQQ 2014-2026)")
    print("========================================================================================")

    def make_filters(wf_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
        filters = [("Baseline (No Time Filter)", pd.Series(True, index=wf_df.index))]
        for w_th in (16, 20, 24, 28, 32):
            filters.append((f"PureTime: wks_52w >= {w_th}", wf_df["wks_52w"] >= w_th))
        for w_th in (24, 28, 32):
            for d_th in (78, 80, 82):
                filters.append((
                    f"Time|Depth: wks_52w>={w_th} | min8w_dd<=-{d_th}%",
                    (wf_df["wks_52w"] >= w_th) | (wf_df["min8w_dd%"] <= -d_th),
                ))
        for w_th in (24, 28, 32):
            for spd in (5.0, 6.0):
                filters.append((
                    f"Time|Speed: wks_52w>={w_th} | speed<=-{spd}%/w",
                    (wf_df["wks_52w"] >= w_th) | (wf_df["crash_speed"] <= -spd),
                ))
        return filters

    f_soxl = make_filters(w_soxl)
    f_tqqq = make_filters(w_tqqq)

    for base_name, base_s, base_t in [("B9_Log", b9_s, b9_t), ("C7", c7_s, c7_t)]:
        print(f"\n=== SOXL Results: {base_name} + Time-Since-Top Filters ===")
        s_rows = []
        for fname, fcond in f_soxl:
            sm, _ = evaluate_bottom_rule(fname, base_s & fcond, w_soxl, d_soxl, soxl_troughs)
            s_rows.append(sm)
        print(pd.DataFrame(s_rows).to_string(index=False))

        print(f"\n=== TQQQ Cross-Check: {base_name} + Time-Since-Top Filters ===")
        t_rows = []
        for fname, fcond in f_tqqq:
            tm, _ = evaluate_bottom_rule(fname, base_t & fcond, w_tqqq, d_tqqq, tqqq_troughs)
            t_rows.append(tm)
        print(pd.DataFrame(t_rows).to_string(index=False))


if __name__ == "__main__":
    main()
