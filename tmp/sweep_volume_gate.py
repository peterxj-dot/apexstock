"""Sweep weekly volume-gate thresholds (0.0 -> 2.0x) across F0, R3 (F0|MFishLin), and R4 (WLog|MFishLog) on SOXL, TQQQ, and 50 synthetic worlds."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from improve_tp3h_and_robustness import prepare_extended_data, run_backtest_param
from scan_soxl_strategies import load_adjusted
from simulate_synthetic_soxl import build_study_data_from_dfs, generate_synthetic_pair
from study_monthly_fisher_never_miss import add_log_monthly_and_weekly_fisher


def make_vol_masks(ds: dict) -> list[tuple[str, np.ndarray]]:
    w = ds["weekly"]
    near_200 = ((1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min() <= 1.10).fillna(False)
    wf_ok = (w["W_Fisher"].rolling(4, min_periods=1).min() < -1.5).fillna(False)
    prev_hi = (w["Close"] > w["High"].shift(1)).fillna(False)
    vol4 = w["max4w_Vol_R"].fillna(0.0)
    mf_lin = w["MF_Lin_Buy"].fillna(False)
    mf_log = w["MF_Log_Buy"].fillna(False)
    w_log = w["W_LogBot_Buy"].fillna(False)

    masks: list[tuple[str, np.ndarray]] = []
    # Family A: Pure F0(vol_th) without monthly cross backstop
    for vt in (0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0):
        v_ok = pd.Series(True, index=w.index) if vt <= 0.0 else (vol4 >= vt)
        m = (near_200 & wf_ok & v_ok & prev_hi).to_numpy(dtype=bool)
        masks.append((f"A_F0_v{vt:.1f}", m))

    # Family B: R3 = F0(vol_th) | MFishLin_Buy
    for vt in (0.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0):
        v_ok = pd.Series(True, index=w.index) if vt <= 0.0 else (vol4 >= vt)
        m = ((near_200 & wf_ok & v_ok & prev_hi) | mf_lin).to_numpy(dtype=bool)
        masks.append((f"B_R3_v{vt:.1f}", m))

    # Family C: R4 = (W_LogBot & Vol >= vol_th) | MFishLog_Buy
    for vt in (0.0, 1.2, 1.4, 1.6, 1.8):
        v_ok = pd.Series(True, index=w.index) if vt <= 0.0 else (vol4 >= vt)
        m = ((w_log & v_ok) | mf_log).to_numpy(dtype=bool)
        masks.append((f"C_R4_v{vt:.1f}", m))

    return masks


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 24)

    soxl = add_log_monthly_and_weekly_fisher(prepare_extended_data("SOXL", "SMH"))
    tqqq = add_log_monthly_and_weekly_fisher(prepare_extended_data("TQQQ", "QQQ"))

    ms_soxl = make_vol_masks(soxl)
    ms_tqqq = dict(make_vol_masks(tqqq))

    # 1. Real SOXL (11yr/12yr & 15.5yr) + Real TQQQ
    real_rows = []
    for name, m in ms_soxl:
        r15 = run_backtest_param(soxl, m, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag=name)
        r11 = run_backtest_param(soxl, m, "TP3h_Base", "2011-06-01", "2026-11-19", filter_tag=name)
        rt = run_backtest_param(tqqq, ms_tqqq[name], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag=name)
        real_rows.append(
            {
                "Rule": name,
                "S12_Final($M)": round(r15["Final"] / 1e6, 2),
                "S12_CAGR%": r15["CAGR"],
                "S12_DD%": r15["MaxDD"],
                "S12_Calmar": r15["Calmar"],
                "S12_Tr": r15["Trades"],
                "S12_Win%": r15["WinRate"],
                "S15_Final($M)": round(r11["Final"] / 1e6, 2),
                "S15_CAGR%": r11["CAGR"],
                "S15_DD%": r11["MaxDD"],
                "S15_Tr": r11["Trades"],
                "S15_Win%": r11["WinRate"],
                "TQ_Final($K)": round(rt["Final"] / 1e3, 1),
                "TQ_CAGR%": rt["CAGR"],
                "TQ_DD%": rt["MaxDD"],
                "TQ_Tr": rt["Trades"],
                "TQ_Win%": rt["WinRate"],
            }
        )

    print("=" * 125)
    print("PART 1: REAL SOXL (2015-2026 & 2011-2026) AND REAL TQQQ (2015-2026) VOLUME GATE SWEEP")
    print("=" * 125)
    print(pd.DataFrame(real_rows).to_string(index=False))

    # 2. 50-World Synthetic SOXL Stress Test
    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)

    rule_names = [n for n, _ in ms_soxl]
    trough_hits = {n: 0 for n in rule_names}
    total_troughs = 0
    mc_recs = []

    for w_i in range(50):
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, 100 + w_i)
        sd = add_log_monthly_and_weekly_fisher(build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH"))
        w_s = sd["weekly"]
        ms = dict(make_vol_masks(sd))

        w_bt = w_s[w_s.index >= "2015-01-01"]
        hi = w_bt["High"].to_numpy(dtype=np.float64)
        lo = w_bt["Low"].to_numpy(dtype=np.float64)
        roll_hi = pd.Series(hi).rolling(52, min_periods=1).max().to_numpy(dtype=np.float64)
        dd_pct = (lo / roll_hi - 1.0) * 100.0
        cands = np.where(dd_pct <= -55.0)[0]
        order = cands[np.argsort(dd_pct[cands])]
        sel: list[int] = []
        for ix in order:
            if all(abs(ix - s) >= 26 for s in sel):
                sel.append(int(ix))
        bt_offset = int(np.searchsorted(w_s.index, pd.Timestamp("2015-01-01")))
        for t_pos in sel:
            total_troughs += 1
            a0 = max(bt_offset, bt_offset + t_pos - 2)
            a1 = min(len(w_s), bt_offset + t_pos + 25)
            for n in rule_names:
                if bool(ms[n][a0:a1].any()):
                    trough_hits[n] += 1

        for n in rule_names:
            r = run_backtest_param(sd, ms[n], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag=n)
            mc_recs.append(
                {
                    "Rule": n,
                    "Final": r["Final"],
                    "CAGR": r["CAGR"],
                    "MaxDD": r["MaxDD"],
                    "Calmar": r["Calmar"],
                    "Trades": r["Trades"],
                    "WinRate": r["WinRate"],
                    "Zero": float(r["Trades"] == 0),
                }
            )

    df_mc = pd.DataFrame(mc_recs)
    mc_rows = []
    for n in rule_names:
        sub = df_mc[df_mc["Rule"] == n]
        mc_rows.append(
            {
                "Rule": n,
                "Catch%": round(trough_hits[n] / total_troughs * 100.0, 1),
                "ZeroW": int(sub["Zero"].sum()),
                "P25($K)": round(float(sub["Final"].quantile(0.25)) / 1e3, 1),
                "Med($K)": round(float(sub["Final"].median()) / 1e3, 1),
                "P75($K)": round(float(sub["Final"].quantile(0.75)) / 1e3, 1),
                "Med_CAGR%": round(float(sub["CAGR"].median()), 2),
                "Med_DD%": round(float(sub["MaxDD"].median()), 2),
                "Med_Calmar": round(float(sub["Calmar"].median()), 3),
                "Med_Tr": round(float(sub["Trades"].median()), 1),
                "Med_Win%": round(float(sub["WinRate"].median()), 1),
            }
        )

    print("\n" + "=" * 125)
    print(f"PART 2: 50-WORLD SYNTHETIC SOXL VOLUME GATE SWEEP (total -55%+ troughs = {total_troughs})")
    print("=" * 125)
    print(pd.DataFrame(mc_rows).to_string(index=False))
    print(f"\nCompleted in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
