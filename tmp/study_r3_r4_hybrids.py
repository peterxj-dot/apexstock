"""Test combinations of F1 (slow-long / fast-deep gate) with R3/R4 Monthly Fisher never-miss backstops."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from improve_tp3h_and_robustness import build_entry_mask, prepare_extended_data, run_backtest_param
from scan_soxl_strategies import load_adjusted
from simulate_synthetic_soxl import build_study_data_from_dfs, generate_synthetic_pair, run_strategy1_from_dfs
from study_major_exits_and_backtest import run_buy_and_hold
from study_monthly_fisher_never_miss import add_log_monthly_and_weekly_fisher
from test_cycle_v2_and_hybrid import run_strategy1_baseline


def build_combo_masks(ds: dict) -> dict[str, np.ndarray]:
    """Build F1, R3/R4, and C1..C4 hybrid entry masks strictly causally."""
    w = ds["weekly"]
    f0 = build_entry_mask(ds, None, None)
    f1 = build_entry_mask(ds, 30, -80.0)
    mf_lin = w["MF_Lin_Buy"].to_numpy(dtype=bool)
    mf_log = w["MF_Log_Buy"].to_numpy(dtype=bool)
    w_log = w["W_LogBot_Buy"].fillna(False).to_numpy(dtype=bool)

    # Relaxed F1-Soft: 200W <= 1.15, WF < -1.5, VolR >= 1.45, (wks >= 24 | dd <= -68%)
    near_200w_15 = ((1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min() <= 1.15).fillna(False)
    wf_ok = (w["W_Fisher"].rolling(4, min_periods=1).min() < -1.5).fillna(False)
    vol_145 = (w["max4w_Vol_R"] >= 1.45).fillna(False)
    td_soft = ((w["wks_52H"] >= 24) | (w["dd52_8wL%"] <= -68.0)).fillna(False)
    prev_hi = (w["Close"] > w["High"].shift(1)).fillna(False)
    f1_soft = (near_200w_15 & wf_ok & vol_145 & td_soft & prev_hi).to_numpy(dtype=bool)

    # Light guard on R4's weekly trigger (wks >= 20 | dd52_Low <= -70%) while leaving MFishLog unrestricted
    td_light = ((w["wks_52H"] >= 20) | (w["DD52_Low%"].rolling(8, min_periods=1).min() <= -70.0)).fillna(False).to_numpy(dtype=bool)
    w_log_guarded = w_log & td_light

    return {
        "F1_Ref": f1,
        "R3_F0_or_MFishLin": f0 | mf_lin,
        "R4_WLog_or_MFishLog": w_log | mf_log,
        "C1_F1_or_MFishLog": f1 | mf_log,
        "C2_F1_or_MFishLin": f1 | mf_lin,
        "C3_F1Soft_or_MFishLog": f1_soft | mf_log,
        "C4_R4Guard_or_MFishLog": w_log_guarded | mf_log,
    }


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)

    soxl = add_log_monthly_and_weekly_fisher(prepare_extended_data("SOXL", "SMH"))
    tqqq = add_log_monthly_and_weekly_fisher(prepare_extended_data("TQQQ", "QQQ"))

    # =========================================================================
    # 1. REAL SOXL (2011-2026 & 2015-2026) & REAL TQQQ (2015-2026)
    # =========================================================================
    print("=" * 115)
    print("1. REAL MARKET COMPARISON: F1 + R3/R4 HYBRID COMBINATIONS (TP3h EXIT)")
    print("=" * 115)
    for title, ds, tk, bk, s, e in [
        ("SOXL Early+Full (2011-06-01 ~ 2026-11-19)", soxl, "SOXL", "SMH", "2011-06-01", "2026-11-19"),
        ("SOXL Main (2015-01-01 ~ 2026-11-19)", soxl, "SOXL", "SMH", "2015-01-01", "2026-11-19"),
        ("TQQQ Main (2015-01-01 ~ 2026-11-19)", tqqq, "TQQQ", "QQQ", "2015-01-01", "2026-11-19"),
    ]:
        masks = build_combo_masks(ds)
        rows = [
            {**run_buy_and_hold(ds, s, e), "System": "Buy&Hold"},
            run_strategy1_baseline(tk, bk, s, e),
        ]
        for mname, m in masks.items():
            r = run_backtest_param(ds, m, "TP3h_Base", s, e, filter_tag=mname)
            r["System"] = f"{mname}+TP3h"
            rows.append(r)
        cols = ["System", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "Exposure", "WorstYear"]
        print(f"\n[{title}]")
        print(pd.DataFrame([{k: r[k] for k in cols} for r in rows]).to_string(index=False))

    # Print trade logs on SOXL 2011-2026 for C1, C2, C3, C4
    s_masks = build_combo_masks(soxl)
    for mname in ("C1_F1_or_MFishLog", "C2_F1_or_MFishLin", "C3_F1Soft_or_MFishLog", "C4_R4Guard_or_MFishLog"):
        r = run_backtest_param(soxl, s_masks[mname], "TP3h_Base", "2011-06-01", "2026-11-19", filter_tag=mname)
        print(f"\n[SOXL 2011-2026 Trade Log: {mname}+TP3h] Final=${r['Final']:,.2f} CAGR={r['CAGR']}% MaxDD={r['MaxDD']}% Trades={r['Trades']} WinRate={r['WinRate']}%")
        print(pd.DataFrame(r["trade_logs"]).to_string(index=False))

    # =========================================================================
    # 2. 50-WORLD SYNTHETIC SOXL STRESS TEST (2015-2026)
    # =========================================================================
    print("\n" + "=" * 115)
    print("2. 50-WORLD SYNTHETIC SOXL TEST (2015-2026): DOES F1 + R3/R4 KEEP BOTH HIGH RETURN & ZERO MISSED WORLDS?")
    print("=" * 115)
    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)

    mask_names = list(build_combo_masks(soxl).keys())
    strat_keys = ["Buy&Hold", "S1"] + [f"{m}+TP3h" for m in mask_names]
    trough_hits = {m: 0 for m in mask_names}
    total_troughs = 0
    recs = []

    for w_i in range(50):
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, 100 + w_i)
        sd = add_log_monthly_and_weekly_fisher(build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH"))
        w_s = sd["weekly"]
        masks = build_combo_masks(sd)

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
            for m in mask_names:
                if bool(masks[m][a0:a1].any()):
                    trough_hits[m] += 1

        r_bh = run_buy_and_hold(sd, "2015-01-01", "2026-11-19")
        r_s1 = run_strategy1_from_dfs(syn_soxl, syn_smh, "2015-01-01", "2026-11-19")
        run_list = [("Buy&Hold", r_bh), ("S1", r_s1)]
        for m in mask_names:
            run_list.append((f"{m}+TP3h", run_backtest_param(sd, masks[m], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag=m)))

        for sk, res in run_list:
            recs.append(
                {
                    "System": sk,
                    "Final": res["Final"],
                    "CAGR": res["CAGR"],
                    "MaxDD": res["MaxDD"],
                    "Calmar": res["Calmar"],
                    "Trades": res["Trades"],
                    "WinRate": res["WinRate"],
                    "Beat_BH": float(res["Final"] > r_bh["Final"]),
                    "ZeroTrades": float(res["Trades"] == 0),
                }
            )

    print(f"Major -55%+ Troughs Caught (within [-2w, +24w], total={total_troughs}):")
    for m in mask_names:
        cnt = trough_hits[m]
        print(f"  {m:24s}: {cnt}/{total_troughs} ({cnt/total_troughs*100:.1f}%)")

    df_rec = pd.DataFrame(recs)
    agg = []
    for sk in strat_keys:
        sub = df_rec[df_rec["System"] == sk]
        agg.append(
            {
                "System": sk,
                "Beat_BH%": round(float(sub["Beat_BH"].mean() * 100.0), 1),
                "ZeroTrade_Worlds": int(sub["ZeroTrades"].sum()),
                "P25_Final": round(float(sub["Final"].quantile(0.25)), 0),
                "Med_Final": round(float(sub["Final"].median()), 0),
                "P75_Final": round(float(sub["Final"].quantile(0.75)), 0),
                "Med_CAGR%": round(float(sub["CAGR"].median()), 2),
                "Med_MaxDD%": round(float(sub["MaxDD"].median()), 2),
                "Med_Calmar": round(float(sub["Calmar"].median()), 3),
                "Med_Trades": round(float(sub["Trades"].median()), 1),
                "Med_WinRate%": round(float(sub["WinRate"].median()), 1),
            }
        )
    print("\n50-World Monte Carlo Summary:")
    print(pd.DataFrame(agg).to_string(index=False))


if __name__ == "__main__":
    main()
