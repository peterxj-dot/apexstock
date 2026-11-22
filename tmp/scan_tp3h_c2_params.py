"""Fine 3D parameter plateau scan of TP3h (hold_th x str_th x mf_th) on C2_F1_or_MFishLin, plus 50-world & TQQQ verification."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from improve_tp3h_and_robustness import build_entry_mask, prepare_extended_data, run_backtest_param
from scan_soxl_strategies import load_adjusted
from simulate_synthetic_soxl import build_study_data_from_dfs, generate_synthetic_pair
from study_monthly_fisher_never_miss import add_log_monthly_and_weekly_fisher


def make_c2_mask(ds: dict) -> np.ndarray:
    f1 = build_entry_mask(ds, 30, -80.0)
    mf_lin = ds["weekly"]["MF_Lin_Buy"].fillna(False).to_numpy(dtype=bool)
    return f1 | mf_lin


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 24)

    soxl = add_log_monthly_and_weekly_fisher(prepare_extended_data("SOXL", "SMH"))
    tqqq = add_log_monthly_and_weekly_fisher(prepare_extended_data("TQQQ", "QQQ"))
    c2_s = make_c2_mask(soxl)
    f0_t = build_entry_mask(tqqq, None, None)

    holds = [42, 45, 48, 50, 52, 55, 58, 60]
    strs = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    mfs = [1.8, 1.9, 2.0, 2.1, 2.2]

    records = []
    for h in holds:
        for s in strs:
            for mf in mfs:
                r12 = run_backtest_param(soxl, c2_s, "TP3h_Base", "2015-01-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
                r15 = run_backtest_param(soxl, c2_s, "TP3h_Base", "2011-06-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
                r11 = run_backtest_param(soxl, c2_s, "TP3h_Base", "2015-01-01", "2025-12-31", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
                records.append(
                    {
                        "hold": h,
                        "str": int(s),
                        "mf": mf,
                        "S12_Final": r12["Final"],
                        "S12_CAGR%": r12["CAGR"],
                        "S12_DD%": r12["MaxDD"],
                        "S12_Calmar": r12["Calmar"],
                        "S15_Final": r15["Final"],
                        "S15_CAGR%": r15["CAGR"],
                        "S15_DD%": r15["MaxDD"],
                        "S15_Calmar": r15["Calmar"],
                        "S11_Final": r11["Final"],
                    }
                )

    df = pd.DataFrame(records)

    # Compute 3x3x3 local neighborhood median on S15_Final & S12_Final to separate broad plateaus from lucky spikes
    h_idx = {v: i for i, v in enumerate(holds)}
    s_idx = {int(v): i for i, v in enumerate(strs)}
    m_idx = {v: i for i, v in enumerate(mfs)}

    nb15_med = []
    nb12_med = []
    nb15_min = []
    for _, row in df.iterrows():
        hi, si, mi = h_idx[int(row["hold"])], s_idx[int(row["str"])], m_idx[float(row["mf"])]
        h_ok = [holds[j] for j in range(max(0, hi - 1), min(len(holds), hi + 2))]
        s_ok = [int(strs[j]) for j in range(max(0, si - 1), min(len(strs), si + 2))]
        m_ok = [mfs[j] for j in range(max(0, mi - 1), min(len(mfs), mi + 2))]
        sub = df[df["hold"].isin(h_ok) & df["str"].isin(s_ok) & df["mf"].isin(m_ok)]
        nb15_med.append(float(sub["S15_Final"].median()))
        nb12_med.append(float(sub["S12_Final"].median()))
        nb15_min.append(float(sub["S15_Final"].min()))

    df["NB15_Med"] = nb15_med
    df["NB12_Med"] = nb12_med
    df["NB15_Min"] = nb15_min

    print("=" * 125)
    print("PART 1: 280-COMBO TP3h SCAN ON SOXL C2_F1_or_MFishLin (TOP 15 BY 15.5yr FINAL & TOP 12 BY PLATEAU MEDIAN)")
    print("=" * 125)
    cols_show = ["hold", "str", "mf", "S12_Final", "S12_CAGR%", "S15_Final", "S15_CAGR%", "S15_DD%", "S11_Final", "NB15_Med", "NB15_Min"]
    fmt = df.copy()
    for c in ("S12_Final", "S15_Final", "S11_Final", "NB15_Med", "NB15_Min"):
        fmt[c] = (fmt[c] / 1e6).round(2)
    fmt = fmt.rename(columns={
        "S12_Final": "S12($M)", "S15_Final": "S15($M)", "S11_Final": "Ex26($M)",
        "NB15_Med": "NB15_Med($M)", "NB15_Min": "NB15_Min($M)",
    })
    show_c = ["hold", "str", "mf", "S12($M)", "S12_CAGR%", "S15($M)", "S15_CAGR%", "S15_DD%", "Ex26($M)", "NB15_Med($M)", "NB15_Min($M)"]

    print("\n[Top 15 Single-Point by 15.5yr Final]")
    print(fmt.sort_values("S15($M)", ascending=False).head(15)[show_c].to_string(index=False))

    print("\n[Top 12 Broadest 3x3x3 Neighborhood Plateau (by NB15_Med)]")
    print(fmt.sort_values("NB15_Med($M)", ascending=False).head(12)[show_c].to_string(index=False))

    print("\n[Baseline Reference: (hold=45, str=40, mf=2.0)]")
    print(fmt[(fmt["hold"] == 45) & (fmt["str"] == 40) & (fmt["mf"] == 2.0)][show_c].to_string(index=False))

    # =========================================================================
    # PART 2: 2D SLICE AT mf=2.0 (hold x str) TO VISUALIZE THE PLATEAU
    # =========================================================================
    print("\n" + "=" * 125)
    print("PART 2: 2D HEATMAP MATRIX AT mf=2.0 — 15.5yr FINAL ($M) [hold x str]")
    print("=" * 125)
    piv15 = fmt[fmt["mf"] == 2.0].pivot(index="hold", columns="str", values="S15($M)")
    print(piv15.to_string())

    print("\n[Same 2D Matrix at mf=2.0 — Ex-2026 (2015-2025) FINAL ($M)]")
    piv11 = fmt[fmt["mf"] == 2.0].pivot(index="hold", columns="str", values="Ex26($M)")
    print(piv11.to_string())

    # =========================================================================
    # PART 3: TEST TOP CANDIDATES vs BASE (45,40,2.0) ON TQQQ + 50 SYNTHETIC WORLDS
    # =========================================================================
    top_pt = df.sort_values("S15_Final", ascending=False).iloc[0]
    top_nb = df.sort_values("NB15_Med", ascending=False).iloc[0]
    top_ex26 = df.sort_values("S11_Final", ascending=False).iloc[0]

    candidates = [
        ("Base_(45,40,2.0)", 45, 40.0, 2.0),
        ("Peak15y", int(top_pt["hold"]), float(top_pt["str"]), float(top_pt["mf"])),
        ("PlateauBest", int(top_nb["hold"]), float(top_nb["str"]), float(top_nb["mf"])),
        ("Ex26Best", int(top_ex26["hold"]), float(top_ex26["str"]), float(top_ex26["mf"])),
        ("Cand_(50,40,2.0)", 50, 40.0, 2.0),
        ("Cand_(52,35,2.0)", 52, 35.0, 2.0),
        ("Cand_(55,35,2.0)", 55, 35.0, 2.0),
    ]
    # Deduplicate identical (h, s, mf) while preserving order
    seen_p = set()
    uniq_cands = []
    for lbl, h, s, mf in candidates:
        if (h, s, mf) not in seen_p:
            seen_p.add((h, s, mf))
            uniq_cands.append((f"{lbl}[{h},{int(s)},{mf}]", h, s, mf))

    print("\n" + "=" * 125)
    print("PART 3: CROSS-ASSET (TQQQ F0) & 50-WORLD SYNTHETIC SOXL VERIFICATION OF TOP TP3h CANDIDATES")
    print("=" * 125)

    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)
    syn_worlds = []
    for w_i in range(50):
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, 100 + w_i)
        sd = add_log_monthly_and_weekly_fisher(build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH"))
        syn_worlds.append((sd, make_c2_mask(sd)))

    summary = []
    for cname, h, s, mf in uniq_cands:
        rs12 = run_backtest_param(soxl, c2_s, "TP3h_Base", "2015-01-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
        rs15 = run_backtest_param(soxl, c2_s, "TP3h_Base", "2011-06-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
        rtq = run_backtest_param(tqqq, f0_t, "TP3h_Base", "2015-01-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="F0")

        mc_finals, mc_cagrs, mc_dds, mc_cals = [], [], [], []
        for sd, cmask in syn_worlds:
            rmc = run_backtest_param(sd, cmask, "TP3h_Base", "2015-01-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
            mc_finals.append(rmc["Final"])
            mc_cagrs.append(rmc["CAGR"])
            mc_dds.append(rmc["MaxDD"])
            mc_cals.append(rmc["Calmar"])

        summary.append(
            {
                "Candidate": cname,
                "S12($M)": round(rs12["Final"] / 1e6, 2),
                "S15($M)": round(rs15["Final"] / 1e6, 2),
                "S15_DD%": rs15["MaxDD"],
                "TQ_F0($K)": round(rtq["Final"] / 1e3, 1),
                "TQ_CAGR%": rtq["CAGR"],
                "MC50_Med($K)": round(float(np.median(mc_finals)) / 1e3, 1),
                "MC50_CAGR%": round(float(np.median(mc_cagrs)), 2),
                "MC50_DD%": round(float(np.median(mc_dds)), 2),
                "MC50_Calmar": round(float(np.median(mc_cals)), 3),
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))

    # Print trade log of top Peak15y and PlateauBest on SOXL 2011-2026
    for lbl, h, s, mf in uniq_cands[:3]:
        r = run_backtest_param(soxl, c2_s, "TP3h_Base", "2011-06-01", "2026-11-19", hold_th=h, str_th=s, mf_th=mf, filter_tag="C2")
        print(f"\n[SOXL 2011-2026 Trade Log: {lbl}] Final=${r['Final']:,.2f} MaxDD={r['MaxDD']}%")
        print(pd.DataFrame(r["trade_logs"]).to_string(index=False))

    print(f"\nCompleted 280-combo scan + 50-world verification in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
