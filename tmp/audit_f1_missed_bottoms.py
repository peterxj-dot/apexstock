"""Audit missed-bottom risk of SOXL F1+TP3h: historical margins, 2026 crash check, TQQQ misses, and 50-world failure breakdown."""

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
from simulate_synthetic_soxl import build_study_data_from_dfs, generate_synthetic_pair


def find_major_drawdown_troughs(w: pd.DataFrame, min_dd_pct: float = -40.0, sep_weeks: int = 26) -> list[dict]:
    """Identify distinct major drawdown troughs (at least `sep_weeks` apart) on weekly data."""
    hi = w["High"].to_numpy(dtype=np.float64)
    lo = w["Low"].to_numpy(dtype=np.float64)
    cl = w["Close"].to_numpy(dtype=np.float64)
    dates = w.index

    # Running 52-week high drawdown using Low
    roll_hi = pd.Series(hi).rolling(52, min_periods=1).max().to_numpy(dtype=np.float64)
    dd_pct = (lo / roll_hi - 1.0) * 100.0

    # Greedy local-minimum selection by deepest drawdown separated by sep_weeks
    candidates = np.where(dd_pct <= min_dd_pct)[0]
    order = candidates[np.argsort(dd_pct[candidates])]
    selected: list[int] = []
    for idx in order:
        if all(abs(idx - s) >= sep_weeks for s in selected):
            selected.append(int(idx))
    selected.sort()

    out = []
    for t_pos in selected:
        # Look in [t_pos - 2, t_pos + 12] window for gate satisfaction
        s_w = max(0, t_pos - 2)
        e_w = min(len(w) - 1, t_pos + 12)
        sub = w.iloc[s_w : e_w + 1]

        min_200w = float((1.0 + sub["Low_vs_200W%"] / 100.0).min()) if sub["Low_vs_200W%"].notna().any() else float("nan")
        min_wf = float(sub["W_Fisher"].min())
        max_volr = float(sub["W_Vol_R"].max())
        max_wks52 = float(sub["wks_52H"].max())
        min_dd52 = float(sub["dd52_8wL%"].min())

        # Did F0 or F1 trigger within [t_pos - 2, t_pos + 16]?
        sub_trig = w.iloc[s_w : min(len(w), t_pos + 17)]
        f0_hits = sub_trig[sub_trig["F0_Base_E2"].fillna(False)]
        f1_hits = sub_trig[sub_trig["F1_Wks30_or_DD80_E2"].fillna(False)]

        # Forward 52-week max gain from trough close (or remaining bars)
        fwd = w.iloc[t_pos : min(len(w), t_pos + 53)]
        fwd_max_ret = (float(fwd["Close"].max()) / float(cl[t_pos]) - 1.0) * 100.0

        out.append(
            {
                "Trough_Week": dates[t_pos].strftime("%Y-%m-%d"),
                "Trough_Low": round(float(lo[t_pos]), 2),
                "DD52_Low%": round(float(dd_pct[t_pos]), 1),
                "Min_Low/200W": round(min_200w, 3) if not np.isnan(min_200w) else np.nan,
                "Min_WFish": round(min_wf, 2),
                "Max_WVolR": round(max_volr, 2),
                "Max_Wks52H": int(max_wks52),
                "F0_Trig": f0_hits.index[0].strftime("%Y-%m-%d") if len(f0_hits) else "MISS",
                "F1_Trig": f1_hits.index[0].strftime("%Y-%m-%d") if len(f1_hits) else "MISS",
                "Fwd52w_Max%": round(fwd_max_ret, 1),
            }
        )
    return out


def audit_entry_margins(w: pd.DataFrame, trade_logs: list[dict]) -> pd.DataFrame:
    """Measure exact safety margins of all 4 gates at each actual F1 entry signal week."""
    rows = []
    for t in trade_logs:
        ed = pd.Timestamp(t["entry_date"])
        # Signal week is the completed week immediately preceding entry_date
        sig_sub = w[w.index < ed]
        s_row = sig_sub.iloc[-1]
        sig_dt = sig_sub.index[-1]

        low_200w = float((1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min().loc[sig_dt])
        min_wf4 = float(w["W_Fisher"].rolling(4, min_periods=1).min().loc[sig_dt])
        max_vr4 = float(s_row["max4w_Vol_R"])
        wks52 = int(s_row["wks_52H"])
        dd52 = float(s_row["dd52_8wL%"])

        rows.append(
            {
                "Entry_Date": t["entry_date"],
                "Sig_Week": sig_dt.strftime("%Y-%m-%d"),
                "Low/200W(<=1.10)": round(low_200w, 3),
                "Margin_200W": round(1.10 - low_200w, 3),
                "WF4(<-1.50)": round(min_wf4, 2),
                "Margin_WF": round(-1.50 - min_wf4, 2),
                "VolR4(>=1.60)": round(max_vr4, 2),
                "Margin_VolR": round(max_vr4 - 1.60, 2),
                "Wks52H(>=30)": wks52,
                "DD52%(<=-80)": round(dd52, 1),
                "F1_By": "BOTH" if (wks52 >= 30 and dd52 <= -80.0) else ("WKS>=30" if wks52 >= 30 else "DD<=-80"),
                "Trade_Ret%": t["ret_pct"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)

    soxl = prepare_extended_data("SOXL", "SMH")
    tqqq = prepare_extended_data("TQQQ", "QQQ")
    f1_mask_s = build_entry_mask(soxl, 30, -80.0)
    r_f1_s = run_backtest_param(soxl, f1_mask_s, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F1")

    # =========================================================================
    # 1. ALL REAL SOXL MAJOR DRAWDOWN TROUGHS (2011 - 2026)
    # =========================================================================
    print("=" * 120)
    print("1. ALL REAL SOXL MAJOR DRAWDOWN TROUGHS (DD <= -40%, 2011-2026)")
    print("=" * 120)
    troughs_s = find_major_drawdown_troughs(soxl["weekly"], min_dd_pct=-40.0, sep_weeks=26)
    print(pd.DataFrame(troughs_s).to_string(index=False))

    # =========================================================================
    # 2. EXACT GATE SAFETY MARGINS ON THE 5 ACTUAL SOXL F1 ENTRIES
    # =========================================================================
    print("\n" + "=" * 120)
    print("2. EXACT GATE SAFETY MARGINS AT THE 5 REAL SOXL F1 ENTRY SIGNALS (HOW CLOSE TO MISSING?)")
    print("=" * 120)
    print(audit_entry_margins(soxl["weekly"], r_f1_s["trade_logs"]).to_string(index=False))

    # =========================================================================
    # 3. CROSS-ASSET CHECK: REAL TQQQ MAJOR TROUGHS & WHY F1 MISSED THEM
    # =========================================================================
    print("\n" + "=" * 120)
    print("3. REAL TQQQ MAJOR DRAWDOWN TROUGHS (DD <= -35%, 2015-2026) — F0 vs F1")
    print("=" * 120)
    w_t = tqqq["weekly"]
    troughs_t = find_major_drawdown_troughs(w_t[w_t.index >= "2015-01-01"], min_dd_pct=-35.0, sep_weeks=26)
    print(pd.DataFrame(troughs_t).to_string(index=False))

    # =========================================================================
    # 4. 50 SYNTHETIC SOXL WORLDS: QUANTIFY MISSED BOTTOMS & ROOT CAUSES
    # =========================================================================
    print("\n" + "=" * 120)
    print("4. 50-WORLD SYNTHETIC SOXL MISSED-BOTTOM BREAKDOWN (2015-2026, DD52 <= -55% TROUGHS)")
    print("=" * 120)
    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)

    total_troughs = 0
    caught_f0 = 0
    caught_f1 = 0
    fail_200w = 0
    fail_wf = 0
    fail_volr = 0
    fail_f1_gate = 0
    fail_e2_break = 0
    sl_wiped_worlds = 0
    zero_trade_worlds = 0

    for w_i in range(50):
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, 100 + w_i)
        sd = build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH")
        w_s = sd["weekly"]
        f0_m = build_entry_mask(sd, None, None)
        f1_m = build_entry_mask(sd, 30, -80.0)
        w_s["F0_Base_E2"] = f0_m
        w_s["F1_Wks30_or_DD80_E2"] = f1_m
        w_bt = w_s[w_s.index >= "2015-01-01"]
        res_f1 = run_backtest_param(sd, f1_m, "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F1")
        if res_f1["Trades"] == 0:
            zero_trade_worlds += 1
        elif all(t["exit_reason"] == "SL_W25%" for t in res_f1["trade_logs"]):
            sl_wiped_worlds += 1

        # Identify major -55%+ troughs in 2015-2026
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

        for t_pos in sel:
            total_troughs += 1
            sub = w_bt.iloc[max(0, t_pos - 2) : min(len(w_bt), t_pos + 17)]
            has_f0 = bool(sub["F0_Base_E2"].fillna(False).any())
            has_f1 = bool(sub["F1_Wks30_or_DD80_E2"].fillna(False).any())
            if has_f0:
                caught_f0 += 1
            if has_f1:
                caught_f1 += 1
            else:
                # Diagnose which gate(s) failed across the bottom window
                ok_200w = bool(((1.0 + sub["Low_vs_200W%"] / 100.0).min()) <= 1.10)
                ok_wf = bool((sub["W_Fisher"].min()) < -1.5)
                ok_vol = bool((sub["max4w_Vol_R"].max()) >= 1.6)
                ok_f1 = bool(((sub["wks_52H"] >= 30) | (sub["dd52_8wL%"] <= -80.0)).any())
                if not ok_200w:
                    fail_200w += 1
                if not ok_wf:
                    fail_wf += 1
                if not ok_vol:
                    fail_volr += 1
                if ok_200w and ok_wf and ok_vol and not ok_f1:
                    fail_f1_gate += 1
                if ok_200w and ok_wf and ok_vol and ok_f1:
                    fail_e2_break += 1

    print(f"Total -55%+ troughs across 50 synthetic worlds (2015-2026): {total_troughs} ({total_troughs/50:.2f}/world)")
    print(f"  Caught by F0 (pre_c7_vol + E2) : {caught_f0}/{total_troughs} ({caught_f0/total_troughs*100:.1f}%)")
    print(f"  Caught by F1 (F0 + Wks30|DD80) : {caught_f1}/{total_troughs} ({caught_f1/total_troughs*100:.1f}%)")
    missed = total_troughs - caught_f1
    print(f"  Missed by F1                   : {missed}/{total_troughs} ({missed/total_troughs*100:.1f}%)")
    print("  Failure reasons among missed troughs (non-exclusive for 200W/WF/Vol):")
    print(f"    - Price never reached <= 1.10 * 200W SMA : {fail_200w}/{missed} ({fail_200w/missed*100:.1f}%)")
    print(f"    - Weekly Fisher never < -1.5             : {fail_wf}/{missed} ({fail_wf/missed*100:.1f}%)")
    print(f"    - 4W Peak Volume never >= 1.6x           : {fail_volr}/{missed} ({fail_volr/missed*100:.1f}%)")
    print(f"    - F0 passed, blocked ONLY by F1(30w|-80%): {fail_f1_gate}/{missed} ({fail_f1_gate/missed*100:.1f}%)")
    print(f"    - All 4 gates passed, no E2 breakout     : {fail_e2_break}/{missed} ({fail_e2_break/missed*100:.1f}%)")
    print(f"  Worlds with 0 F1 trades (12yr cash)        : {zero_trade_worlds}/50")
    print(f"  Worlds where ALL F1 trades hit SL_W25%     : {sl_wiped_worlds}/50")


if __name__ == "__main__":
    main()
