"""Reconstruct and stress-test 'Can buy wrong, NEVER misses' Monthly+Weekly LOG Fisher strategies across SOXL, TQQQ, and 50 synthetic worlds."""

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
from indicators import fisher_transform
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL, load_adjusted
from simulate_synthetic_soxl import build_study_data_from_dfs, generate_synthetic_pair
from study_major_exits_and_backtest import compute_metrics, run_buy_and_hold
from test_cycle_v2_and_hybrid import run_strategy1_baseline


def add_log_monthly_and_weekly_fisher(data: dict) -> dict:
    """Add strictly causal linear & LOG-price monthly/weekly Fisher cross flags to weekly dataframe."""
    d = data["daily"]
    w = data["weekly"].copy()

    # Completed monthly bars
    with_dt = d.assign(_dt=d.index)
    m = (
        with_dt.groupby(pd.Grouper(freq="ME"))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    m.index = pd.DatetimeIndex(m["_dt"])

    # Linear Monthly Fisher(10)
    mf_lin = fisher_transform(m["High"], m["Low"], period=10)
    m["MF_Lin"] = mf_lin["fisher"]
    m["MF_Lin_Sig"] = mf_lin["fisher_signal"]
    m["MF_Lin_CrossUp"] = (m["MF_Lin"] > m["MF_Lin_Sig"]) & (m["MF_Lin"].shift(1) <= m["MF_Lin_Sig"].shift(1))
    m["MF_Lin_CrossDn"] = (m["MF_Lin"] < m["MF_Lin_Sig"]) & (m["MF_Lin"].shift(1) >= m["MF_Lin_Sig"].shift(1))

    # LOG-price Monthly Fisher(10)
    mf_log = fisher_transform(np.log(m["High"].clip(lower=1e-6)), np.log(m["Low"].clip(lower=1e-6)), period=10)
    m["MF_Log"] = mf_log["fisher"]
    m["MF_Log_Sig"] = mf_log["fisher_signal"]
    m["MF_Log_CrossUp"] = (m["MF_Log"] > m["MF_Log_Sig"]) & (m["MF_Log"].shift(1) <= m["MF_Log_Sig"].shift(1))
    m["MF_Log_CrossDn"] = (m["MF_Log"] < m["MF_Log_Sig"]) & (m["MF_Log"].shift(1) >= m["MF_Log_Sig"].shift(1))

    # Map monthly cross events ONLY onto the first completed weekly bar at or after month-end (causal, one-shot!)
    m_cross_up_lin_dates = set(m.index[m["MF_Lin_CrossUp"] & (m["MF_Lin"].shift(1) < 0.8)])
    m_cross_up_log_dates = set(m.index[m["MF_Log_CrossUp"] & (m["MF_Log"].shift(1) < 0.8)])

    # Check exact alignment: month-end trading day is ALWAYS a week-end or inside a week!
    # Map each month-end cross date `mdt` to the FIRST weekly index `w_dt >= mdt`!
    w_idx = w.index
    w["MF_Lin_Buy"] = False
    w["MF_Log_Buy"] = False
    for mdt in m_cross_up_lin_dates:
        pos = int(w_idx.searchsorted(mdt, side="left"))
        if pos < len(w_idx):
            w.iloc[pos, w.columns.get_loc("MF_Lin_Buy")] = True
    for mdt in m_cross_up_log_dates:
        pos = int(w_idx.searchsorted(mdt, side="left"))
        if pos < len(w_idx):
            w.iloc[pos, w.columns.get_loc("MF_Log_Buy")] = True

    # Also align continuous M_Fish_Log & M_Fish_Log_Sig via ffill for exit/diagnostics
    m_al = m[["MF_Log", "MF_Log_Sig"]].reindex(w_idx, method="ffill")
    w["MF_Log"] = m_al["MF_Log"]
    w["MF_Log_Sig"] = m_al["MF_Log_Sig"]

    # LOG Weekly Fisher(10) already in w as W_Fish_Log / W_Fish_Log_Sig!
    wf_log_cross_up = (w["W_Fish_Log"] > w["W_Fish_Log_Sig"]) & (w["W_Fish_Log"].shift(1) <= w["W_Fish_Log_Sig"].shift(1))
    min4_wf_log = w["W_Fish_Log"].rolling(4, min_periods=1).min()
    # 52w drawdown using Low
    dd52_cur = (w["Low"] / w["High"].rolling(52, min_periods=1).max() - 1.0) * 100.0
    w["DD52_Low%"] = dd52_cur

    # Relaxed weekly bottom confirmation ("can buy wrong, won't miss"):
    # Either (A) deep 52w drawdown <= -50% + W_Fish_Log oversold cross-up + Close > prev_W_High
    # Or     (B) near 200W (<=1.20) + W_Fish_Log < -1.5 + Close > prev_W_High
    near_200w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min() <= 1.20
    w["W_LogBot_Buy"] = (
        (min4_wf_log < -1.5)
        & wf_log_cross_up
        & (w["Close"] > w["High"].shift(1))
        & ((dd52_cur.rolling(8, min_periods=1).min() <= -50.0) | near_200w.fillna(False))
    )

    # Relaxed F0-style (doesn't require 1.6x volume or 200W if DD52 <= -55%, plus volume spike option)
    w["W_EarlyBot_Buy"] = (
        (w["W_Fisher"].rolling(4, min_periods=1).min() < -1.5)
        & (w["Close"] > w["High"].shift(1))
        & (
            (dd52_cur.rolling(8, min_periods=1).min() <= -55.0)
            | (near_200w.fillna(False) & (w["max4w_Vol_R"] >= 1.4))
        )
    )

    out = dict(data)
    out["weekly"] = w
    out["monthly_df"] = m
    return out


def list_monthly_crosses(m: pd.DataFrame, d: pd.DataFrame, start_date: str = "2011-01-01") -> pd.DataFrame:
    """List every completed-month Linear & LOG Fisher golden cross with forward 12m return."""
    rows = []
    sub_m = m[m.index >= pd.Timestamp(start_date)]
    for dt, r in sub_m.iterrows():
        lin_up = bool(r["MF_Lin_CrossUp"])
        log_up = bool(r["MF_Log_CrossUp"])
        if not (lin_up or log_up):
            continue
        # Next trading day open
        pos = int(d.index.searchsorted(dt, side="right"))
        if pos >= len(d):
            continue
        edate = d.index[pos].strftime("%Y-%m-%d")
        epx = float(d["Open"].iloc[pos])
        fwd_1y = d.iloc[pos : min(len(d), pos + 253)]
        max_1y_ret = (float(fwd_1y["Close"].max()) / epx - 1.0) * 100.0
        end_1y_ret = (float(fwd_1y["Close"].iloc[-1]) / epx - 1.0) * 100.0
        rows.append(
            {
                "MonthEnd": dt.strftime("%Y-%m-%d"),
                "NextOpenDate": edate,
                "OpenPx": round(epx, 2),
                "LinCross": "YES" if lin_up else "-",
                "MF_Lin": round(float(r["MF_Lin"]), 2),
                "LogCross": "YES" if log_up else "-",
                "MF_Log": round(float(r["MF_Log"]), 2),
                "Fwd1Y_End%": round(end_1y_ret, 1),
                "Fwd1Y_Max%": round(max_1y_ret, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 20)

    soxl = add_log_monthly_and_weekly_fisher(prepare_extended_data("SOXL", "SMH"))
    tqqq = add_log_monthly_and_weekly_fisher(prepare_extended_data("TQQQ", "QQQ"))

    # =========================================================================
    # 1. INSPECT ALL MONTHLY FISHER GOLDEN CROSSES ON REAL SOXL (2011-2026)
    # =========================================================================
    print("=" * 115)
    print("1. ALL REAL SOXL COMPLETED-MONTH FISHER(10) GOLDEN CROSSES (LINEAR vs LOG, 2011-2026)")
    print("=" * 115)
    print(list_monthly_crosses(soxl["monthly_df"], soxl["daily"], "2011-01-01").to_string(index=False))

    # =========================================================================
    # 2. COMPARE 5 'NEVER-MISS' RECONSTRUCTIONS vs F1+TP3h & STRATEGY 1
    # =========================================================================
    def make_masks(ds: dict) -> dict[str, np.ndarray]:
        w = ds["weekly"]
        f0 = build_entry_mask(ds, None, None)
        f1 = build_entry_mask(ds, 30, -80.0)
        mf_lin = w["MF_Lin_Buy"].to_numpy(dtype=bool)
        mf_log = w["MF_Log_Buy"].to_numpy(dtype=bool)
        w_log = w["W_LogBot_Buy"].fillna(False).to_numpy(dtype=bool)
        w_early = w["W_EarlyBot_Buy"].fillna(False).to_numpy(dtype=bool)
        return {
            "F1_Ref": f1,
            "R1_Pure_MFishLin": mf_lin,
            "R2_Pure_MFishLog": mf_log,
            "R3_F0_or_MFishLin": f0 | mf_lin,
            "R4_WLogBot_or_MFishLog": w_log | mf_log,
            "R5_WEarly_or_MFishLog": w_early | mf_log,
        }

    print("\n" + "=" * 115)
    print("2. BACKTEST COMPARISON ON REAL SOXL (2011-2026 & 2015-2026) AND REAL TQQQ (2015-2026)")
    print("=" * 115)
    for title, ds, tk, bk, s, e in [
        ("SOXL Early+Full (2011-06-01 ~ 2026-11-19)", soxl, "SOXL", "SMH", "2011-06-01", "2026-11-19"),
        ("SOXL Main (2015-01-01 ~ 2026-11-19)", soxl, "SOXL", "SMH", "2015-01-01", "2026-11-19"),
        ("TQQQ Main (2015-01-01 ~ 2026-11-19)", tqqq, "TQQQ", "QQQ", "2015-01-01", "2026-11-19"),
    ]:
        masks = make_masks(ds)
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

    # Print detailed trade logs of R4 and R5 on SOXL (2011-2026)
    soxl_masks = make_masks(soxl)
    for mname in ("R2_Pure_MFishLog", "R4_WLogBot_or_MFishLog", "R5_WEarly_or_MFishLog"):
        r = run_backtest_param(soxl, soxl_masks[mname], "TP3h_Base", "2011-06-01", "2026-11-19", filter_tag=mname)
        print(f"\n[SOXL 2011-2026 Trade Log: {mname}+TP3h] Final=${r['Final']:,.2f} CAGR={r['CAGR']}% MaxDD={r['MaxDD']}% Trades={r['Trades']} WinRate={r['WinRate']}%")
        print(pd.DataFrame(r["trade_logs"]).to_string(index=False))

    # =========================================================================
    # 3. 50-WORLD SYNTHETIC SOXL STRESS TEST: DOES IT FIX MISSED BOTTOMS?
    # =========================================================================
    print("\n" + "=" * 115)
    print("3. 50-WORLD SYNTHETIC SOXL TEST (2015-2026): F1+TP3h vs 'NEVER-MISS' LOG+MONTHLY FISHER")
    print("=" * 115)
    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)

    strat_keys = ["Buy&Hold", "S1", "F1+TP3h", "R2_Pure_MFishLog+TP3h", "R4_WLogBot_or_MFishLog+TP3h", "R5_WEarly_or_MFishLog+TP3h"]
    recs = []
    trough_hits = {k: 0 for k in ("F1_Ref", "R2_Pure_MFishLog", "R4_WLogBot_or_MFishLog", "R5_WEarly_or_MFishLog")}
    total_troughs = 0

    for w_i in range(50):
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, 100 + w_i)
        sd = add_log_monthly_and_weekly_fisher(build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH"))
        w_s = sd["weekly"]
        masks = make_masks(sd)

        # Count -55%+ troughs caught within [-2w, +24w] (allows monthly cross slightly after trough)
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
            for k in trough_hits:
                if bool(masks[k][a0:a1].any()):
                    trough_hits[k] += 1

        r_bh = run_buy_and_hold(sd, "2015-01-01", "2026-11-19")
        r_s1 = run_strategy1_baseline.__module__  # use simulate from simulate_synthetic_soxl
        from simulate_synthetic_soxl import run_strategy1_from_dfs
        r_s1 = run_strategy1_from_dfs(syn_soxl, syn_smh, "2015-01-01", "2026-11-19")
        r_f1 = run_backtest_param(sd, masks["F1_Ref"], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="F1")
        r_r2 = run_backtest_param(sd, masks["R2_Pure_MFishLog"], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="R2")
        r_r4 = run_backtest_param(sd, masks["R4_WLogBot_or_MFishLog"], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="R4")
        r_r5 = run_backtest_param(sd, masks["R5_WEarly_or_MFishLog"], "TP3h_Base", "2015-01-01", "2026-11-19", filter_tag="R5")

        for sk, res in zip(strat_keys, [r_bh, r_s1, r_f1, r_r2, r_r4, r_r5]):
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

    print(f"Major -55%+ Troughs Caught (within [-2w, +24w] of trough, total={total_troughs}):")
    for k, cnt in trough_hits.items():
        print(f"  {k:24s}: {cnt}/{total_troughs} ({cnt/total_troughs*100:.1f}%)")

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
