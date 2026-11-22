"""Bottom-up anatomy and entry-rule study of SOXL's true major bottoms (2014-2026)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, fisher_transform, rsi, sma
from scan_soxl_strategies import load_adjusted


def build_weekly_dataset(target_df: pd.DataFrame, bench_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly causal weekly bars with linear/log Fisher, 30W/200W SMAs, and aligned benchmark."""
    def agg_weekly(df: pd.DataFrame) -> pd.DataFrame:
        with_dt = df.assign(_dt=df.index)
        g = (
            with_dt.groupby(pd.Grouper(freq="W-FRI"))
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
            .dropna(subset=["Open", "Close", "_dt"])
        )
        g.index = pd.DatetimeIndex(g["_dt"])
        return g.drop(columns=["_dt"])

    def agg_monthly(df: pd.DataFrame) -> pd.DataFrame:
        with_dt = df.assign(_dt=df.index)
        g = (
            with_dt.groupby(pd.Grouper(freq="ME"))
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
            .dropna(subset=["Open", "Close", "_dt"])
        )
        g.index = pd.DatetimeIndex(g["_dt"])
        return g.drop(columns=["_dt"])

    w = agg_weekly(target_df)
    m = agg_monthly(target_df)
    bw = agg_weekly(bench_df).reindex(w.index, method="ffill")

    # Target weekly indicators
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Sig"] = wf["fisher_signal"]
    w["dW_Fish"] = w["W_Fisher"].diff()

    wf_log = fisher_transform(np.log(w["High"]), np.log(w["Low"]), period=10)
    w["W_Fish_Log"] = wf_log["fisher"]
    w["W_Fish_Log_Sig"] = wf_log["fisher_signal"]
    w["dW_Fish_Log"] = w["W_Fish_Log"].diff()

    w["W_SMA30"] = sma(w["Close"], 30)
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_EMA10"] = ema(w["Close"], 10)
    w["W_RSI14"] = rsi(w["Close"], 14)
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()

    w["Close_vs_200W%"] = (w["Close"] / w["W_SMA200"] - 1.0) * 100.0
    w["Low_vs_200W%"] = (w["Low"] / w["W_SMA200"] - 1.0) * 100.0
    w["Close_vs_30W%"] = (w["Close"] / w["W_SMA30"] - 1.0) * 100.0
    w["LogDist_200W"] = np.log(w["Close"] / w["W_SMA200"])

    roll52_hi = w["High"].rolling(52, min_periods=10).max()
    w["DD_52W%"] = (w["Close"] / roll52_hi - 1.0) * 100.0
    w["Low_DD_52W%"] = (w["Low"] / roll52_hi - 1.0) * 100.0

    # Completed monthly Fisher aligned strictly causally to weekly index
    mf = fisher_transform(m["High"], m["Low"], period=10)
    mf_log = fisher_transform(np.log(m["High"]), np.log(m["Low"]), period=10)
    m_ind = pd.DataFrame({"M_Fish_prev": mf["fisher"], "M_Fish_Log_prev": mf_log["fisher"]}, index=m.index)
    m_aligned = m_ind.reindex(w.index, method="ffill")
    w["M_Fish_prev"] = m_aligned["M_Fish_prev"]
    w["M_Fish_Log_prev"] = m_aligned["M_Fish_Log_prev"]

    # Benchmark weekly indicators
    bw_f = fisher_transform(bw["High"], bw["Low"], period=10)
    bw_sma200 = sma(bw["Close"], 200)
    w["SMH_vs_200W%"] = (bw["Close"] / bw_sma200 - 1.0) * 100.0
    w["SMH_Low_vs_200W%"] = (bw["Low"] / bw_sma200 - 1.0) * 100.0
    w["SMH_W_Fisher"] = bw_f["fisher"]
    w["SMH_dW_Fish"] = w["SMH_W_Fisher"].diff()

    return w, target_df


def evaluate_bottom_rule(
    name: str,
    cond: pd.Series,
    w: pd.DataFrame,
    daily_df: pd.DataFrame,
    trough_indices: dict[str, int],
    start_date: str = "2014-01-01",
    cluster_weeks: int = 8,
) -> tuple[dict, list[dict]]:
    """Collapse clustered signals into distinct entry events and evaluate next-day Open performance."""
    w_dates = w.index
    d_dates = daily_df.index
    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    lows = daily_df["Low"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)
    n_d = len(daily_df)

    valid_mask = (w_dates >= start_date) & cond.fillna(False).to_numpy(dtype=bool)
    sig_positions = np.where(valid_mask)[0]

    # Collapse into distinct entry clusters (first trigger, then skip next cluster_weeks)
    events = []
    last_ev_pos = -1000
    for pos in sig_positions:
        if (pos - last_ev_pos) <= cluster_weeks:
            continue
        last_ev_pos = pos
        w_dt = w_dates[pos]
        d_pos = int(d_dates.get_loc(w_dt))
        if d_pos + 1 >= n_d:
            continue
        entry_idx = d_pos + 1
        entry_dt = d_dates[entry_idx]
        entry_px = opens[entry_idx]

        # MAE over next 126 trading days (~6 months)
        w6_end = min(n_d, entry_idx + 126)
        mae_6m = (np.min(lows[entry_idx:w6_end]) / entry_px - 1.0) * 100.0

        def fwd_ret(bars: int) -> float:
            if entry_idx + bars >= n_d:
                return np.nan
            return (closes[entry_idx + bars] / entry_px - 1.0) * 100.0

        # Distance in weeks to nearest major bottom
        nearest_lbl, nearest_dist = None, 999
        for b_lbl, t_pos in trough_indices.items():
            diff = pos - t_pos
            if abs(diff) < abs(nearest_dist):
                nearest_dist = diff
                nearest_lbl = b_lbl

        events.append({
            "w_pos": int(pos),
            "sig_week": w_dt.strftime("%Y-%m-%d"),
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "entry_px": round(float(entry_px), 2),
            "MAE_6M%": round(float(mae_6m), 1),
            "Fwd_3M%": round(fwd_ret(63), 1) if not np.isnan(fwd_ret(63)) else np.nan,
            "Fwd_6M%": round(fwd_ret(126), 1) if not np.isnan(fwd_ret(126)) else np.nan,
            "Fwd_12M%": round(fwd_ret(252), 1) if not np.isnan(fwd_ret(252)) else np.nan,
            "NearestBottom": f"{nearest_lbl}({nearest_dist:+d}w)" if nearest_lbl else "—",
            "InBottomWin": abs(nearest_dist) <= 8 if -6 <= nearest_dist <= 8 else False,
        })

    # Coverage across the 5 major bottoms: hit if any event falls in [T0 - 6w, T0 + 8w]
    caught = []
    missed = []
    for b_lbl, t_pos in trough_indices.items():
        if any(-6 <= (ev["w_pos"] - t_pos) <= 8 for ev in events):
            caught.append(b_lbl)
        else:
            missed.append(b_lbl)

    false_alarms = sum(
        1 for ev in events
        if not any(-6 <= (ev["w_pos"] - t_pos) <= 8 for t_pos in trough_indices.values())
    )

    ev_df = pd.DataFrame(events)
    if ev_df.empty:
        summary = {
            "Rule": name, "Events": 0, "Caught": f"0/{len(trough_indices)}",
            "FalseAlarms": 0, "Med_MAE6M%": np.nan, "Worst_MAE6M%": np.nan,
            "Avg_6M%": np.nan, "Med_6M%": np.nan, "Win_6M%": np.nan,
            "Avg_12M%": np.nan, "Win_12M%": np.nan, "Missed": ",".join(missed),
        }
    else:
        f6 = ev_df["Fwd_6M%"].dropna()
        f12 = ev_df["Fwd_12M%"].dropna()
        summary = {
            "Rule": name,
            "Events": len(ev_df),
            "Caught": f"{len(caught)}/{len(trough_indices)}",
            "FalseAlarms": false_alarms,
            "Med_MAE6M%": round(float(ev_df["MAE_6M%"].median()), 1),
            "Worst_MAE6M%": round(float(ev_df["MAE_6M%"].min()), 1),
            "Avg_6M%": round(float(f6.mean()), 1) if len(f6) else np.nan,
            "Med_6M%": round(float(f6.median()), 1) if len(f6) else np.nan,
            "Win_6M%": round(float((f6 > 0).mean() * 100.0), 1) if len(f6) else np.nan,
            "Avg_12M%": round(float(f12.mean()), 1) if len(f12) else np.nan,
            "Win_12M%": round(float((f12 > 0).mean() * 100.0), 1) if len(f12) else np.nan,
            "Missed": ",".join(missed) if missed else "None",
        }
    return summary, events


def main() -> None:
    soxl = load_adjusted("SOXL")
    smh = load_adjusted("SMH")
    w, daily_soxl = build_weekly_dataset(soxl, smh)

    major_months = ["2016-02", "2018-12", "2020-03", "2022-10", "2025-04"]
    trough_indices: dict[str, int] = {}
    ym_series = w.index.strftime("%Y-%m")
    for bm in major_months:
        sub = w[ym_series == bm]
        t_dt = sub["Low"].idxmin()
        trough_indices[bm] = int(w.index.get_loc(t_dt))

    # =========================================================================
    # PART A: Week-by-Week Anatomy of the 5 Major Bottoms (T-2w .. T+3w)
    # =========================================================================
    print("========================================================================================")
    print("PART A: WEEK-BY-WEEK ANATOMY OF SOXL'S 5 MAJOR BOTTOMS (2016-2025) & CURRENT 2026 EDGE")
    print("========================================================================================")
    cols_anat = [
        "Low", "Close", "Low_vs_200W%", "Close_vs_200W%", "W_Fisher", "W_Fish_Log",
        "dW_Fish", "M_Fish_prev", "W_Vol_R", "Low_DD_52W%", "SMH_vs_200W%"
    ]

    t0_rows = []
    t1_rows = []
    for bm, t_pos in trough_indices.items():
        print(f"\n--- Major Bottom {bm} (Trough Week T0 = {w.index[t_pos].strftime('%Y-%m-%d')}) ---")
        sl_w = w.iloc[max(0, t_pos - 2) : min(len(w), t_pos + 4)][cols_anat].copy()
        sl_w.insert(0, "Offset", [f"T{k:+d}w" if k != 0 else "T0" for k in range(max(0, t_pos - 2) - t_pos, min(len(w), t_pos + 4) - t_pos)])
        sl_w.insert(1, "WeekEnd", sl_w.index.strftime("%Y-%m-%d"))
        print(sl_w.round(2).to_string(index=False))

        r0 = w.iloc[t_pos][cols_anat].to_dict()
        r0["Bottom"] = bm; r0["Week"] = w.index[t_pos].strftime("%Y-%m-%d")
        t0_rows.append(r0)

        r1 = w.iloc[t_pos + 1][cols_anat].to_dict()
        r1["Bottom"] = bm; r1["Week"] = w.index[t_pos + 1].strftime("%Y-%m-%d")
        t1_rows.append(r1)

    print("\n--- T0 (Exact Trough Week) Summary Across All 5 Major Bottoms ---")
    print(pd.DataFrame(t0_rows)[["Bottom", "Week"] + cols_anat].round(2).to_string(index=False))
    print("\n--- T+1w (First Recovery Week) Summary Across All 5 Major Bottoms ---")
    print(pd.DataFrame(t1_rows)[["Bottom", "Week"] + cols_anat].round(2).to_string(index=False))

    print("\n--- Latest 4 Weeks (2026-10 ~ 2026-11 Context) ---")
    latest_w = w.iloc[-4:][cols_anat].copy()
    latest_w.insert(0, "WeekEnd", latest_w.index.strftime("%Y-%m-%d"))
    print(latest_w.round(2).to_string(index=False))

    # =========================================================================
    # PART B: Systematic Evaluation of Causal Bottom Entry Rules (2014-2026)
    # =========================================================================
    print("\n========================================================================================")
    print("PART B: CAUSAL MAJOR BOTTOM ENTRY RULES EVALUATION (SOXL 2014-01 ~ 2026-11)")
    print("========================================================================================")

    def build_rule_dict(wf_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
        # Helper flags
        min_wf_4w = wf_df["W_Fisher"].rolling(4, min_periods=1).min()
        min_low_8w = wf_df["Low"].rolling(8, min_periods=1).min()
        min_low_200_8w = (wf_df["Low"] / wf_df["W_SMA200"]).rolling(8, min_periods=1).min()

        return [
            # Family A: 200W SMA position alone
            ("A1: Low <= 200W_SMA", wf_df["Low"] <= wf_df["W_SMA200"]),
            ("A2: Close <= 200W_SMA", wf_df["Close"] <= wf_df["W_SMA200"]),
            ("A3: Close <= 1.10*200W_SMA", wf_df["Close"] <= wf_df["W_SMA200"] * 1.10),
            ("A4: Close <= 0.90*200W_SMA", wf_df["Close"] <= wf_df["W_SMA200"] * 0.90),

            # Family B: 200W SMA + Weekly Fisher Extreme + Turn-up
            ("B1: Low<=1.05*200W & W_Fish<-1.5", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.05) & (wf_df["W_Fisher"] < -1.5)),
            ("B2: Low<=1.05*200W & W_Fish<-1.5 & dW>0", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.05) & (wf_df["W_Fisher"] < -1.5) & (wf_df["dW_Fish"] > 0)),
            ("B3: Close<=1.10*200W & W_Fish<-1.5 & dW>0", (wf_df["Close"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fisher"] < -1.5) & (wf_df["dW_Fish"] > 0)),
            ("B4: Close<=1.10*200W & W_Fish<-1.5 & W_Vol>1.3", (wf_df["Close"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fisher"] < -1.5) & (wf_df["W_Vol_R"] > 1.3)),
            ("B5: Low<=1.10*200W & W_Fish<-1.5 & (dW>0|Close>Open)", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fisher"] < -1.5) & ((wf_df["dW_Fish"] > 0) | (wf_df["Close"] > wf_df["Open"]))),
            ("B6: Low<=1.05*200W & FishCross & min4w<-1.5", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.05) & (wf_df["W_Fisher"] > wf_df["W_Fisher_Sig"]) & (min_wf_4w < -1.5)),
            ("B7: Low<=1.10*200W & FishCross & min4w<-1.5", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fisher"] > wf_df["W_Fisher_Sig"]) & (min_wf_4w < -1.5)),
            ("B8: Low<=1.10*200W & W_Fish<-1.8 & dW>0", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fisher"] < -1.8) & (wf_df["dW_Fish"] > 0)),

            # Log-price Fisher variants
            ("B9_Log: Low<=1.10*200W & FishLog<-1.5 & dWLog>0", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fish_Log"] < -1.5) & (wf_df["dW_Fish_Log"] > 0)),
            ("B10_Log: Low<=1.10*200W & FishLogCross & min4w<-1.5", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["W_Fish_Log"] > wf_df["W_Fish_Log_Sig"]) & (wf_df["W_Fish_Log"].rolling(4).min() < -1.5)),

            # Family C: Reclaim / Deep Drawdown / SMH Confirmation
            ("C1: Low8w<=1.10*200W & FishCross & min4w<-1.5", (min_low_200_8w <= 1.10) & (wf_df["W_Fisher"] > wf_df["W_Fisher_Sig"]) & (min_wf_4w < -1.5)),
            ("C2: Low8w<=1.00*200W & FishCross & min4w<-1.5", (min_low_200_8w <= 1.00) & (wf_df["W_Fisher"] > wf_df["W_Fisher_Sig"]) & (min_wf_4w < -1.5)),
            ("C3: Low<=1.10*200W & LowDD52W<=-60 & W_Fish<-1.5 & dW>0", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["Low_DD_52W%"] <= -60.0) & (wf_df["W_Fisher"] < -1.5) & (wf_df["dW_Fish"] > 0)),
            ("C4: Low8w<=1.10*200W & LowDD52W<=-60 & FishCross & min4w<-1.5", (min_low_200_8w <= 1.10) & (wf_df["Low_DD_52W%"] <= -60.0) & (wf_df["W_Fisher"] > wf_df["W_Fisher_Sig"]) & (min_wf_4w < -1.5)),
            ("C5: Low<=1.10*200W & SMH_W_Fish<-1.5 & dW>0", (wf_df["Low"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["SMH_W_Fisher"] < -1.5) & (wf_df["dW_Fish"] > 0)),
            ("C6: Close<=1.10*200W & M_Fish_prev<0 & W_Fish<-1.5", (wf_df["Close"] <= wf_df["W_SMA200"] * 1.10) & (wf_df["M_Fish_prev"] < 0) & (wf_df["W_Fisher"] < -1.5)),
            ("C7: Low8w<=1.10*200W & Close>W_EMA10 & min4w<-1.5", (min_low_200_8w <= 1.10) & (wf_df["Close"] > wf_df["W_EMA10"]) & (min_wf_4w < -1.5)),
        ]

    rule_list = build_rule_dict(w)
    summaries = []
    events_map = {}
    for rname, cond in rule_list:
        s, evs = evaluate_bottom_rule(rname, cond, w, daily_soxl, trough_indices)
        summaries.append(s)
        events_map[rname] = evs

    sdf = pd.DataFrame(summaries)
    print(sdf.to_string(index=False))

    # =========================================================================
    # PART C: Event Logs for Top Cleanest Rules & TQQQ Cross-Check
    # =========================================================================
    print("\n========================================================================================")
    print("PART C: DETAILED ENTRY LOGS & TQQQ CROSS-CHECK FOR TOP CLEANEST BOTTOM RULES")
    print("========================================================================================")

    # Pick top 4 rules with >=4/5 or 5/5 caught and lowest false alarms / best Med_6M & MAE_6M
    sdf["_caught_n"] = sdf["Caught"].str.split("/").str[0].astype(int)
    top_rules = (
        sdf[sdf["_caught_n"] >= 4]
        .sort_values(["_caught_n", "FalseAlarms", "Med_MAE6M%", "Med_6M%"], ascending=[False, True, False, False])
        .head(4)["Rule"]
        .tolist()
    )
    if not top_rules:
        top_rules = sdf.sort_values(["_caught_n", "FalseAlarms"], ascending=[False, True]).head(4)["Rule"].tolist()

    for rname in top_rules:
        print(f"\n--- SOXL Event Log: {rname} ---")
        print(pd.DataFrame(events_map[rname]).drop(columns=["w_pos"]).to_string(index=False))

    # Cross-check on TQQQ/QQQ (2014-2026)
    tqqq = load_adjusted("TQQQ")
    qqq = load_adjusted("QQQ")
    wt, daily_tqqq = build_weekly_dataset(tqqq, qqq)
    tqqq_bottoms = ["2016-02", "2018-12", "2020-03", "2022-12", "2025-04"]
    t_troughs = {}
    ym_t = wt.index.strftime("%Y-%m")
    for bm in tqqq_bottoms:
        sub = wt[ym_t == bm]
        if not sub.empty:
            t_troughs[bm] = int(wt.index.get_loc(sub["Low"].idxmin()))

    t_rule_map = dict(build_rule_dict(wt))
    t_rows = []
    for rname in top_rules:
        ts, tevs = evaluate_bottom_rule(rname, t_rule_map[rname], wt, daily_tqqq, t_troughs)
        t_rows.append(ts)
        print(f"\n--- TQQQ Cross-Check Event Log: {rname} ---")
        if tevs:
            print(pd.DataFrame(tevs).drop(columns=["w_pos"]).to_string(index=False))
    print("\n--- TQQQ Cross-Check Summary (2014-2026) ---")
    print(pd.DataFrame(t_rows).to_string(index=False))


if __name__ == "__main__":
    main()
