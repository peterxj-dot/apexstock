from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, fisher_transform
from scan_soxl_strategies import load_adjusted
from study_major_bottoms import build_weekly_dataset, evaluate_bottom_rule
from verify_bottom_volume import add_volume_features
from test_time_since_top import get_troughs


def prepare_asset_data(ticker: str, bench_ticker: str, trough_months: list[str]):
    daily_df = load_adjusted(ticker)
    bench_df = load_adjusted(bench_ticker)
    w, daily_df = build_weekly_dataset(daily_df, bench_df)
    w = add_volume_features(w, daily_df, bench_df)
    troughs = get_troughs(w, trough_months)

    min_low_200_8w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()

    w["min_low_200_8w"] = min_low_200_8w
    w["min_wf_4w"] = min_wf_4w
    w["pre_c7_vol"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (w["max4w_Vol_R"] >= 1.6)
    w["pre_c7_vol_15"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (w["max4w_Vol_R"] >= 1.5)

    for p in (3, 4, 5, 6, 8, 10):
        col = f"W_EMA{p}"
        if col not in w.columns:
            w[col] = ema(w["Close"], p)
    w["prev_W_High"] = w["High"].shift(1)
    w["prev_W_Close"] = w["Close"].shift(1)

    for p in (5, 10, 15, 20, 25, 30):
        daily_df[f"D_EMA{p}"] = ema(daily_df["Close"], p)
    d_fish = fisher_transform(daily_df["High"], daily_df["Low"], period=10)
    daily_df["D_Fisher"] = d_fish["fisher"]
    daily_df["D_Fisher_Sig"] = d_fish["fisher_signal"]

    # Strictly causal weekly->daily alignment: completed week last_trade_date -> ffill
    daily_df["pre_c7_vol"] = (
        w["pre_c7_vol"].astype("boolean").reindex(daily_df.index, method="ffill").fillna(False).to_numpy(dtype=bool)
    )
    daily_df["pre_c7_vol_15"] = (
        w["pre_c7_vol_15"].astype("boolean").reindex(daily_df.index, method="ffill").fillna(False).to_numpy(dtype=bool)
    )

    verify_causal_weekly_alignment(w["pre_c7_vol"], daily_df["pre_c7_vol"])
    return w, daily_df, troughs


def verify_causal_weekly_alignment(w_pre: pd.Series, d_pre: pd.Series) -> None:
    """Assert every daily row only sees completed weeks with last_trade_date <= d."""
    week_dates = w_pre.index
    week_vals = w_pre.to_numpy(dtype=bool)
    week_set = set(week_dates)
    for dt, val in d_pre.items():
        if dt in week_set:
            expected = bool(w_pre.loc[dt])
        else:
            pos = int(week_dates.searchsorted(dt, side="right")) - 1
            expected = bool(week_vals[pos]) if pos >= 0 else False
        if bool(val) != expected:
            raise AssertionError(f"Look-ahead / alignment mismatch at {dt}: got {val}, expected {expected}")


def evaluate_daily_bottom_rule(
    name: str,
    cond: pd.Series,
    w: pd.DataFrame,
    daily_df: pd.DataFrame,
    trough_indices: dict[str, int],
    start_date: str = "2014-01-01",
    cluster_days: int = 40,
) -> tuple[dict, list[dict]]:
    start_ts = pd.Timestamp(start_date)
    cond_b = cond.fillna(False).astype(bool)
    events: list[dict] = []
    last_sig_pos: int | None = None

    idx = daily_df.index
    n = len(daily_df)
    w_idx = w.index
    t_items = list(trough_indices.items())

    for i in range(n - 1):
        sig_date = idx[i]
        if sig_date < start_ts:
            continue
        if not bool(cond_b.iloc[i]):
            continue
        if last_sig_pos is not None and (i - last_sig_pos) <= cluster_days:
            continue
        last_sig_pos = i

        entry_pos = i + 1
        entry_date = idx[entry_pos]
        entry_px = float(daily_df["Open"].iloc[entry_pos])

        w6 = daily_df.iloc[entry_pos : min(n, entry_pos + 126)]
        mae6 = (float(w6["Low"].min()) / entry_px - 1.0) * 100.0 if len(w6) > 0 else np.nan
        fwd3 = (float(daily_df["Close"].iloc[entry_pos + 63]) / entry_px - 1.0) * 100.0 if entry_pos + 63 < n else np.nan
        fwd6 = (float(daily_df["Close"].iloc[entry_pos + 126]) / entry_px - 1.0) * 100.0 if entry_pos + 126 < n else np.nan
        fwd12 = (float(daily_df["Close"].iloc[entry_pos + 252]) / entry_px - 1.0) * 100.0 if entry_pos + 252 < n else np.nan

        w_pos = int(w_idx.searchsorted(sig_date, side="left"))
        if w_pos >= len(w_idx):
            w_pos = len(w_idx) - 1

        nearest_m = None
        min_abs_diff = 999
        in_bottom_win = False
        for m_name, t_pos in t_items:
            diff = w_pos - t_pos
            if -2 <= diff <= 8:
                in_bottom_win = True
            if abs(diff) < min_abs_diff:
                min_abs_diff = abs(diff)
                nearest_m = m_name

        events.append({
            "w_pos": w_pos,
            "sig_week": sig_date.strftime("%Y-%m-%d"),
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "entry_px": entry_px,
            "MAE_6M%": mae6,
            "Fwd_3M%": fwd3,
            "Fwd_6M%": fwd6,
            "Fwd_12M%": fwd12,
            "NearestBottom": f"{nearest_m} ({'+' if w_pos - trough_indices[nearest_m] >= 0 else ''}{w_pos - trough_indices[nearest_m]}w)" if nearest_m else "",
            "InBottomWin": in_bottom_win,
        })

    caught_months = set()
    for ev in events:
        for m_name, t_pos in t_items:
            if -2 <= (ev["w_pos"] - t_pos) <= 8:
                caught_months.add(m_name)

    false_alarms = sum(1 for ev in events if not ev["InBottomWin"])
    maes = [ev["MAE_6M%"] for ev in events if not np.isnan(ev["MAE_6M%"])]
    f6s = [ev["Fwd_6M%"] for ev in events if not np.isnan(ev["Fwd_6M%"])]
    f12s = [ev["Fwd_12M%"] for ev in events if not np.isnan(ev["Fwd_12M%"])]

    summary = {
        "Rule": name,
        "Events": len(events),
        "Caught": f"{len(caught_months)}/{len(trough_indices)}",
        "FalseAlarms": false_alarms,
        "Med_MAE6M%": round(float(np.median(maes)), 1) if maes else np.nan,
        "Worst_MAE6M%": round(float(np.min(maes)), 1) if maes else np.nan,
        "Avg_6M%": round(float(np.mean(f6s)), 1) if f6s else np.nan,
        "Med_6M%": round(float(np.median(f6s)), 1) if f6s else np.nan,
        "Win_6M%": round(100.0 * sum(1 for x in f6s if x > 0) / len(f6s), 1) if f6s else np.nan,
        "Avg_12M%": round(float(np.mean(f12s)), 1) if f12s else np.nan,
        "Win_12M%": round(100.0 * sum(1 for x in f12s if x > 0) / len(f12s), 1) if f12s else np.nan,
        "Missed": ",".join(m for m in trough_indices if m not in caught_months) or "-",
    }
    return summary, events


def match_bottom_events(events: list[dict], w: pd.DataFrame, trough_indices: dict[str, int]) -> dict[str, dict]:
    """Return earliest event matching each bottom window [-2w, +8w]."""
    matched: dict[str, dict] = {}
    for m_name, t_pos in trough_indices.items():
        t_date = w.index[t_pos]
        candidates = [ev for ev in events if -2 <= (ev["w_pos"] - t_pos) <= 8]
        if candidates:
            ev = candidates[0]
            sig_ts = pd.Timestamp(ev["sig_week"])
            ev_copy = dict(ev)
            ev_copy["offset_w"] = (sig_ts - t_date).days / 7.0
            matched[m_name] = ev_copy
    return matched


def augment_timing_metrics(
    summary: dict,
    events: list[dict],
    base_matched: dict[str, dict],
    w: pd.DataFrame,
    trough_indices: dict[str, int],
) -> tuple[dict, dict[str, dict]]:
    matched = match_bottom_events(events, w, trough_indices)
    offsets = [ev["offset_w"] for ev in matched.values()]
    px_chgs = [
        (ev["entry_px"] / base_matched[m]["entry_px"] - 1.0) * 100.0
        for m, ev in matched.items()
        if m in base_matched
    ]
    out = dict(summary)
    out["Avg_Offset_w"] = round(float(np.mean(offsets)), 2) if offsets else np.nan
    out["Avg_Px_Chg_vs_C7Vol%"] = round(float(np.mean(px_chgs)), 1) if px_chgs else np.nan
    return out, matched


def run_suite(ticker: str, bench_ticker: str, trough_months: list[str]):
    w, daily_df, troughs = prepare_asset_data(ticker, bench_ticker, trough_months)

    weekly_rules: list[tuple[str, pd.Series]] = [
        ("W1_C7_Vol (EMA10)", w["pre_c7_vol"] & (w["Close"] > w["W_EMA10"])),
        ("W1_EMA8", w["pre_c7_vol"] & (w["Close"] > w["W_EMA8"])),
        ("W1_EMA6", w["pre_c7_vol"] & (w["Close"] > w["W_EMA6"])),
        ("W1_EMA5", w["pre_c7_vol"] & (w["Close"] > w["W_EMA5"])),
        ("W1_EMA4", w["pre_c7_vol"] & (w["Close"] > w["W_EMA4"])),
        ("W1_EMA3", w["pre_c7_vol"] & (w["Close"] > w["W_EMA3"])),
        ("W1_PrevHigh", w["pre_c7_vol"] & (w["Close"] > w["prev_W_High"])),
        ("W1_Green_PrevClose", w["pre_c7_vol"] & (w["Close"] > w["Open"]) & (w["Close"] > w["prev_W_Close"])),
        ("W1_FishLog_Turn", w["pre_c7_vol"] & (w["dW_Fish_Log"] > 0)),
        ("W1_FishLog_Turn_Green", w["pre_c7_vol"] & (w["dW_Fish_Log"] > 0) & (w["Close"] > w["Open"])),
        ("W1_FishLog_Turn_PrevClose", w["pre_c7_vol"] & (w["dW_Fish_Log"] > 0) & (w["Close"] > w["prev_W_Close"])),
        (
            "W1_EMA5_or_FishLogPrevHigh",
            w["pre_c7_vol"] & ((w["Close"] > w["W_EMA5"]) | ((w["dW_Fish_Log"] > 0) & (w["Close"] > w["prev_W_High"]))),
        ),
        # Vol 1.5x sensitivity checks
        ("W1_EMA10_Vol15", w["pre_c7_vol_15"] & (w["Close"] > w["W_EMA10"])),
        ("W1_EMA8_Vol15", w["pre_c7_vol_15"] & (w["Close"] > w["W_EMA8"])),
        ("W1_EMA6_Vol15", w["pre_c7_vol_15"] & (w["Close"] > w["W_EMA6"])),
        ("W1_EMA5_Vol15", w["pre_c7_vol_15"] & (w["Close"] > w["W_EMA5"])),
        (
            "W1_FishLog_Turn_PrevClose_Vol15",
            w["pre_c7_vol_15"] & (w["dW_Fish_Log"] > 0) & (w["Close"] > w["prev_W_Close"]),
        ),
        (
            "W1_EMA5_or_FishLogPrevHigh_Vol15",
            w["pre_c7_vol_15"] & ((w["Close"] > w["W_EMA5"]) | ((w["dW_Fish_Log"] > 0) & (w["Close"] > w["prev_W_High"]))),
        ),
    ]

    daily_rules: list[tuple[str, pd.Series]] = [
        ("D1_EMA10", daily_df["pre_c7_vol"] & (daily_df["Close"] > daily_df["D_EMA10"])),
        ("D1_EMA15", daily_df["pre_c7_vol"] & (daily_df["Close"] > daily_df["D_EMA15"])),
        ("D1_EMA20", daily_df["pre_c7_vol"] & (daily_df["Close"] > daily_df["D_EMA20"])),
        ("D1_EMA25", daily_df["pre_c7_vol"] & (daily_df["Close"] > daily_df["D_EMA25"])),
        ("D1_EMA30", daily_df["pre_c7_vol"] & (daily_df["Close"] > daily_df["D_EMA30"])),
        (
            "D1_EMA10_Fish",
            daily_df["pre_c7_vol"]
            & (daily_df["Close"] > daily_df["D_EMA10"])
            & (daily_df["D_Fisher"] > daily_df["D_Fisher_Sig"]),
        ),
        (
            "D1_EMA15_Fish",
            daily_df["pre_c7_vol"]
            & (daily_df["Close"] > daily_df["D_EMA15"])
            & (daily_df["D_Fisher"] > daily_df["D_Fisher_Sig"]),
        ),
        (
            "D1_EMA20_Fish",
            daily_df["pre_c7_vol"]
            & (daily_df["Close"] > daily_df["D_EMA20"])
            & (daily_df["D_Fisher"] > daily_df["D_Fisher_Sig"]),
        ),
        ("D1_EMA5x15", daily_df["pre_c7_vol"] & (daily_df["D_EMA5"] > daily_df["D_EMA15"])),
        ("D1_EMA5x20", daily_df["pre_c7_vol"] & (daily_df["D_EMA5"] > daily_df["D_EMA20"])),
    ]

    # Sanity parity check: weekly C7_Vol mapped to exact weekly last_trade_date rows in daily evaluator
    c7_w_cond = weekly_rules[0][1]
    base_summary, base_events = evaluate_bottom_rule("W1_C7_Vol (EMA10)", c7_w_cond, w, daily_df, troughs)
    base_matched = match_bottom_events(base_events, w, troughs)

    c7_daily_sig = pd.Series(False, index=daily_df.index)
    for dt in w.index[c7_w_cond.fillna(False)]:
        if dt in c7_daily_sig.index:
            c7_daily_sig.loc[dt] = True
    parity_summary, parity_events = evaluate_daily_bottom_rule("Parity_C7_Vol", c7_daily_sig, w, daily_df, troughs)

    rows = []
    all_events: dict[str, list[dict]] = {}
    all_matched: dict[str, dict[str, dict]] = {}

    for name, cond in weekly_rules:
        s, evs = evaluate_bottom_rule(name, cond, w, daily_df, troughs)
        s_aug, matched = augment_timing_metrics(s, evs, base_matched, w, troughs)
        rows.append(s_aug)
        all_events[name] = evs
        all_matched[name] = matched

    for name, cond in daily_rules:
        s, evs = evaluate_daily_bottom_rule(name, cond, w, daily_df, troughs)
        s_aug, matched = augment_timing_metrics(s, evs, base_matched, w, troughs)
        rows.append(s_aug)
        all_events[name] = evs
        all_matched[name] = matched

    res_df = pd.DataFrame(rows)
    return w, daily_df, troughs, res_df, all_events, all_matched, base_summary, parity_summary


def print_bottom_comparison(
    ticker: str,
    w: pd.DataFrame,
    troughs: dict[str, int],
    rule_names: list[str],
    all_matched: dict[str, dict[str, dict]],
    all_events: dict[str, list[dict]],
):
    print(f"\n=== {ticker}: 5大底部 入场日期 & 价格逐底对比 ===")
    headers = ["Bottom", "T0_Week(Low/Close)"] + rule_names
    lines = []
    for m, t_pos in troughs.items():
        t_row = w.iloc[t_pos]
        t_str = f"{w.index[t_pos].strftime('%Y-%m-%d')} (${t_row['Low']:.2f}/${t_row['Close']:.2f})"
        row_cells = [m, t_str]
        base_px = all_matched["W1_C7_Vol (EMA10)"][m]["entry_px"] if m in all_matched["W1_C7_Vol (EMA10)"] else None
        for rname in rule_names:
            ev = all_matched[rname].get(m)
            if ev is None:
                row_cells.append("MISSED")
            else:
                off = f"+{ev['offset_w']:.1f}w" if ev["offset_w"] >= 0 else f"{ev['offset_w']:.1f}w"
                if rname == "W1_C7_Vol (EMA10)" or base_px is None:
                    row_cells.append(f"{ev['entry_date']} ${ev['entry_px']:.2f} ({off}, MAE {ev['MAE_6M%']:.1f}%)")
                else:
                    chg = (ev["entry_px"] / base_px - 1.0) * 100.0
                    row_cells.append(
                        f"{ev['entry_date']} ${ev['entry_px']:.2f} ({off}, {chg:+.1f}%, MAE {ev['MAE_6M%']:.1f}%)"
                    )
        lines.append(row_cells)

    df_cmp = pd.DataFrame(lines, columns=headers)
    print(df_cmp.to_string(index=False))

    print(f"\n--- {ticker} 误报明细 (False Alarms) ---")
    for rname in rule_names:
        fas = [ev for ev in all_events[rname] if not ev["InBottomWin"]]
        if not fas:
            print(f"  {rname}: 0 False Alarms")
        else:
            fa_str = "; ".join(
                f"{ev['entry_date']} @ ${ev['entry_px']:.2f} (MAE {ev['MAE_6M%']:.1f}%, 6M {ev['Fwd_6M%']:+.1f}%, {ev['NearestBottom']})"
                for ev in fas
            )
            print(f"  {rname} ({len(fas)} FA): {fa_str}")


def pick_top_candidates(soxl_df: pd.DataFrame, tqqq_df: pd.DataFrame) -> list[str]:
    merged = soxl_df[
        [
            "Rule",
            "Caught",
            "FalseAlarms",
            "Med_MAE6M%",
            "Worst_MAE6M%",
            "Avg_6M%",
            "Avg_12M%",
            "Avg_Offset_w",
            "Avg_Px_Chg_vs_C7Vol%",
        ]
    ].merge(
        tqqq_df[["Rule", "Caught", "FalseAlarms", "Avg_Offset_w", "Avg_Px_Chg_vs_C7Vol%", "Med_MAE6M%"]],
        on="Rule",
        suffixes=("_SOXL", "_TQQQ"),
    )
    cand = merged[
        (merged["Rule"] != "W1_C7_Vol (EMA10)")
        & (merged["Caught_SOXL"] == "5/5")
        & (merged["Avg_Px_Chg_vs_C7Vol%_SOXL"] < -1.0)
        & (merged["FalseAlarms_SOXL"] <= 2)
    ].copy()
    cand = cand.sort_values(
        ["FalseAlarms_SOXL", "FalseAlarms_TQQQ", "Avg_Px_Chg_vs_C7Vol%_SOXL", "Med_MAE6M%_SOXL"],
        ascending=[True, True, True, False],
    )
    return cand["Rule"].head(5).tolist()


def print_window_diagnostics(w: pd.DataFrame, troughs: dict[str, int], pre_col: str = "pre_c7_vol") -> None:
    cols = ["Close", "Low_vs_200W%", "Close_vs_200W%", "Close_vs_30W%", "DD_52W%", "W_Fisher", "W_Fish_Log", "W_Vol_R", "max4w_Vol_R", pre_col]
    print("\n--- Trough +8w Window Snapshots (checking what triggers/misses) ---")
    for m, t_pos in troughs.items():
        sl = w.iloc[max(0, t_pos - 1) : min(len(w), t_pos + 8)][cols]
        print(f"\n[{m}] T0={w.index[t_pos].strftime('%Y-%m-%d')}")
        print(sl.round(2).to_string())


def main():
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)

    soxl_months = ["2016-02", "2018-12", "2020-03", "2022-10", "2025-04"]
    tqqq_months = ["2016-02", "2018-12", "2020-03", "2022-12", "2025-04"]

    w_s, d_s, tr_s, res_s, ev_s, m_s, base_s, par_s = run_suite("SOXL", "SMH", soxl_months)
    w_t, d_t, tr_t, res_t, ev_t, m_t, base_t, par_t = run_suite("TQQQ", "QQQ", tqqq_months)

    print("=== PARITY & CAUSALITY CHECK ===")
    print("SOXL Weekly C7_Vol :", base_s)
    print("SOXL Daily Parity  :", par_s)
    print("TQQQ Weekly C7_Vol :", base_t)
    print("TQQQ Daily Parity  :", par_t)
    print("Daily weekly-precondition causality assertions PASSED on both SOXL and TQQQ.")

    cols = [
        "Rule", "Events", "Caught", "FalseAlarms",
        "Med_MAE6M%", "Worst_MAE6M%", "Avg_6M%", "Win_6M%", "Avg_12M%",
        "Avg_Offset_w", "Avg_Px_Chg_vs_C7Vol%", "Missed",
    ]
    print("\n=== SOXL (2014-2026) TIMING IMPROVEMENT SCAN ===")
    print(res_s[cols].to_string(index=False))

    print("\n=== TQQQ (2014-2026) CROSS-ASSET VALIDATION (UNMODIFIED PARAMS) ===")
    print(res_t[cols].to_string(index=False))

    top_rules = pick_top_candidates(res_s, res_t)
    # Ensure key archetypes are visible side-by-side
    for must in ["W1_EMA5", "W1_EMA5_or_FishLogPrevHigh", "W1_PrevHigh", "D1_EMA30", "D1_EMA5x20"]:
        if must not in top_rules:
            top_rules.append(must)
    show_rules = ["W1_C7_Vol (EMA10)"] + top_rules
    print(f"\nSelected Faster Candidates vs Baseline: {show_rules}")
    print_bottom_comparison("SOXL", w_s, tr_s, show_rules, m_s, ev_s)
    print_bottom_comparison("TQQQ", w_t, tr_t, show_rules, m_t, ev_t)

    # Print SOXL 2022-05 FA week context & TQQQ 2022-12 missed bottom context
    print("\n=== DIAGNOSTICS: SOXL 2022-04~06 Bear Rally vs Real Bottoms ===")
    fa_slice = w_s.loc["2022-04-01":"2022-06-15", ["Close", "W_EMA5", "W_EMA8", "W_EMA10", "Low_vs_200W%", "DD_52W%", "W_Fisher", "W_Fish_Log", "max4w_Vol_R", "pre_c7_vol"]]
    print(fa_slice.round(2).to_string())
    print("\n=== DIAGNOSTICS: TQQQ 2022-10~2023-02 Bottom ===")
    t_slice = w_t.loc["2022-10-01":"2023-02-15", ["Close", "W_EMA5", "W_EMA8", "W_EMA10", "Low_vs_200W%", "min_low_200_8w", "min_wf_4w", "max4w_Vol_R", "pre_c7_vol"]]
    print(t_slice.round(2).to_string())


if __name__ == "__main__":
    main()
