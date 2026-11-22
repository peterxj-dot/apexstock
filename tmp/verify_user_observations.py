"""Verify user's 3 chart observations and test dist30w/dist200w/time-depth extensions to Strategy 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, fisher_transform, historical_volatility, sma
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL, load_adjusted


def build_features(target_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    """Build strictly causal daily, weekly (ffill from last trade date), and monthly features."""
    full_idx = target_df.index
    bench_d = bench_df.reindex(full_idx, method="ffill")

    # Weekly causal aggregation
    with_dt = target_df.assign(_dt=full_idx)
    w_grp = (
        with_dt.groupby(pd.Grouper(freq="W-FRI"))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    w_dates = pd.DatetimeIndex(w_grp["_dt"])
    w = w_grp.drop(columns=["_dt"]).copy()
    w.index = w_dates

    wf = fisher_transform(w["High"], w["Low"], period=10)
    w_ind = pd.DataFrame(index=w.index)
    w_ind["W_Fisher"] = wf["fisher"]
    w_ind["dW_Fisher"] = w_ind["W_Fisher"].diff()
    w_ind["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    w_ind["W_SMA30"] = sma(w["Close"], 30)
    w_ind["W_SMA200"] = sma(w["Close"], 200)

    w_aligned = w_ind.reindex(full_idx, method="ffill")
    is_week_end = full_idx.isin(w_dates)

    # Daily indicators
    d_close = target_df["Close"]
    d_high = target_df["High"]
    d_hv20 = historical_volatility(d_close, period=20, annualization_factor=252)
    d_ema10 = ema(d_close, 10)
    b_sma200 = sma(bench_d["Close"], 200)

    # 252-day rolling high distance and days since high (strictly causal)
    n = len(target_df)
    highs_arr = d_high.to_numpy(dtype=np.float64)
    closes_arr = d_close.to_numpy(dtype=np.float64)
    days_since_high = np.zeros(n, dtype=np.int32)
    dd_from_high = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_w = max(0, i - 251)
        win = highs_arr[start_w : i + 1]
        rel_argmax = int(np.argmax(win))
        peak_idx = start_w + rel_argmax
        peak_val = highs_arr[peak_idx]
        days_since_high[i] = i - peak_idx
        dd_from_high[i] = (closes_arr[i] / peak_val - 1.0) * 100.0

    dist30w = (d_close / w_aligned["W_SMA30"] - 1.0) * 100.0
    dist200w = (d_close / w_aligned["W_SMA200"] - 1.0) * 100.0

    return {
        "index": full_idx,
        "opens": target_df["Open"].to_numpy(dtype=np.float64),
        "closes": closes_arr,
        "highs": highs_arr,
        "lows": target_df["Low"].to_numpy(dtype=np.float64),
        "d_ema10": d_ema10.to_numpy(dtype=np.float64),
        "hv20": d_hv20.to_numpy(dtype=np.float64),
        "is_week_end": is_week_end,
        "wf": w_aligned["W_Fisher"].to_numpy(dtype=np.float64),
        "dwf": w_aligned["dW_Fisher"].to_numpy(dtype=np.float64),
        "wvr": w_aligned["W_Vol_R"].to_numpy(dtype=np.float64),
        "w_sma30": w_aligned["W_SMA30"].to_numpy(dtype=np.float64),
        "w_sma200": w_aligned["W_SMA200"].to_numpy(dtype=np.float64),
        "dist30w": dist30w.to_numpy(dtype=np.float64),
        "dist200w": dist200w.to_numpy(dtype=np.float64),
        "days_since_high": days_since_high,
        "dd_from_high": dd_from_high,
        "b_close": bench_d["Close"].to_numpy(dtype=np.float64),
        "b_sma200": b_sma200.to_numpy(dtype=np.float64),
        "w_aligned_df": w_aligned,
    }


def detect_monthly_zigzag(m: pd.DataFrame, up_th: float = 1.20, down_th: float = 0.55) -> list[tuple[int, str, float]]:
    pivots = []
    trend = None
    lh_i, lh = 0, float(m["High"].iloc[0])
    ll_i, ll = 0, float(m["Low"].iloc[0])
    for i in range(1, len(m)):
        h, l = float(m["High"].iloc[i]), float(m["Low"].iloc[i])
        if trend is None:
            if h >= ll * (1.0 + up_th):
                pivots.append((ll_i, "L", ll))
                trend = "UP"
                lh_i, lh = i, h
            elif l <= lh * (1.0 - down_th):
                pivots.append((lh_i, "H", lh))
                trend = "DOWN"
                ll_i, ll = i, l
            else:
                if h > lh: lh_i, lh = i, h
                if l < ll: ll_i, ll = i, l
        elif trend == "UP":
            if h >= lh:
                lh_i, lh = i, h
            elif l <= lh * (1.0 - down_th):
                pivots.append((lh_i, "H", lh))
                trend = "DOWN"
                ll_i, ll = i, l
        elif trend == "DOWN":
            if l <= ll:
                ll_i, ll = i, l
            elif h >= ll * (1.0 + up_th):
                pivots.append((ll_i, "L", ll))
                trend = "UP"
                lh_i, lh = i, h
    return pivots


def simulate_s1_ext(
    f: dict,
    start_date: str,
    end_date: str,
    # H1a: Overextended trailing stop or EMA10 exit
    ext_d30_th: float | None = None,
    ext_d200_th: float | None = None,
    ext_tsl: float | None = None,
    ext_ema10_exit: bool = False,
    # H1b: No-chase cap on TREND entry
    cap_d30: float | None = None,
    cap_d200: float | None = None,
    # H2: ALT bottom depth/time/distance gate
    alt_d30_max: float | None = None,
    alt_d200_max: float | None = None,
    alt_dd_min: float | None = None,
    alt_days_min: int | None = None,
    # H3: Depth-conditioned ALT take profit
    deep_d30_th: float | None = None,
    deep_d200_th: float | None = None,
    deep_dd_th: float | None = None,
    deep_tp: float = 0.25,
    deep_max_days: int = 40,
    record_trades: bool = False,
) -> dict:
    full_idx = f["index"]
    mask = (full_idx >= start_date) & (full_idx <= end_date)
    bt_idx = full_idx[mask]

    opens = f["opens"][mask]
    closes = f["closes"][mask]
    d_ema10 = f["d_ema10"][mask]
    hv20 = f["hv20"][mask]
    is_w_end = f["is_week_end"][mask]
    wf = f["wf"][mask]
    dwf = f["dwf"][mask]
    wvr = f["wvr"][mask]
    d30 = f["dist30w"][mask]
    d200 = f["dist200w"][mask]
    days_hi = f["days_since_high"][mask]
    dd_hi = f["dd_from_high"][mask]
    bc = f["b_close"][mask]
    bs200 = f["b_sma200"][mask]

    tb_raw = bc > bs200 * 1.01
    tb_def = tb_raw & (hv20 <= 105.8)
    tb_vol = tb_raw & (hv20 <= 95.0)
    ma_break = bc < bs200 * 0.99
    vol_exit = hv20 > 115.0

    # H1b no-chase filter (NaN-safe)
    if cap_d30 is not None or cap_d200 is not None:
        ok30 = np.ones(len(closes), dtype=bool) if cap_d30 is None else (np.isnan(d30) | (d30 <= cap_d30))
        ok200 = np.ones(len(closes), dtype=bool) if cap_d200 is None else (np.isnan(d200) | (d200 <= cap_d200))
        chase_ok = ok30 & ok200
        tb_def = tb_def & chase_ok
        tb_vol = tb_vol & chase_ok

    # Base ALT buy + H2 gates (NaN-safe)
    alt_buy = is_w_end & (wf < -1.8) & ((dwf > 0) | (wvr > 1.3))
    if alt_d30_max is not None:
        alt_buy = alt_buy & (np.isnan(d30) | (d30 <= alt_d30_max))
    if alt_d200_max is not None:
        alt_buy = alt_buy & (np.isnan(d200) | (d200 <= alt_d200_max))
    if alt_dd_min is not None or alt_days_min is not None:
        cond_dd = np.zeros(len(closes), dtype=bool) if alt_dd_min is None else (dd_hi <= -alt_dd_min)
        cond_dy = np.zeros(len(closes), dtype=bool) if alt_days_min is None else (days_hi >= alt_days_min)
        alt_buy = alt_buy & (cond_dd | cond_dy)

    n = len(closes)
    cash = INITIAL_CAPITAL
    shares = 0.0
    mode = 0
    initial_mode = 0
    pending = 0
    pending_reason = ""
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_reason = "INIT"
    peak_close = 0.0
    ext_armed = False
    cur_alt_tp = 0.25
    cur_alt_max_days = 40
    pending_deep = False

    equity = np.empty(n, dtype=np.float64)
    trade_rets = []
    trade_logs = []

    for i in range(n):
        o = opens[i]
        c = closes[i]

        if pending in (1, 2) and mode == 0:
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            mode = pending
            initial_mode = pending
            entry_px = o
            entry_idx = i
            peak_close = c
            ext_armed = False
            if mode == 2:
                cur_alt_tp = deep_tp if pending_deep else 0.25
                cur_alt_max_days = deep_max_days if pending_deep else 40
        elif pending == -1 and mode in (1, 2):
            proceeds = shares * o * (1.0 - FEE_RATE)
            ret = (o * (1.0 - FEE_RATE)) / (entry_px * (1.0 + FEE_RATE)) - 1.0
            trade_rets.append(ret)
            if record_trades:
                trade_logs.append({
                    "mode": "TREND" if initial_mode == 1 else "ALT",
                    "entry_date": bt_idx[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": bt_idx[i].strftime("%Y-%m-%d"),
                    "ret_pct": round(ret * 100.0, 2),
                    "reason": pending_reason,
                    "entry_wf": round(float(wf[max(0, entry_idx - 1)]), 2),
                    "entry_d30": round(float(d30[max(0, entry_idx - 1)]), 1) if not np.isnan(d30[max(0, entry_idx - 1)]) else np.nan,
                    "entry_d200": round(float(d200[max(0, entry_idx - 1)]), 1) if not np.isnan(d200[max(0, entry_idx - 1)]) else np.nan,
                    "entry_dd_hi": round(float(dd_hi[max(0, entry_idx - 1)]), 1),
                    "entry_days_hi": int(days_hi[max(0, entry_idx - 1)]),
                })
            cash = proceeds
            shares = 0.0
            mode = 0
            exit_idx = i
            last_exit_reason = pending_reason
        pending = 0
        pending_reason = ""
        pending_deep = False

        if mode == 1:
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            is_ext = False
            if ext_d30_th is not None and not np.isnan(d30[i]) and d30[i] > ext_d30_th:
                is_ext = True
            if ext_d200_th is not None and not np.isnan(d200[i]) and d200[i] > ext_d200_th:
                is_ext = True
            if is_ext:
                ext_armed = True

            if ext_tsl is not None and ext_armed and c <= peak_close * (1.0 - ext_tsl):
                pending = -1
                pending_reason = "EXT_TSL"
            elif ext_ema10_exit and is_ext and c < d_ema10[i]:
                pending = -1
                pending_reason = "EXT_EMA10"
            elif vol_exit[i]:
                pending = -1
                pending_reason = "VOL_SPIKE"
            elif ma_break[i]:
                pending = -1
                pending_reason = "TREND_BREAK"
        elif mode == 2:
            equity[i] = shares * c
            if c <= entry_px * 0.80:
                pending = -1
                pending_reason = "ALT_FSL"
            elif c >= entry_px * (1.0 + cur_alt_tp):
                pending = -1
                pending_reason = "ALT_TP"
            elif (i - entry_idx) >= cur_alt_max_days:
                pending = -1
                pending_reason = "ALT_TP"
        else:
            equity[i] = cash
            elapsed = i - exit_idx
            tb_cur = tb_vol if last_exit_reason == "VOL_SPIKE" else tb_def
            allow_enter = elapsed >= 10 if last_exit_reason in ("ALT_FSL", "ALT_TP") else True
            if allow_enter:
                if tb_cur[i]:
                    pending = 1
                elif alt_buy[i]:
                    pending = 2
                    is_deep = False
                    if deep_d30_th is not None and not np.isnan(d30[i]) and d30[i] <= deep_d30_th:
                        is_deep = True
                    if deep_d200_th is not None and not np.isnan(d200[i]) and d200[i] <= deep_d200_th:
                        is_deep = True
                    if deep_dd_th is not None and dd_hi[i] <= -deep_dd_th:
                        is_deep = True
                    pending_deep = is_deep

    if mode != 0:
        ret = closes[-1] / (entry_px * (1.0 + FEE_RATE)) - 1.0
        trade_rets.append(ret)
        if record_trades:
            trade_logs.append({
                "mode": "TREND" if initial_mode == 1 else "ALT",
                "entry_date": bt_idx[entry_idx].strftime("%Y-%m-%d"),
                "exit_date": bt_idx[-1].strftime("%Y-%m-%d") + " (Open)",
                "ret_pct": round(ret * 100.0, 2),
                "reason": "OPEN",
                "entry_wf": round(float(wf[max(0, entry_idx - 1)]), 2),
                "entry_d30": round(float(d30[max(0, entry_idx - 1)]), 1) if not np.isnan(d30[max(0, entry_idx - 1)]) else np.nan,
                "entry_d200": round(float(d200[max(0, entry_idx - 1)]), 1) if not np.isnan(d200[max(0, entry_idx - 1)]) else np.nan,
                "entry_dd_hi": round(float(dd_hi[max(0, entry_idx - 1)]), 1),
                "entry_days_hi": int(days_hi[max(0, entry_idx - 1)]),
            })

    final_val = float(equity[-1])
    years_span = n / 252.0
    cagr = ((final_val / INITIAL_CAPITAL) ** (1.0 / years_span) - 1.0) * 100.0
    max_dd = float(((equity / np.maximum.accumulate(equity)) - 1.0).min() * 100.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0
    dr = np.diff(equity) / equity[:-1]
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252.0)) if dr.std() > 1e-12 else 0.0
    wins = sum(1 for r in trade_rets if r > 0)
    return {
        "Final": round(final_val, 2),
        "Ret%": round((final_val / INITIAL_CAPITAL - 1.0) * 100.0, 1),
        "CAGR%": round(cagr, 2),
        "MaxDD%": round(max_dd, 2),
        "Calmar": round(calmar, 3),
        "Sharpe": round(sharpe, 2),
        "Trades": len(trade_rets),
        "WinRate%": round(wins / len(trade_rets) * 100.0, 1) if trade_rets else 0.0,
        "logs": trade_logs,
    }


def part1_verify_observations(soxl_df: pd.DataFrame, f_soxl: dict) -> None:
    print("==========================================================================")
    print("PART 1: QUANTITATIVE VERIFICATION OF THE 3 CHART OBSERVATIONS (SOXL 2010-2026)")
    print("==========================================================================")

    # Build monthly bars & align strictly causal month-end features
    m = (
        soxl_df.assign(_dt=soxl_df.index)
        .groupby(pd.Grouper(freq="ME"))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    m["ym"] = m.index.strftime("%Y-%m")
    last_dts = pd.DatetimeIndex(m["_dt"])

    w_df = f_soxl["w_aligned_df"]
    m["W_SMA30"] = w_df["W_SMA30"].reindex(last_dts).to_numpy()
    m["W_SMA200"] = w_df["W_SMA200"].reindex(last_dts).to_numpy()
    m["dist30w"] = (m["Close"] / m["W_SMA30"] - 1.0) * 100.0
    m["dist200w"] = (m["Close"] / m["W_SMA200"] - 1.0) * 100.0
    idx_pos = f_soxl["index"].get_indexer(last_dts)
    m["days_since_hi"] = f_soxl["days_since_high"][idx_pos]
    m["dd_from_hi"] = f_soxl["dd_from_high"][idx_pos]

    pivots = detect_monthly_zigzag(m, up_th=1.20, down_th=0.55)

    # --- Observation 1: UP vs DOWN legs ---
    legs = []
    for k in range(len(pivots) - 1):
        i1, t1, p1 = pivots[k]
        i2, t2, p2 = pivots[k + 1]
        dur = i2 - i1
        chg = (p2 / p1 - 1.0) * 100.0
        leg_type = "UP" if t2 == "H" else "DOWN"
        legs.append({
            "Leg": leg_type,
            "Start": m["ym"].iloc[i1],
            "End": m["ym"].iloc[i2],
            "Months": dur,
            "StartPx": round(p1, 2),
            "EndPx": round(p2, 2),
            "Change%": round(chg, 1),
            "PctPerMonth": round(chg / dur, 1) if dur > 0 else 0.0,
        })
    legs_df = pd.DataFrame(legs)
    print("\n--- 1A. All Monthly ZigZag UP vs DOWN Legs (up>=120%, down>=55%) ---")
    print(legs_df.to_string(index=False))

    up_l = legs_df[legs_df["Leg"] == "UP"]
    dn_l = legs_df[legs_df["Leg"] == "DOWN"]
    dn_fast = dn_l[dn_l["Months"] <= 4]
    dn_slow = dn_l[dn_l["Months"] > 4]
    print(f"\n[Obs 1 Summary] UP legs ({len(up_l)}): mean={up_l['Months'].mean():.1f}m, median={up_l['Months'].median():.1f}m, range={up_l['Months'].min()}-{up_l['Months'].max()}m")
    print(f"[Obs 1 Summary] DOWN legs ({len(dn_l)}): mean={dn_l['Months'].mean():.1f}m, median={dn_l['Months'].median():.1f}m, range={dn_l['Months'].min()}-{dn_l['Months'].max()}m")
    print(f"  -> Fast crashes (<=4m, n={len(dn_fast)}): mean_dur={dn_fast['Months'].mean():.1f}m, mean_chg={dn_fast['Change%'].mean():.1f}%, vel={dn_fast['PctPerMonth'].mean():.1f}%/m")
    print(f"  -> Slow grinds  (>4m, n={len(dn_slow)}): mean_dur={dn_slow['Months'].mean():.1f}m, mean_chg={dn_slow['Change%'].mean():.1f}%, vel={dn_slow['PctPerMonth'].mean():.1f}%/m")

    # --- Observation 2: 30W SMA channel during UP legs + Pullback risk by dist30w & dist200w ---
    up_above_flags = []
    up_early_flags = []
    up_mid_late_flags = []
    for k in range(len(pivots) - 1):
        i1, t1, _ = pivots[k]
        i2, t2, _ = pivots[k + 1]
        if t2 == "H":
            dur = i2 - i1
            for j in range(i1, i2 + 1):
                if pd.notna(m["W_SMA30"].iloc[j]):
                    above = float(m["Close"].iloc[j]) > float(m["W_SMA30"].iloc[j])
                    up_above_flags.append(above)
                    if (j - i1) <= max(1, dur // 4):
                        up_early_flags.append(above)
                    else:
                        up_mid_late_flags.append(above)

    print(f"\n--- 2A. Monthly Close > 30W SMA Inside ZigZag UP Legs ---")
    print(f"All UP-leg months above 30W SMA: {np.mean(up_above_flags)*100:.1f}% ({sum(up_above_flags)}/{len(up_above_flags)})")
    print(f"  -> Early quarter of UP legs: {np.mean(up_early_flags)*100:.1f}% ({sum(up_early_flags)}/{len(up_early_flags)})")
    print(f"  -> Remaining 3/4 of UP legs: {np.mean(up_mid_late_flags)*100:.1f}% ({sum(up_mid_late_flags)}/{len(up_mid_late_flags)})")

    # Forward pullback risk by dist30w & dist200w buckets
    m["fwd1m_ret%"] = (m["Close"].shift(-1) / m["Close"] - 1.0) * 100.0
    m["fwd3m_ret%"] = (m["Close"].shift(-3) / m["Close"] - 1.0) * 100.0
    fwd3_min_low = pd.concat([m["Low"].shift(-1), m["Low"].shift(-2), m["Low"].shift(-3)], axis=1).min(axis=1)
    m["fwd3m_min_dd%"] = (fwd3_min_low / m["Close"] - 1.0) * 100.0
    m["fwd3m_pullback20%"] = m["fwd3m_min_dd%"] <= -20.0
    below_30w = (m["Close"] < m["W_SMA30"])
    m["fwd3m_below30w"] = (
        below_30w.shift(-1, fill_value=False)
        | below_30w.shift(-2, fill_value=False)
        | below_30w.shift(-3, fill_value=False)
    )

    valid_m = m.dropna(subset=["dist30w", "fwd3m_ret%"]).copy()
    valid_m["bin30"] = pd.cut(valid_m["dist30w"], bins=[-999, -20, 0, 30, 60, 100, 999],
                              labels=["<-20%", "-20~0%", "0~30%", "30~60%", "60~100%", ">100%"])
    print("\n--- 2B. Forward Pullback Risk by dist30w (Monthly Close vs 30W SMA) ---")
    b30 = valid_m.groupby("bin30", observed=False).agg(
        Months=("Close", "count"),
        Fwd1M_Avg=("fwd1m_ret%", "mean"),
        Fwd3M_Avg=("fwd3m_ret%", "mean"),
        Fwd3M_MedMaxDD=("fwd3m_min_dd%", "median"),
        Pullback20pct_Prob=("fwd3m_pullback20%", lambda s: s.mean() * 100.0),
        TouchBelow30W_Prob=("fwd3m_below30w", lambda s: s.mean() * 100.0),
    ).round(1)
    print(b30.to_string())

    valid_200 = m.dropna(subset=["dist200w", "fwd3m_ret%"]).copy()
    valid_200["bin200"] = pd.cut(valid_200["dist200w"], bins=[-999, 0, 50, 120, 200, 9999],
                                 labels=["<0%", "0~50%", "50~120%", "120~200%", ">200%"])
    print("\n--- 2C. Forward Pullback Risk by dist200w (Monthly Close vs 200W SMA) ---")
    b200 = valid_200.groupby("bin200", observed=False).agg(
        Months=("Close", "count"),
        Fwd1M_Avg=("fwd1m_ret%", "mean"),
        Fwd3M_Avg=("fwd3m_ret%", "mean"),
        Fwd3M_MedMaxDD=("fwd3m_min_dd%", "median"),
        Pullback20pct_Prob=("fwd3m_pullback20%", lambda s: s.mean() * 100.0),
    ).round(1)
    print(b200.to_string())

    # --- Observation 3: Time & Depth Before True ZigZag Bottoms vs Strategy 1 ALT Trades ---
    bot_rows = []
    for k, (idx, tp, pr) in enumerate(pivots):
        if tp == "L":
            prev_top_m = (idx - pivots[k - 1][0]) if k > 0 else np.nan
            bot_rows.append({
                "BottomMonth": m["ym"].iloc[idx],
                "LowPx": round(pr, 2),
                "MonthsFromPrevTop": prev_top_m,
                "DaysSince52WHi": int(m["days_since_hi"].iloc[idx]),
                "DD_from_52WHi%": round(float((pr / (m["Close"].iloc[idx] / (1 + m["dd_from_hi"].iloc[idx] / 100.0)) - 1.0) * 100.0), 1),
                "Close_DD_52WHi%": round(float(m["dd_from_hi"].iloc[idx]), 1),
                "dist30w%": round(float(m["dist30w"].iloc[idx]), 1) if pd.notna(m["dist30w"].iloc[idx]) else np.nan,
                "dist200w%": round(float(m["dist200w"].iloc[idx]), 1) if pd.notna(m["dist200w"].iloc[idx]) else np.nan,
            })
    print("\n--- 3A. Time & Depth at Every True ZigZag Bottom (2010-2026) ---")
    print(pd.DataFrame(bot_rows).to_string(index=False))

    # Inspect all Strategy 1 ALT trades (2015-2026)
    s1_full = simulate_s1_ext(f_soxl, "2015-01-01", "2026-11-19", record_trades=True)
    alt_logs = pd.DataFrame([t for t in s1_full["logs"] if t["mode"] == "ALT"])
    print("\n--- 3B. All Strategy 1 ALT (Panic Bottom) Entries (2015-2026) ---")
    print(alt_logs.to_string(index=False))
    if not alt_logs.empty:
        wins_a = alt_logs[alt_logs["ret_pct"] > 0]
        loss_a = alt_logs[alt_logs["ret_pct"] <= 0]
        print(f"ALT Winners (n={len(wins_a)}): median dd_hi={wins_a['entry_dd_hi'].median():.1f}%, median days_hi={wins_a['entry_days_hi'].median():.0f}d, median dist30w={wins_a['entry_d30'].median():.1f}%, median dist200w={wins_a['entry_d200'].median():.1f}%")
        print(f"ALT Losers  (n={len(loss_a)}): median dd_hi={loss_a['entry_dd_hi'].median():.1f}%, median days_hi={loss_a['entry_days_hi'].median():.0f}d, median dist30w={loss_a['entry_d30'].median():.1f}%, median dist200w={loss_a['entry_d200'].median():.1f}%")


def part2_test_optimizations(f_soxl: dict, f_soxl_raw_dfs: tuple) -> None:
    soxl_df, smh_df = f_soxl_raw_dfs
    print("\n==========================================================================")
    print("PART 2: TESTING dist30w / dist200w / TIME-DEPTH OPTIMIZATIONS ON STRATEGY 1")
    print("==========================================================================")

    # Verify exact baseline parity
    base_25 = simulate_s1_ext(f_soxl, "2015-01-01", "2025-12-31")
    print(f"Baseline S1 (2015-2025): Final=${base_25['Final']:,.2f} | CAGR={base_25['CAGR%']}% | MaxDD={base_25['MaxDD%']}% | Calmar={base_25['Calmar']} | Sharpe={base_25['Sharpe']} | Trades={base_25['Trades']} | WinRate={base_25['WinRate%']}%")

    candidates = []

    def add_cand(group: str, label: str, **kw) -> None:
        r = simulate_s1_ext(f_soxl, "2015-01-01", "2025-12-31", **kw)
        r_clean = {k: v for k, v in r.items() if k != "logs"}
        candidates.append({"Group": group, "Variant": label, **r_clean, "_kw": kw})

    # H1a: Overextended tightening (dist30w / dist200w)
    for th30 in (50, 70, 90, 120):
        for etsl in (0.15, 0.20, 0.25):
            add_cand("H1a_ExtTSL", f"d30>{th30}->tsl={etsl}", ext_d30_th=th30, ext_tsl=etsl)
        add_cand("H1a_ExtEMA", f"d30>{th30}->EMA10", ext_d30_th=th30, ext_ema10_exit=True)
    for th200 in (150, 200, 250, 300):
        for etsl in (0.15, 0.20, 0.25):
            add_cand("H1a_ExtTSL", f"d200>{th200}->tsl={etsl}", ext_d200_th=th200, ext_tsl=etsl)
        add_cand("H1a_ExtEMA", f"d200>{th200}->EMA10", ext_d200_th=th200, ext_ema10_exit=True)

    # H1b: No-chase cap on TREND entry
    for c30 in (60, 80, 100):
        add_cand("H1b_NoChase", f"cap_d30<={c30}", cap_d30=c30)
    for c200 in (180, 250):
        add_cand("H1b_NoChase", f"cap_d200<={c200}", cap_d200=c200)

    # H2: Bottom ALT gating via dist30w, dist200w, or post-top (depth | time)
    for d30m in (-10, -20, -30, -40):
        add_cand("H2_AltGate", f"alt_d30<={d30m}", alt_d30_max=d30m)
    for d200m in (50, 20, 0, -20):
        add_cand("H2_AltGate", f"alt_d200<={d200m}", alt_d200_max=d200m)
    for dd_m in (45, 55, 65):
        for dy_m in (60, 100, 140):
            add_cand("H2_AltGate", f"alt_(dd>={dd_m}|days>={dy_m})", alt_dd_min=dd_m, alt_days_min=dy_m)

    # H3: Depth-conditioned ALT take-profit
    for dtp in (0.35, 0.45, 0.55):
        for dmd in (40, 60):
            add_cand("H3_DeepTP", f"d200<=0->tp{dtp}_{dmd}d", deep_d200_th=0, deep_tp=dtp, deep_max_days=dmd)
            add_cand("H3_DeepTP", f"d30<=-35->tp{dtp}_{dmd}d", deep_d30_th=-35, deep_tp=dtp, deep_max_days=dmd)
            add_cand("H3_DeepTP", f"dd>=65->tp{dtp}_{dmd}d", deep_dd_th=65, deep_tp=dtp, deep_max_days=dmd)

    cdf = pd.DataFrame(candidates)
    cols = ["Group", "Variant", "Final", "CAGR%", "MaxDD%", "Calmar", "Sharpe", "Trades", "WinRate%"]

    print("\n--- Best per Hypothesis Family (2015-2025) ---")
    for grp, sub in cdf.groupby("Group"):
        print(f"\n[{grp}] Top 5 by Calmar:")
        print(sub.sort_values("Calmar", ascending=False).head(5)[cols].to_string(index=False))

    # H4: Combine best H1, H2, H3 candidates
    best_h1 = cdf[cdf["Group"].isin(["H1a_ExtTSL", "H1a_ExtEMA", "H1b_NoChase"])].sort_values("Calmar", ascending=False).head(3)
    best_h2 = cdf[cdf["Group"] == "H2_AltGate"].sort_values("Calmar", ascending=False).head(3)
    best_h3 = cdf[cdf["Group"] == "H3_DeepTP"].sort_values("Calmar", ascending=False).head(3)

    h4_cands = []
    for _, r1 in best_h1.iterrows():
        for _, r2 in best_h2.iterrows():
            kw = {**r1["_kw"], **r2["_kw"]}
            res = simulate_s1_ext(f_soxl, "2015-01-01", "2025-12-31", **kw)
            h4_cands.append({"Group": "H4_Combined", "Variant": f"{r1['Variant']} + {r2['Variant']}", **{k: v for k, v in res.items() if k != "logs"}, "_kw": kw})
        for _, r3 in best_h3.iterrows():
            kw = {**r1["_kw"], **r3["_kw"]}
            res = simulate_s1_ext(f_soxl, "2015-01-01", "2025-12-31", **kw)
            h4_cands.append({"Group": "H4_Combined", "Variant": f"{r1['Variant']} + {r3['Variant']}", **{k: v for k, v in res.items() if k != "logs"}, "_kw": kw})
    for _, r2 in best_h2.iterrows():
        for _, r3 in best_h3.iterrows():
            kw = {**r2["_kw"], **r3["_kw"]}
            res = simulate_s1_ext(f_soxl, "2015-01-01", "2025-12-31", **kw)
            h4_cands.append({"Group": "H4_Combined", "Variant": f"{r2['Variant']} + {r3['Variant']}", **{k: v for k, v in res.items() if k != "logs"}, "_kw": kw})

    h4_df = pd.DataFrame(h4_cands).sort_values("Calmar", ascending=False)
    print("\n--- [H4_Combined] Top 8 by Calmar (2015-2025) ---")
    print(h4_df.head(8)[cols].to_string(index=False))

    # --- Part 3: Multi-Window Generalization Verification ---
    print("\n==========================================================================")
    print("PART 3: MULTI-WINDOW GENERALIZATION CHECK (IS + 2026 OOS + 2011-14 OOS + TQQQ)")
    print("==========================================================================")

    # Select distinct top variants from H1, H2, H3, H4 vs Baseline S1
    selected = [("Baseline S1 (Paired_SMA200_hv95)", {})]
    for grp_df, tag in [(best_h1.head(1), "Best H1"), (best_h2.head(1), "Best H2"), (best_h3.head(1), "Best H3"), (h4_df.head(2), "Best H4")]:
        for _, row in grp_df.iterrows():
            selected.append((f"{tag}: {row['Variant']}", row["_kw"]))

    tqqq_df = load_adjusted("TQQQ")
    qqq_df = load_adjusted("QQQ")
    f_tqqq = build_features(tqqq_df, qqq_df)

    windows = [
        ("SOXL 2015-2025 (IS)", f_soxl, "2015-01-01", "2025-12-31"),
        ("SOXL 2026 OOS", f_soxl, "2026-01-01", "2026-11-19"),
        ("SOXL 2011-2014 Early OOS", f_soxl, "2011-01-01", "2014-12-31"),
        ("TQQQ 2015-2025 Cross-Asset", f_tqqq, "2015-01-01", "2025-12-31"),
    ]

    for w_label, feat, sd, ed in windows:
        w_rows = []
        for s_label, kw in selected:
            res = simulate_s1_ext(feat, sd, ed, **kw)
            w_rows.append({
                "Strategy": s_label,
                "Final": res["Final"],
                "Ret%": res["Ret%"],
                "CAGR%": res["CAGR%"],
                "MaxDD%": res["MaxDD%"],
                "Calmar": res["Calmar"],
                "Sharpe": res["Sharpe"],
                "Trades": res["Trades"],
                "WinRate%": res["WinRate%"],
            })
        print(f"\n--- {w_label} ---")
        print(pd.DataFrame(w_rows).to_string(index=False))


def main() -> None:
    soxl_df = load_adjusted("SOXL")
    smh_df = load_adjusted("SMH")
    f_soxl = build_features(soxl_df, smh_df)
    part1_verify_observations(soxl_df, f_soxl)
    part2_test_optimizations(f_soxl, (soxl_df, smh_df))


if __name__ == "__main__":
    main()
