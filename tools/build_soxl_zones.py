"""Build a dedicated SOXL monthly LOG chart with top/bottom zones and verified strategy overlays."""

from __future__ import annotations

import argparse
import html as html_lib
import importlib.util
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import ema, fisher_transform, historical_volatility, sma

DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "SOXL.csv"
SMH_PATH = DATA_DIR / "SMH.csv"
OUTPUT_HTML = ROOT / "soxl_monthly_zones.html"

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005
BT_START = "2015-01-01"
BT_END = "2026-11-19"


def load_daily_adjusted(csv_path: Path) -> pd.DataFrame:
    """Load daily CSV, drop incomplete rows, and split/dividend-adjust OHLC."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    if "Adj Close" in df.columns:
        ratio = (df["Adj Close"] / df["Close"]).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * ratio
    return df


def load_soxl_daily_adjusted(csv_path: Path) -> pd.DataFrame:
    """Backward-compatible wrapper for loading adjusted SOXL daily data."""
    return load_daily_adjusted(csv_path)


def build_monthly_and_weekly(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Aggregate daily data to monthly and weekly OHLCV records."""
    indexed = df.set_index("Date")

    monthly = (
        indexed.resample("ME")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    monthly_records = [
        {
            "m": idx.strftime("%Y-%m"),
            "o": round(float(r.Open), 4),
            "h": round(float(r.High), 4),
            "l": round(float(r.Low), 4),
            "c": round(float(r.Close), 4),
            "v": int(r.Volume),
        }
        for idx, r in zip(monthly.index, monthly.itertuples(index=False))
    ]

    weekly = (
        indexed.resample("W-FRI")
        .agg({"Close": "last"})
        .dropna(subset=["Close"])
    )
    weekly["sma30w"] = weekly["Close"].rolling(30, min_periods=30).mean()
    weekly["sma200w"] = weekly["Close"].rolling(200, min_periods=200).mean()
    weekly_by_month = weekly.resample("ME").last()
    weekly_sma_records = [
        {
            "m": idx.strftime("%Y-%m"),
            "s30": round(float(r.sma30w), 4) if pd.notna(r.sma30w) else None,
            "s200": round(float(r.sma200w), 4) if pd.notna(r.sma200w) else None,
        }
        for idx, r in zip(weekly_by_month.index, weekly_by_month.itertuples(index=False))
    ]

    return monthly_records, weekly_sma_records


def _agg_periodic(daily_idx_df: pd.DataFrame, freq: str, drop_incomplete: bool = True) -> pd.DataFrame:
    with_dt = daily_idx_df.assign(_dt=daily_idx_df.index)
    grouped = (
        with_dt.groupby(pd.Grouper(freq=freq))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    grouped.index = pd.DatetimeIndex(grouped["_dt"])
    grouped = grouped.drop(columns=["_dt"])
    if drop_incomplete and len(grouped) > 1:
        last_dt = pd.Timestamp(grouped.index[-1]).normalize()
        today = pd.Timestamp.today().normalize()
        if freq == "W-FRI":
            if last_dt.weekday() < 4:
                week_friday = last_dt + pd.Timedelta(days=(4 - last_dt.weekday()))
                if today <= week_friday:
                    grouped = grouped.iloc[:-1]
        elif freq == "ME":
            month_end = (last_dt + pd.offsets.MonthEnd(0)).normalize()
            if last_dt < month_end and today <= month_end:
                grouped = grouped.iloc[:-1]
    return grouped


def build_causal_features(df_soxl: pd.DataFrame, df_smh: pd.DataFrame) -> dict:
    """Build strictly causal daily, weekly, and completed-month features for SOXL driven by SMH."""
    daily = df_soxl.set_index("Date").copy()
    bench = df_smh.set_index("Date").reindex(daily.index, method="ffill")

    daily["HV20"] = historical_volatility(daily["Close"], period=20, annualization_factor=252)
    daily["Bench_Close"] = bench["Close"]
    daily["Bench_SMA200"] = sma(bench["Close"], 200)

    w = _agg_periodic(daily, "W-FRI")
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Sig"] = wf["fisher_signal"]
    w["dW_Fish"] = w["W_Fisher"].diff()
    w["W_SMA30"] = sma(w["Close"], 30)
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_EMA10"] = ema(w["Close"], 10)
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()

    w["Low_vs_200W%"] = (w["Low"] / w["W_SMA200"] - 1.0) * 100.0
    w["Close_vs_30W%"] = (w["Close"] / w["W_SMA30"] - 1.0) * 100.0

    hi_arr = w["High"].to_numpy(dtype=np.float64)
    lo_arr = w["Low"].to_numpy(dtype=np.float64)
    wks_52h = np.empty(len(w), dtype=np.float64)
    dd52_8wl = np.empty(len(w), dtype=np.float64)
    for k in range(len(w)):
        s52 = max(0, k - 51)
        p52 = s52 + int(np.argmax(hi_arr[s52 : k + 1]))
        wks_52h[k] = float(k - p52)
        min8l = float(np.min(lo_arr[max(0, k - 7) : k + 1]))
        dd52_8wl[k] = (min8l / hi_arr[p52] - 1.0) * 100.0
    w["wks_52H"] = wks_52h
    w["dd52_8wL%"] = dd52_8wl

    m = _agg_periodic(daily, "ME")
    mf = fisher_transform(m["High"], m["Low"], period=10)
    m_fish = mf["fisher"]
    m_sig = mf["fisher_signal"]
    m_cross_up = (m_fish > m_sig) & (m_fish.shift(1) <= m_sig.shift(1)) & (m_fish.shift(1) < 0.8)

    w["M_Fish_prev"] = m_fish.reindex(w.index, method="ffill")
    w["M_Fish_Sig_prev"] = m_sig.reindex(w.index, method="ffill")
    daily["M_Fish_prev_ffill"] = m_fish.reindex(daily.index, method="ffill")

    w["MF_Lin_Buy"] = False
    for mdt in m.index[m_cross_up.fillna(False)]:
        w_pos = int(w.index.searchsorted(mdt, side="left"))
        if w_pos < len(w):
            w.iloc[w_pos, w.columns.get_loc("MF_Lin_Buy")] = True

    min_low_200_8w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    max4w_vol_r = w["W_Vol_R"].rolling(4, min_periods=1).max()

    w["prev_W_High"] = w["High"].shift(1)
    w["prev_W_Low"] = w["Low"].shift(1)
    w["W_HigherLow"] = w["Low"] > w["prev_W_Low"]
    w["Close_vs_52H%"] = (w["Close"] / w["High"].rolling(52, min_periods=1).max() - 1.0) * 100.0
    w["min8w_Low_200W_ratio"] = min_low_200_8w
    w["min4w_WF"] = min_wf_4w
    w["max4w_Vol_R"] = max4w_vol_r

    w["pre_c7_vol"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (max4w_vol_r >= 1.6)
    w["E1_C7Vol_EMA10"] = w["pre_c7_vol"] & (w["Close"] > w["W_EMA10"])
    w["E2_PrevHigh"] = w["pre_c7_vol"] & (w["Close"] > w["High"].shift(1))
    w["F1_E2_PrevHigh"] = (
        w["pre_c7_vol"]
        & ((w["wks_52H"] >= 30) | (w["dd52_8wL%"] <= -80.0))
        & (w["Close"] > w["High"].shift(1))
    )
    w["Deep85_HL"] = w["pre_c7_vol"] & (w["Close_vs_52H%"] <= -85.0) & w["W_HigherLow"]
    w["C2_F1_or_MFishLin"] = w["F1_E2_PrevHigh"] | w["MF_Lin_Buy"]
    w["C2_Deep85HL"] = w["F1_E2_PrevHigh"] | w["Deep85_HL"] | w["MF_Lin_Buy"]
    w["ALT_Buy_S1"] = (w["W_Fisher"] < -1.8) & ((w["dW_Fish"] > 0) | (w["W_Vol_R"] > 1.3))

    daily_pos_map = {dt: pos for pos, dt in enumerate(daily.index)}
    week_end_daily_pos = {daily_pos_map[w_dt]: w_pos for w_pos, w_dt in enumerate(w.index)}

    return {
        "daily": daily,
        "weekly": w,
        "week_end_daily_pos": week_end_daily_pos,
    }


def _safe_num(val, decimals: int = 2) -> float | None:
    if val is None or pd.isna(val):
        return None
    f = float(val)
    if not np.isfinite(f):
        return None
    return round(f, decimals)


def build_live_market_snapshot(feats: dict) -> dict:
    """Build latest-bar causal market snapshot for the HTML live cockpit."""
    daily: pd.DataFrame = feats["daily"]
    w: pd.DataFrame = feats["weekly"]
    last_dt = daily.index[-1]
    d_last = daily.iloc[-1]
    w_last = w.iloc[-1]
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = int(last_dt.weekday())

    return {
        "last_date": str(last_dt.date()),
        "weekday_cn": weekday_names[wd],
        "is_friday": wd == 4,
        "soxl_close": _safe_num(d_last["Close"], 2),
        "smh_close": _safe_num(d_last["Bench_Close"], 2),
        "smh_sma200": _safe_num(d_last["Bench_SMA200"], 2),
        "hv20": _safe_num(d_last["HV20"], 1),
        "w_date": str(w.index[-1].date()),
        "w_close": _safe_num(w_last["Close"], 2),
        "w_low": _safe_num(w_last["Low"], 2),
        "prev_w_high": _safe_num(w_last["prev_W_High"], 2),
        "prev_w_low": _safe_num(w_last["prev_W_Low"], 2),
        "w_higher_low": bool(w_last["W_HigherLow"]),
        "close_vs_52h_pct": _safe_num(w_last["Close_vs_52H%"], 1),
        "deep85_hl": bool(w_last["Deep85_HL"]),
        "w_sma30": _safe_num(w_last["W_SMA30"], 2),
        "w_sma200": _safe_num(w_last["W_SMA200"], 2),
        "w_ema10": _safe_num(w_last["W_EMA10"], 2),
        "w_fisher": _safe_num(w_last["W_Fisher"], 3),
        "w_fisher_sig": _safe_num(w_last["W_Fisher_Sig"], 3),
        "dw_fish": _safe_num(w_last["dW_Fish"], 3),
        "w_vol_r": _safe_num(w_last["W_Vol_R"], 2),
        "m_fish_prev": _safe_num(w_last["M_Fish_prev"], 3),
        "m_fish_sig_prev": _safe_num(w_last["M_Fish_Sig_prev"], 3),
        "mf_lin_buy": bool(w_last["MF_Lin_Buy"]),
        "pre_c7_vol": bool(w_last["pre_c7_vol"]),
        "min8w_low_200w": _safe_num(w_last["min8w_Low_200W_ratio"], 3),
        "min4w_wf": _safe_num(w_last["min4w_WF"], 2),
        "max4w_vol_r": _safe_num(w_last["max4w_Vol_R"], 2),
        "wks_52h": _safe_num(w_last["wks_52H"], 0),
        "dd52_8wl": _safe_num(w_last["dd52_8wL%"], 1),
        "close_vs_30w_pct": _safe_num(w_last["Close_vs_30W%"], 1),
    }


def build_soxl_deep85_gap_tracker(feats: dict) -> dict:
    """Build strictly causal SOXL Deep85 weekly-light and true-breakdown gap tracker."""
    daily: pd.DataFrame = feats["daily"]
    w: pd.DataFrame = feats["weekly"]

    armed_s = (w["pre_c7_vol"] & (w["Close_vs_52H%"] <= -85.0)).fillna(False)
    w_last = w.iloc[-1]
    w_date = str(w.index[-1].date())
    pre_c7_ok = bool(w_last["pre_c7_vol"])
    dd52_pct = _safe_num(w_last["Close_vs_52H%"], 1)
    dd52_ok = bool(dd52_pct is not None and dd52_pct <= -85.0)
    is_armed = bool(armed_s.iloc[-1])

    armed_start_date = None
    armed_weeks_count = 0
    anchor_w_close = None
    anchor_w_date = None

    if is_armed:
        streak_indices: list[int] = []
        for wi in range(len(w) - 1, -1, -1):
            if bool(armed_s.iloc[wi]):
                streak_indices.append(wi)
            else:
                break
        streak_indices.reverse()
        armed_weeks_count = len(streak_indices)
        armed_start_date = str(w.index[streak_indices[0]].date())
        sub_w = w.iloc[streak_indices]
        min_idx = sub_w["Close"].idxmin()
        anchor_w_close = _safe_num(sub_w.loc[min_idx, "Close"], 2)
        anchor_w_date = str(min_idx.date())

    last_dt = daily.index[-1]
    last_close = _safe_num(daily["Close"].iloc[-1], 2)
    gap8_trigger_px = round(last_close * 0.92, 2) if last_close is not None else None
    effective_open_max = (
        round(min(gap8_trigger_px, anchor_w_close), 2)
        if (is_armed and gap8_trigger_px is not None and anchor_w_close is not None)
        else None
    )

    def _collect_daily_for_streak(w_pos_indices: list[int], end_dt: pd.Timestamp | None = None) -> list[dict]:
        rows: list[dict] = []
        start_w_ts = w.index[w_pos_indices[0]]
        cur_anchor = float(w.iloc[w_pos_indices[0]]["Close"])
        next_wi_ptr = 1

        d_start_pos = int(daily.index.searchsorted(start_w_ts, side="right"))
        d_end_pos = len(daily) - 1 if end_dt is None else int(daily.index.searchsorted(end_dt, side="right")) - 1

        for d_pos in range(d_start_pos, d_end_pos + 1):
            d_ts = daily.index[d_pos]
            while next_wi_ptr < len(w_pos_indices) and d_ts > w.index[w_pos_indices[next_wi_ptr]]:
                cur_anchor = min(cur_anchor, float(w.iloc[w_pos_indices[next_wi_ptr]]["Close"]))
                next_wi_ptr += 1

            prev_c = float(daily["Close"].iloc[d_pos - 1]) if d_pos > 0 else float(daily["Open"].iloc[d_pos])
            o = float(daily["Open"].iloc[d_pos])
            l = float(daily["Low"].iloc[d_pos])
            c = float(daily["Close"].iloc[d_pos])
            gap_pct = (o / prev_c - 1.0) * 100.0 if prev_c > 0 else 0.0
            dip_from_open_pct = (l / o - 1.0) * 100.0 if o > 0 else 0.0
            gap_ok = gap_pct <= -8.0
            below_anchor = o <= cur_anchor
            triggered = gap_ok and below_anchor

            dip3_px = round(o * 0.97, 2)
            dip5_px = round(o * 0.95, 2)
            dip3_hit = bool(triggered and l <= dip3_px)
            dip5_hit = bool(triggered and l <= dip5_px)

            if triggered:
                status_code = "TRIGGERED"
                status_text = "⚡ 真破位跳空触发"
            elif gap_ok and not below_anchor:
                status_code = "BLOCKED_TRAP"
                status_text = "🛡️ 假跳空拦截 (Open > 锚点)"
            else:
                status_code = "WAIT"
                status_text = "○ 未低开 ≥ 8%"

            rows.append({
                "date": str(d_ts.date()),
                "open": round(o, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "anchor": round(cur_anchor, 2),
                "gap_pct": round(gap_pct, 1),
                "dip_open_pct": round(dip_from_open_pct, 1),
                "gap_ok": bool(gap_ok),
                "below_anchor": bool(below_anchor),
                "triggered": bool(triggered),
                "dip3_px": dip3_px,
                "dip3_hit": dip3_hit,
                "dip5_px": dip5_px,
                "dip5_hit": dip5_hit,
                "status_code": status_code,
                "status_text": status_text,
            })
        return rows

    daily_rows: list[dict] = []
    triggered_days: list[dict] = []
    latest_trigger = None
    if is_armed:
        all_rows = _collect_daily_for_streak(streak_indices)
        triggered_days = [r for r in all_rows if r["triggered"]]
        latest_trigger = triggered_days[-1] if triggered_days else None
        daily_rows = all_rows[-15:]

    if not is_armed:
        stage_code = "OFF"
        stage_label = "熄灭 · 未进入 Deep85 亮灯区"
        stage_tone = "muted"
    elif latest_trigger is None:
        stage_code = "ARMED_WAIT"
        stage_label = "🔥 已亮灯 · 等待开盘真破位跳空 (≤ -8%)"
        stage_tone = "warn"
    else:
        stage_code = "TRIGGERED"
        stage_label = "⚡ 已亮灯 · 本轮已触发真破位跳空信号"
        stage_tone = "ready"

    # Collect independent historical Deep85_Armed episodes (gap > 90 days)
    armed_wis = [wi for wi in range(len(w)) if bool(armed_s.iloc[wi])]
    episodes_wis: list[list[int]] = []
    for wi in armed_wis:
        if not episodes_wis or (w.index[wi] - w.index[episodes_wis[-1][-1]]).days > 90:
            episodes_wis.append([wi])
        else:
            episodes_wis[-1].append(wi)

    history_episodes: list[dict] = []
    for ep_wis in episodes_wis:
        first_wi = ep_wis[0]
        last_wi = ep_wis[-1]
        ep_end_dt = None if last_wi == len(w) - 1 else w.index[last_wi] + pd.Timedelta(days=7)
        ep_daily = _collect_daily_for_streak(ep_wis, end_dt=ep_end_dt)
        ep_trigs = [r for r in ep_daily if r["triggered"]]
        ft = ep_trigs[0] if ep_trigs else None
        history_episodes.append({
            "armed_date": str(w.index[first_wi].date()),
            "anchor_close": _safe_num(w.iloc[ep_wis]["Close"].min(), 2),
            "first_trigger_date": ft["date"] if ft else None,
            "trigger_open": ft["open"] if ft else None,
            "trigger_gap_pct": ft["gap_pct"] if ft else None,
            "trigger_low": ft["low"] if ft else None,
            "dip3_hit": ft["dip3_hit"] if ft else False,
            "dip5_hit": ft["dip5_hit"] if ft else False,
            "dip5_px": ft["dip5_px"] if ft else None,
        })

    return {
        "w_date": w_date,
        "last_date": str(last_dt.date()),
        "last_close": last_close,
        "is_armed": is_armed,
        "pre_c7_vol": pre_c7_ok,
        "dd52_pct": dd52_pct,
        "dd52_ok": dd52_ok,
        "min8w_low_200w": _safe_num(w_last["min8w_Low_200W_ratio"], 2),
        "min4w_wf": _safe_num(w_last["min4w_WF"], 2),
        "max4w_vol_r": _safe_num(w_last["max4w_Vol_R"], 2),
        "armed_start_date": armed_start_date,
        "armed_weeks_count": armed_weeks_count,
        "anchor_w_close": anchor_w_close,
        "anchor_w_date": anchor_w_date,
        "gap8_trigger_px": gap8_trigger_px,
        "effective_open_max": effective_open_max,
        "stage_code": stage_code,
        "stage_label": stage_label,
        "stage_tone": stage_tone,
        "latest_trigger": latest_trigger,
        "daily_rows": daily_rows,
        "history_episodes": history_episodes,
    }


def _compute_stats(
    equity: np.ndarray,
    trades: list[dict],
    days_invested: int,
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    n = len(equity)
    final_val = float(equity[-1])
    years_span = n / 252.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0

    cum_max = np.maximum.accumulate(equity)
    max_dd = float(((equity / cum_max) - 1.0).min() * 100.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    daily_ret = np.diff(equity) / equity[:-1]
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252.0)) if std > 1e-12 else 0.0

    trades_cnt = len(trades)
    win_rate = (sum(1 for t in trades if t["ret_pct"] > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = days_invested / n * 100.0

    yrs = dates.year.to_numpy()
    yearly_rets: dict[int, float] = {}
    prev_eq = initial_capital
    for yr in sorted(set(yrs)):
        last_pos = int(np.where(yrs == yr)[0][-1])
        yr_eq = float(equity[last_pos])
        yearly_rets[int(yr)] = (yr_eq / prev_eq - 1.0) * 100.0
        prev_eq = yr_eq
    worst_year = min(yearly_rets.values()) if yearly_rets else 0.0

    return {
        "final": round(final_val, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "sharpe": round(sharpe, 2),
        "trades": trades_cnt,
        "win_rate": round(win_rate, 1),
        "exposure": round(exposure, 1),
        "worst_year": round(worst_year, 1),
    }


def simulate_d1_x1a_wtsl35(
    feats: dict,
    use_split_50_50: bool = False,
    start_date: str = BT_START,
    end_date: str = BT_END,
) -> dict:
    """Simulate Strategy 3: D1_X1a_WTSL35 (full or 50/50 split entry)."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    tranches = 0
    first_entry_px = 0.0
    avg_entry_px = 0.0
    entry_date_str = ""
    entry_m_str = ""
    tranche2_date_str = "-"
    tranche2_m_str: str | None = None
    tranche2_px: float | None = None
    entry_fill_pos = -1
    trade_cash_start = INITIAL_CAPITAL
    trade_mode = ""

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_entry_mode = ""
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        just_exited = False
        if pending_action == "BUY_FULL" and tranches == 0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            tranches = 2
            first_entry_px = float(o)
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            tranche2_date_str = "-"
            tranche2_m_str = None
            tranche2_px = None
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
            sl_streak = 0
        elif pending_action == "BUY_HALF" and tranches == 0:
            trade_cash_start = cash
            cost = 0.5 * cash
            shares = (cost * (1.0 - FEE_RATE)) / o
            cash -= cost
            tranches = 1
            first_entry_px = float(o)
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            tranche2_date_str = "-"
            tranche2_m_str = None
            tranche2_px = None
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
            sl_streak = 0
        elif pending_action == "BUY_TOPUP" and tranches == 1 and shares > 0:
            new_shares = (cash * (1.0 - FEE_RATE)) / o
            tot = shares + new_shares
            avg_entry_px = (shares * avg_entry_px + new_shares * o) / tot
            shares = tot
            cash = 0.0
            tranches = 2
            tranche2_date_str = dt_str
            tranche2_m_str = m_str
            tranche2_px = round(float(o), 2)
            if c > peak_d_close:
                peak_d_close = c
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            proceeds = shares * o * (1.0 - FEE_RATE)
            cash += proceeds
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append(
                {
                    "mode": trade_mode,
                    "entry_date": entry_date_str,
                    "entry_m": entry_m_str,
                    "tranche2_date": tranche2_date_str,
                    "tranche2_m": tranche2_m_str,
                    "tranche2_px": tranche2_px,
                    "entry_px": round(first_entry_px, 2),
                    "avg_entry_px": round(avg_entry_px, 2),
                    "exit_date": dt_str,
                    "exit_m": m_str,
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            if pending_reason.startswith("SL_"):
                cooldown_until_pos = pos + 10
            shares = 0.0
            tranches = 0
            avg_entry_px = 0.0
            just_exited = True
        pending_action = ""
        pending_reason = ""
        pending_entry_mode = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end or pos not in week_end_daily_pos:
            continue

        w_row = w_df.iloc[week_end_daily_pos[pos]]

        if shares > 0:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            mf = w_row["M_Fish_prev"]
            unmatured = (pd.isna(mf) or float(mf) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
            elif pos > entry_fill_pos and pd.notna(mf) and float(mf) >= 2.0:
                vs30 = w_row["Close_vs_30W%"]
                if pd.notna(vs30) and float(vs30) < 0.0:
                    pending_action = "SELL"
                    pending_reason = "M>=2&<30W0"
                elif peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
                    pending_action = "SELL"
                    pending_reason = "M>=2&WTSL35%"

            if pending_action == "" and use_split_50_50 and tranches == 1 and bool(w_row["E1_C7Vol_EMA10"]):
                pending_action = "BUY_TOPUP"

        elif not just_exited and pos >= cooldown_until_pos:
            if use_split_50_50:
                if bool(w_row["E1_C7Vol_EMA10"]):
                    pending_action = "BUY_FULL"
                    pending_entry_mode = "BOT_E1"
                elif bool(w_row["E2_PrevHigh"]):
                    pending_action = "BUY_HALF"
                    pending_entry_mode = "BOT_SPLIT"
            elif bool(w_row["E2_PrevHigh"]):
                pending_action = "BUY_FULL"
                pending_entry_mode = "BOT_E2"

    if shares > 0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trades.append(
            {
                "mode": trade_mode,
                "entry_date": entry_date_str,
                "entry_m": entry_m_str,
                "tranche2_date": tranche2_date_str,
                "tranche2_m": tranche2_m_str,
                "tranche2_px": tranche2_px,
                "entry_px": round(first_entry_px, 2),
                "avg_entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    # Evaluate strictly causal live action at final bar i_end for next trading day's Open
    w_live = w_df.iloc[week_end_daily_pos[i_end]] if i_end in week_end_daily_pos else w_df.iloc[-1]
    cooldown_rem = max(0, cooldown_until_pos - i_end) if shares == 0 else 0
    if shares > 0:
        wc_live = float(w_live["Close"])
        cur_peak_w = max(peak_w_close, wc_live)
        mf_live = w_live["M_Fish_prev"]
        unmatured_live = (pd.isna(mf_live) or float(mf_live) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
        live_act, live_badge, live_tone, live_reason = (
            "HOLD",
            "继续持有",
            "hold",
            "持仓中，未触发假底止损或逃顶信号",
        )
        if unmatured_live and wc_live < avg_entry_px * 0.75:
            live_act, live_badge, live_tone, live_reason = (
                "SELL",
                "卖出 (假底止损)",
                "sell",
                f"触发 SL_W25%: 周收盘 {wc_live:.2f} < 均价75% ({avg_entry_px * 0.75:.2f})",
            )
        elif i_end > entry_fill_pos and pd.notna(mf_live) and float(mf_live) >= 2.0:
            vs30_live = w_live["Close_vs_30W%"]
            if pd.notna(vs30_live) and float(vs30_live) < 0.0:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (破30W逃顶)",
                    "sell",
                    f"M_Fish≥2.0 ({float(mf_live):.2f}) 且周收盘跌破 30周SMA",
                )
            elif cur_peak_w > 0.0 and wc_live <= cur_peak_w * 0.65:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (周回撤35%逃顶)",
                    "sell",
                    f"M_Fish≥2.0 ({float(mf_live):.2f}) 且自峰值 {cur_peak_w:.2f} 回撤≥35%",
                )
        if live_act == "HOLD" and use_split_50_50 and tranches == 1 and bool(w_live["E1_C7Vol_EMA10"]):
            live_act, live_badge, live_tone, live_reason = (
                "BUY_TOPUP",
                "加仓补满 (50%→100%)",
                "buy",
                "满足 E1_C7Vol_EMA10，下个交易日开盘补齐剩余 50%",
            )
        live_ret = round(float((cash + shares * closes[i_end]) / trade_cash_start - 1.0) * 100.0, 1)
        pos_pct = 50 if (use_split_50_50 and tranches == 1) else 100
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": f"持仓 {pos_pct}% ({trade_mode})",
            "pos_mode": "BOT",
            "entry_date": entry_date_str,
            "entry_px": round(avg_entry_px, 2),
            "ret_pct": live_ret,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": round(avg_entry_px * 0.75, 2) if unmatured_live else None,
            "wtsl_price": round(cur_peak_w * 0.65, 2) if cur_peak_w > 0.0 else None,
            "cooldown_days": 0,
            "reason": live_reason,
        }
    else:
        live_act, live_badge, live_tone, live_reason = (
            "WAIT",
            "保持空仓",
            "wait",
            f"冷静期剩余 {cooldown_rem} 个交易日" if cooldown_rem > 0 else "空仓观望，等待大底入场信号",
        )
        if not just_exited and i_end >= cooldown_until_pos:
            if use_split_50_50:
                if bool(w_live["E1_C7Vol_EMA10"]):
                    live_act, live_badge, live_tone, live_reason = (
                        "BUY",
                        "买入 100% (E1建仓)",
                        "buy",
                        "满足 pre_c7_vol 且周收盘 > 10周EMA，下个交易日开盘全仓买入",
                    )
                elif bool(w_live["E2_PrevHigh"]):
                    live_act, live_badge, live_tone, live_reason = (
                        "BUY_HALF",
                        "买入 50% (E2分批建仓)",
                        "buy",
                        "满足 pre_c7_vol 且周收盘 > 上周高点，下个交易日开盘先买 50%",
                    )
            elif bool(w_live["E2_PrevHigh"]):
                live_act, live_badge, live_tone, live_reason = (
                    "BUY",
                    "买入 100% (E2建仓)",
                    "buy",
                    "满足 pre_c7_vol 且周收盘 > 上周高点，下个交易日开盘全仓买入",
                )
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": "空仓 (0%)",
            "pos_mode": "FLAT",
            "entry_date": None,
            "entry_px": None,
            "ret_pct": None,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": None,
            "wtsl_price": None,
            "cooldown_days": cooldown_rem,
            "reason": live_reason,
        }

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    if use_split_50_50:
        return {
            "id": "D1_X1a_WTSL35_Split50",
            "name": "策略 3 分批版: D1_X1a_WTSL35_Split50",
            "short_label": "策略3分批 (D1_X1a_WTSL35_Split50)",
            "rule_desc": "pre_c7_vol + PrevHigh先买50%/EMA10补满 | 假底SL_W25% | M_Fish≥2.0后 破30W或周回撤35%逃顶",
            "stats": stats,
            "trades": trades,
            "live_status": live_status,
        }
    return {
        "id": "D1_X1a_WTSL35",
        "name": "策略 3 极简大周期: D1_X1a_WTSL35",
        "short_label": "策略3 (D1_X1a_WTSL35)",
        "rule_desc": "pre_c7_vol + 周收盘>上周高点买入 | 假底SL_W25% | M_Fish≥2.0后 破30W或周回撤35%逃顶",
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
    }


def simulate_h5_adapt_hv(
    feats: dict,
    start_date: str = BT_START,
    end_date: str = BT_END,
) -> dict:
    """Simulate Strategy 4: H5_E2_Promote_S1_AdaptHV."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)
    hv20 = daily_df["HV20"].to_numpy(dtype=np.float64)
    b_close = daily_df["Bench_Close"].to_numpy(dtype=np.float64)
    b_sma200 = daily_df["Bench_SMA200"].to_numpy(dtype=np.float64)
    m_fish_ffill = daily_df["M_Fish_prev_ffill"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    pos_mode = 0  # 0=FLAT, 1=TREND, 2=ALT, 3=BOT
    initial_mode_str = ""
    promoted_in_trade = False
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_reason = "INIT"
    peak_d_close = 0.0

    pending = 0
    pending_reason = ""

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, i in enumerate(range(i_start, i_end + 1)):
        o = opens[i]
        c = closes[i]

        just_exited = False
        if pending in (1, 2, 3) and pos_mode == 0:
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            pos_mode = pending
            initial_mode_str = {1: "TREND", 2: "ALT", 3: "BOT"}[pending]
            promoted_in_trade = False
            entry_px = float(o)
            entry_idx = i
            peak_d_close = c
        elif pending == -1 and pos_mode != 0:
            if i <= entry_idx:
                raise AssertionError(f"Same-day buy-sell on {idx[i]}")
            proceeds = shares * o * (1.0 - FEE_RATE)
            ret_pct = ((o * (1.0 - FEE_RATE)) / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
            mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
            trades.append(
                {
                    "mode": mode_label,
                    "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                    "entry_m": idx[entry_idx].strftime("%Y-%m"),
                    "tranche2_date": "-",
                    "tranche2_m": None,
                    "tranche2_px": None,
                    "entry_px": round(entry_px, 2),
                    "avg_entry_px": round(entry_px, 2),
                    "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "exit_m": idx[i].strftime("%Y-%m"),
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            cash = proceeds
            shares = 0.0
            pos_mode = 0
            exit_idx = i
            last_exit_reason = pending_reason
            just_exited = True
        pending = 0
        pending_reason = ""

        equity[step] = cash + shares * c
        if pos_mode != 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if i >= i_end:
            continue

        is_w_end = i in week_end_daily_pos
        w_row = w_df.iloc[week_end_daily_pos[i]] if is_w_end else None

        tb_raw = (not np.isnan(b_sma200[i])) and (b_close[i] > b_sma200[i] * 1.01)
        ma_break = (not np.isnan(b_sma200[i])) and (b_close[i] < b_sma200[i] * 0.99)
        hv = hv20[i]
        mf = m_fish_ffill[i]

        high_mf = (not np.isnan(mf)) and (mf >= 1.5)
        vol_exit = (not np.isnan(hv)) and ((high_mf and hv > 115.0) or (hv > 140.0))
        tb_def = tb_raw and (not np.isnan(hv)) and (hv <= 115.0 * 0.92)
        tb_after_vol = tb_raw and (not np.isnan(hv)) and (hv <= 95.0) and (not vol_exit)

        if pos_mode == 3:
            hit_sl = False
            if is_w_end and w_row is not None:
                m_w = w_row["M_Fish_prev"]
                unmatured = (pd.isna(m_w) or float(m_w) < 1.5) and (peak_d_close < entry_px * 1.50)
                if unmatured and float(w_row["Close"]) < entry_px * 0.75:
                    hit_sl = True
            if hit_sl:
                pending = -1
                pending_reason = "SL_W25"
            if pending == 0 and tb_raw and (not vol_exit) and (not ma_break):
                pos_mode = 1
                promoted_in_trade = True

        elif pos_mode == 1:
            if vol_exit:
                pending = -1
                pending_reason = "VOL_SPIKE"
            elif ma_break:
                pending = -1
                pending_reason = "TREND_BREAK"

        elif pos_mode == 2:
            if c <= entry_px * 0.80:
                pending = -1
                pending_reason = "ALT_FSL"
            elif c >= entry_px * 1.25 or (i - entry_idx) >= 40:
                pending = -1
                pending_reason = "ALT_TP"

        elif not just_exited:
            elapsed = i - exit_idx
            bot_sig = bool(is_w_end and w_row is not None and bool(w_row["E2_PrevHigh"]))
            alt_sig = bool(is_w_end and w_row is not None and bool(w_row["ALT_Buy_S1"]))

            if last_exit_reason == "SL_W25":
                if elapsed >= 10 and bot_sig:
                    pending = 3
            else:
                bot_allowed = (elapsed >= 10) if last_exit_reason in ("ALT_FSL", "ALT_TP") else True
                if last_exit_reason in ("INIT", "TREND_BREAK"):
                    if tb_def:
                        pending = 1
                    elif bot_allowed and bot_sig:
                        pending = 3
                    elif alt_sig:
                        pending = 2
                elif last_exit_reason == "VOL_SPIKE":
                    if tb_after_vol:
                        pending = 1
                    elif bot_allowed and bot_sig:
                        pending = 3
                    elif alt_sig:
                        pending = 2
                elif last_exit_reason in ("ALT_FSL", "ALT_TP"):
                    if elapsed >= 10:
                        if tb_def:
                            pending = 1
                        elif bot_sig:
                            pending = 3
                        elif alt_sig:
                            pending = 2

    if pos_mode != 0:
        ret_pct = (closes[i_end] / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
        trades.append(
            {
                "mode": mode_label,
                "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                "entry_m": idx[entry_idx].strftime("%Y-%m"),
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(entry_px, 2),
                "avg_entry_px": round(entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    # Evaluate strictly causal live action at final bar i_end for next trading day's Open
    hv_live = hv20[i_end]
    bc_live = b_close[i_end]
    bs_live = b_sma200[i_end]
    c_live = closes[i_end]
    m_ff_live = m_fish_ffill[i_end]
    w_live = w_df.iloc[week_end_daily_pos[i_end]] if i_end in week_end_daily_pos else None
    hv_limit_live = 140.0 if (np.isnan(m_ff_live) or m_ff_live < 1.5) else 115.0
    vol_exit_live = (not np.isnan(hv_live)) and (hv_live > hv_limit_live)
    valid_bench_live = (not np.isnan(bc_live)) and (not np.isnan(bs_live))
    ma_break_live = valid_bench_live and (bc_live < bs_live * 0.99)
    tb_raw_live = valid_bench_live and (bc_live > bs_live * 1.01)
    tb_def_live = tb_raw_live and (not np.isnan(hv_live)) and (hv_live <= 115.0 * 0.92)
    tb_after_vol_live = tb_raw_live and (not np.isnan(hv_live)) and (hv_live <= 95.0) and (not vol_exit_live)

    if pos_mode != 0:
        live_ret = round(float((c_live / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0), 1)
        cur_mode_name = {1: "TREND", 2: "ALT", 3: "BOT"}[pos_mode]
        live_act, live_badge, live_tone, live_reason = (
            "HOLD",
            "继续持有",
            "hold",
            f"{cur_mode_name} 持仓中，未触发离场条件",
        )
        if pos_mode == 3:
            hit_sl_live = False
            if w_live is not None:
                m_w_live = w_live["M_Fish_prev"]
                unm_live = (pd.isna(m_w_live) or float(m_w_live) < 1.5) and (peak_d_close < entry_px * 1.50)
                if unm_live and float(w_live["Close"]) < entry_px * 0.75:
                    hit_sl_live = True
            if hit_sl_live:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (假底止损)",
                    "sell",
                    f"触发 SL_W25: 周收盘 < 买入价75% ({entry_px * 0.75:.2f})",
                )
            elif tb_raw_live and (not vol_exit_live) and (not ma_break_live):
                live_reason = "SMH > 200SMA×1.01 且波动率可控，BOT 已晋升为 TREND 趋势仓"
        elif pos_mode == 1:
            if vol_exit_live:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (波动率刹车)",
                    "sell",
                    f"HV20={hv_live:.1f}% > 自适应上限 {hv_limit_live:.0f}% (VOL_SPIKE)",
                )
            elif ma_break_live:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (趋势跌破)",
                    "sell",
                    f"SMH 收盘 {bc_live:.2f} < 200SMA×0.99 ({bs_live * 0.99:.2f})",
                )
        elif pos_mode == 2:
            if c_live <= entry_px * 0.80:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (ALT止损-20%)",
                    "sell",
                    f"收盘 {c_live:.2f} ≤ 买入价80% ({entry_px * 0.80:.2f})",
                )
            elif c_live >= entry_px * 1.25 or (i_end - entry_idx) >= 40:
                live_act, live_badge, live_tone, live_reason = (
                    "SELL",
                    "卖出 (ALT止盈/超时)",
                    "sell",
                    f"触发 +25% 止盈或持仓满 40 日 ({i_end - entry_idx} 日)",
                )
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": f"持仓 100% ({initial_mode_str + ('->TREND' if promoted_in_trade else '')})",
            "pos_mode": cur_mode_name,
            "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
            "entry_px": round(entry_px, 2),
            "ret_pct": live_ret,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": round(entry_px * (0.80 if pos_mode == 2 else 0.75), 2) if pos_mode in (2, 3) else None,
            "wtsl_price": None,
            "cooldown_days": 0,
            "reason": live_reason,
        }
    else:
        elapsed_live = i_end - exit_idx
        cooldown_rem = max(0, 10 - elapsed_live) if last_exit_reason in ("SL_W25", "ALT_FSL", "ALT_TP") else 0
        bot_sig_live = bool(w_live is not None and bool(w_live["E2_PrevHigh"]))
        alt_sig_live = bool(w_live is not None and bool(w_live["ALT_Buy_S1"]))
        pending_live = 0
        live_reason = "空仓观望，等待趋势或恐慌/大底信号"
        if not just_exited:
            if last_exit_reason == "SL_W25":
                if elapsed_live >= 10 and bot_sig_live:
                    pending_live = 3
            else:
                bot_allowed_live = (elapsed_live >= 10) if last_exit_reason in ("ALT_FSL", "ALT_TP") else True
                if last_exit_reason in ("INIT", "TREND_BREAK"):
                    if tb_def_live:
                        pending_live = 1
                    elif bot_allowed_live and bot_sig_live:
                        pending_live = 3
                    elif alt_sig_live:
                        pending_live = 2
                elif last_exit_reason == "VOL_SPIKE":
                    if tb_after_vol_live:
                        pending_live = 1
                    elif bot_allowed_live and bot_sig_live:
                        pending_live = 3
                    elif alt_sig_live:
                        pending_live = 2
                elif last_exit_reason in ("ALT_FSL", "ALT_TP") and elapsed_live >= 10:
                    if tb_def_live:
                        pending_live = 1
                    elif bot_sig_live:
                        pending_live = 3
                    elif alt_sig_live:
                        pending_live = 2
        if pending_live == 1:
            live_act, live_badge, live_tone, live_reason = (
                "BUY",
                "买入 100% (TREND趋势)",
                "buy",
                f"SMH > 200SMA×1.01 且 HV20={hv_live:.1f}% 达标",
            )
        elif pending_live == 3:
            live_act, live_badge, live_tone, live_reason = (
                "BUY",
                "买入 100% (BOT大底)",
                "buy",
                "满足 E2_PrevHigh 大底早买信号",
            )
        elif pending_live == 2:
            live_act, live_badge, live_tone, live_reason = (
                "BUY",
                "买入 100% (ALT恐慌底)",
                "buy",
                "满足 W_Fisher<-1.8 恐慌抄底信号",
            )
        else:
            live_act, live_badge, live_tone = "WAIT", "保持空仓", "wait"
            if cooldown_rem > 0:
                live_reason = f"离场冷静期剩余 {cooldown_rem} 个交易日"
            elif last_exit_reason == "VOL_SPIKE":
                live_reason = f"上次因高波离场，需等 HV20 ≤ 95%（当前 {hv_live:.1f}%）"
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": "空仓 (0%)",
            "pos_mode": "FLAT",
            "entry_date": None,
            "entry_px": None,
            "ret_pct": None,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": None,
            "wtsl_price": None,
            "cooldown_days": cooldown_rem,
            "reason": live_reason,
        }

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    return {
        "id": "H5_E2_Promote_S1_AdaptHV",
        "name": "策略 4 双引擎旗舰: H5_E2_Promote_S1_AdaptHV",
        "short_label": "策略4 (H5_E2_Promote_S1_AdaptHV)",
        "rule_desc": "E2大底早买(SL_W25)->无缝晋升SMH200D趋势仓 | 自适应HV刹车(M_Fish<1.5容忍140%, ≥1.5限115%) | 熊市ALT快撤",
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
    }


def simulate_tp3h(
    feats: dict,
    use_f1_filter: bool = True,
    start_date: str = BT_START,
    end_date: str = BT_END,
    entry_col_override: str | None = None,
    mode_tag_override: str | None = None,
    strategy_id: str | None = None,
    tp_split_lottery_pct: float = 0.0,
    consec_sl_weeks: int = 1,
) -> dict:
    """Simulate upgraded Strategy 3 with proactive high-zone TP3h (with or without F1 SOXL anti-fake-bottom filter)."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    entry_col = entry_col_override or ("F1_E2_PrevHigh" if use_f1_filter else "E2_PrevHigh")
    mode_tag = mode_tag_override or ("BOT_F1" if use_f1_filter else "BOT_E2")
    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    highs = daily_df["High"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    trade_cash_start = INITIAL_CAPITAL
    avg_entry_px = 0.0
    entry_date_str = ""
    entry_m_str = ""
    entry_fill_pos = -1
    entry_sig_w_pos = -1
    high_zone_armed = False

    lot_active = False
    lot_px = 0.0
    lot_exp_pos = -1
    lot_first_leg_px = 0.0

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_sig_w_pos = -1
    cooldown_until_pos = -1
    sl_streak = 0

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        h = highs[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        just_exited = False
        if pending_action == "BUY" and shares == 0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            entry_fill_pos = pos
            entry_sig_w_pos = pending_sig_w_pos
            high_zone_armed = False
            lot_active = False
            peak_d_close = c
            peak_w_close = 0.0
            sl_streak = 0
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            if tp_split_lottery_pct > 0.0 and pending_reason == "HighStr40&WFish<Sig":
                sell_sh = shares * 0.5
                cash += sell_sh * o * (1.0 - FEE_RATE)
                shares -= sell_sh
                lot_active = True
                lot_px = float(o) * (1.0 + tp_split_lottery_pct)
                lot_exp_pos = pos + 4
                lot_first_leg_px = float(o)
            else:
                proceeds = shares * o * (1.0 - FEE_RATE)
                ret_pct = ((o * (1.0 - FEE_RATE)) / (avg_entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
                trades.append(
                    {
                        "mode": mode_tag,
                        "entry_date": entry_date_str,
                        "entry_m": entry_m_str,
                        "tranche2_date": "-",
                        "tranche2_m": None,
                        "tranche2_px": None,
                        "entry_px": round(avg_entry_px, 2),
                        "avg_entry_px": round(avg_entry_px, 2),
                        "exit_date": dt_str,
                        "exit_m": m_str,
                        "exit_px": round(float(o), 2),
                        "ret_pct": round(float(ret_pct), 1),
                        "exit_reason": pending_reason,
                        "is_open": False,
                    }
                )
                if pending_reason.startswith("SL_"):
                    cooldown_until_pos = pos + 10
                cash = proceeds
                shares = 0.0
                avg_entry_px = 0.0
                just_exited = True
        pending_action = ""
        pending_reason = ""

        if lot_active and shares > 0:
            hit_lot = h >= lot_px
            if hit_lot or pos >= lot_exp_pos:
                fill_px = max(float(o), lot_px) if hit_lot else float(c)
                cash += shares * fill_px * (1.0 - FEE_RATE)
                ret_pct = (cash / trade_cash_start - 1.0) * 100.0
                avg_exit_px = 0.5 * lot_first_leg_px + 0.5 * fill_px
                exit_tag = f"TP3h_50%+10%({'中奖' if hit_lot else '5日兜底'})"
                trades.append(
                    {
                        "mode": mode_tag,
                        "entry_date": entry_date_str,
                        "entry_m": entry_m_str,
                        "tranche2_date": "-",
                        "tranche2_m": None,
                        "tranche2_px": None,
                        "entry_px": round(avg_entry_px, 2),
                        "avg_entry_px": round(avg_entry_px, 2),
                        "exit_date": dt_str,
                        "exit_m": m_str,
                        "exit_px": round(avg_exit_px, 2),
                        "ret_pct": round(float(ret_pct), 1),
                        "exit_reason": exit_tag,
                        "is_open": False,
                    }
                )
                shares = 0.0
                avg_entry_px = 0.0
                lot_active = False
                high_zone_armed = False
                just_exited = True

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end or pos not in week_end_daily_pos:
            continue

        w_pos = week_end_daily_pos[pos]
        w_row = w_df.iloc[w_pos]

        if shares > 0 and not lot_active:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            mf = w_row["M_Fish_prev"]
            vs30 = w_row["Close_vs_30W%"]
            unmatured = (pd.isna(mf) or float(mf) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if unmatured and wc < avg_entry_px * 0.75:
                sl_streak += 1
                if sl_streak >= consec_sl_weeks:
                    pending_action = "SELL"
                    pending_reason = "SL_W25%"
            else:
                sl_streak = 0
            if pending_action == "" and pos > entry_fill_pos:
                hold_weeks = w_pos - entry_sig_w_pos
                if hold_weeks >= 50 and pd.notna(mf) and float(mf) >= 2.0 and pd.notna(vs30) and float(vs30) >= 40.0:
                    high_zone_armed = True
                if (
                    high_zone_armed
                    and pd.notna(w_row["W_Fisher"])
                    and pd.notna(w_row["W_Fisher_Sig"])
                    and float(w_row["W_Fisher"]) < float(w_row["W_Fisher_Sig"])
                ):
                    pending_action = "SELL"
                    pending_reason = "HighStr40&WFish<Sig"
                elif pd.notna(mf) and float(mf) >= 2.0:
                    if pd.notna(vs30) and float(vs30) < 0.0:
                        pending_action = "SELL"
                        pending_reason = "M>=2&<30W0"
                    elif peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
                        pending_action = "SELL"
                        pending_reason = "M>=2&WTSL35%"

        elif not lot_active and not just_exited and pos >= cooldown_until_pos:
            if bool(w_row[entry_col]):
                pending_action = "BUY"
                pending_sig_w_pos = w_pos

    if shares > 0:
        ret_pct = (closes[i_end] / (avg_entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        trades.append(
            {
                "mode": mode_tag,
                "entry_date": entry_date_str,
                "entry_m": entry_m_str,
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(avg_entry_px, 2),
                "avg_entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    # Evaluate strictly causal live action at final bar i_end for next trading day's Open
    w_pos_live = week_end_daily_pos[i_end] if i_end in week_end_daily_pos else (len(w_df) - 1)
    w_live = w_df.iloc[w_pos_live]
    cooldown_rem = max(0, cooldown_until_pos - i_end) if shares == 0 else 0
    if shares > 0:
        wc_live = float(w_live["Close"])
        cur_peak_w = max(peak_w_close, wc_live)
        mf_live = w_live["M_Fish_prev"]
        vs30_live = w_live["Close_vs_30W%"]
        unmatured_live = (pd.isna(mf_live) or float(mf_live) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
        hold_w_live = int(w_pos_live - entry_sig_w_pos)
        armed_live = bool(
            high_zone_armed
            or (hold_w_live >= 50 and pd.notna(mf_live) and float(mf_live) >= 2.0 and pd.notna(vs30_live) and float(vs30_live) >= 40.0)
        )
        live_act, live_badge, live_tone, live_reason = (
            "HOLD",
            "继续持有",
            "hold",
            f"已持仓 {hold_w_live} 周，" + ("高位止盈区已激活，等待周Fisher死叉" if armed_live else "等待高位止盈区激活 (50w+M_Fish≥2+30W≥40%)"),
        )
        if lot_active:
            live_act, live_badge, live_tone, live_reason = (
                "SELL",
                "挂单止盈中 (剩50%)",
                "sell",
                f"已于开盘卖出50% (@ ${lot_first_leg_px:.2f})，剩余50%挂限价 ${lot_px:.2f} (+{int(tp_split_lottery_pct*100)}%)，第5交易日收盘兜底全平",
            )
        elif unmatured_live and wc_live < avg_entry_px * 0.75:
            live_act, live_badge, live_tone, live_reason = (
                "SELL",
                "卖出 (假底止损)",
                "sell",
                f"触发 SL_W25%: 周收盘 {wc_live:.2f} < 买入价75% ({avg_entry_px * 0.75:.2f})",
            )
        elif i_end > entry_fill_pos:
            if (
                armed_live
                and pd.notna(w_live["W_Fisher"])
                and pd.notna(w_live["W_Fisher_Sig"])
                and float(w_live["W_Fisher"]) < float(w_live["W_Fisher_Sig"])
            ):
                if tp_split_lottery_pct > 0.0:
                    live_act, live_badge, live_tone, live_reason = (
                        "SELL",
                        f"卖出 (TP3h: 开盘50%+挂+{int(tp_split_lottery_pct*100)}%)",
                        "sell",
                        f"高位区已激活且周 Fisher ({float(w_live['W_Fisher']):.2f}) < Sig ({float(w_live['W_Fisher_Sig']):.2f})：下个交易日开盘卖50%，剩50%挂+{int(tp_split_lottery_pct*100)}%限价（5日兜底）",
                    )
                else:
                    live_act, live_badge, live_tone, live_reason = (
                        "SELL",
                        "卖出 (TP3h高位止盈)",
                        "sell",
                        f"高位区已激活且周 Fisher ({float(w_live['W_Fisher']):.2f}) < Signal ({float(w_live['W_Fisher_Sig']):.2f})",
                    )
            elif pd.notna(mf_live) and float(mf_live) >= 2.0:
                if pd.notna(vs30_live) and float(vs30_live) < 0.0:
                    live_act, live_badge, live_tone, live_reason = (
                        "SELL",
                        "卖出 (破30W兜底逃顶)",
                        "sell",
                        f"M_Fish≥2.0 ({float(mf_live):.2f}) 且周收盘跌破 30周SMA",
                    )
                elif cur_peak_w > 0.0 and wc_live <= cur_peak_w * 0.65:
                    live_act, live_badge, live_tone, live_reason = (
                        "SELL",
                        "卖出 (周回撤35%逃顶)",
                        "sell",
                        f"M_Fish≥2.0 ({float(mf_live):.2f}) 且自峰值 {cur_peak_w:.2f} 回撤≥35%",
                    )
        live_ret = round((closes[i_end] / (avg_entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0, 1)
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": f"持仓 {'50% (TP3h挂单止盈中)' if lot_active else '100%'} ({mode_tag})",
            "pos_mode": "BOT",
            "entry_date": entry_date_str,
            "entry_px": round(avg_entry_px, 2),
            "ret_pct": live_ret,
            "hold_weeks": hold_w_live,
            "high_zone_armed": armed_live,
            "sl_price": round(avg_entry_px * 0.75, 2) if unmatured_live else None,
            "wtsl_price": round(cur_peak_w * 0.65, 2) if cur_peak_w > 0.0 else None,
            "cooldown_days": 0,
            "reason": live_reason,
        }
    else:
        live_act, live_badge, live_tone, live_reason = (
            "WAIT",
            "保持空仓",
            "wait",
            f"冷静期剩余 {cooldown_rem} 个交易日" if cooldown_rem > 0 else "空仓观望，等待大底或月线金叉信号",
        )
        if not just_exited and i_end >= cooldown_until_pos and bool(w_live[entry_col]):
            live_act, live_badge, live_tone, live_reason = (
                "BUY",
                "买入 100% (触发建仓信号)",
                "buy",
                f"满足 {entry_col} 入场信号，下个交易日开盘全仓买入",
            )
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": "空仓 (0%)",
            "pos_mode": "FLAT",
            "entry_date": None,
            "entry_px": None,
            "ret_pct": None,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": None,
            "wtsl_price": None,
            "cooldown_days": cooldown_rem,
            "reason": live_reason,
        }

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    if strategy_id == "C2_F1_or_MFishLin_TP3h":
        return {
            "id": "C2_F1_or_MFishLin_TP3h",
            "name": "SOXL 高容错旗舰: C2+Deep85_HL+TP3h(50%+10%) (12年$1.61亿 / 15.5年$2.89亿)",
            "short_label": "SOXL高容错旗舰 (C2+Deep85_HL+TP3h分批)",
            "rule_desc": "三入口: F1主入口 | Deep85+周HigherLow | 月Fisher<0.8金叉兜底 || 出口: TP3h触发开盘卖50%+剩50%挂+10%(5日兜底) (X1a+SL_W25兜底)",
            "stats": stats,
            "trades": trades,
            "live_status": live_status,
        }
    if use_f1_filter:
        return {
            "id": "F1_TP3h",
            "name": "SOXL 极致大周期止盈: F1+TP3h (5战全胜)",
            "short_label": "SOXL极致版 (F1+TP3h)",
            "rule_desc": "pre_c7_vol & (wks_52H≥30 或 8w低跌≤-80%) & PrevHigh买 | 持满50w&M_Fish≥2&乖离30W≥40%后周Fisher死叉止盈 (X1a兜底)",
            "stats": stats,
            "trades": trades,
            "live_status": live_status,
        }
    return {
        "id": "F0_TP3h",
        "name": "通用大周期主动止盈: F0+TP3h (跨标的稳健)",
        "short_label": "通用止盈版 (F0+TP3h)",
        "rule_desc": "pre_c7_vol & PrevHigh买 + SL_W25% | 持满50w&M_Fish≥2&乖离30W≥40%后周Fisher死叉止盈 (X1a兜底)",
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
    }


def detect_c7_vol_entries(
    df_daily: pd.DataFrame,
    start_date: str = "2014-01-01",
    cluster_weeks: int = 8,
) -> list[dict]:
    """Detect causal C7_Vol major-bottom entries on weekly bars, executed at next day's Open."""
    daily = df_daily.set_index("Date")
    w = _agg_periodic(daily, "W-FRI")

    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_EMA10"] = ema(w["Close"], 10)
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()

    min_low_200_8w = (w["Low"] / w["W_SMA200"]).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    max_vol_4w = w["W_Vol_R"].rolling(4, min_periods=1).max()

    cond = (
        (min_low_200_8w <= 1.10)
        & (min_wf_4w < -1.5)
        & (max_vol_4w >= 1.6)
        & (w["Close"] > w["W_EMA10"])
    )

    w_dates = w.index
    d_dates = daily.index
    opens = daily["Open"].to_numpy(dtype=np.float64)
    lows = daily["Low"].to_numpy(dtype=np.float64)
    closes = daily["Close"].to_numpy(dtype=np.float64)
    n_d = len(daily)

    valid_mask = (w_dates >= start_date) & cond.fillna(False).to_numpy(dtype=bool)
    sig_positions = np.where(valid_mask)[0]

    entries: list[dict] = []
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
        entry_px = float(opens[entry_idx])

        w6_end = min(n_d, entry_idx + 126)
        mae_6m = (float(np.min(lows[entry_idx:w6_end])) / entry_px - 1.0) * 100.0

        def fwd_ret(bars: int) -> float | None:
            if entry_idx + bars >= n_d:
                return None
            return round((float(closes[entry_idx + bars]) / entry_px - 1.0) * 100.0, 1)

        entries.append(
            {
                "sig_week": w_dt.strftime("%Y-%m-%d"),
                "entry_date": entry_dt.strftime("%Y-%m-%d"),
                "entry_m": entry_dt.strftime("%Y-%m"),
                "entry_px": round(entry_px, 2),
                "max4w_vol_r": round(float(max_vol_4w.iloc[pos]), 2),
                "mae_6m": round(mae_6m, 1),
                "fwd_3m": fwd_ret(63),
                "fwd_6m": fwd_ret(126),
                "fwd_12m": fwd_ret(252),
            }
        )
    return entries


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SOXL 月线 LOG — 顶部与底部区域及最优策略标注</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-bg: rgba(88, 166, 255, 0.14);
      --green: #26a69a;
      --red: #ef5350;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 12px;
      padding: 10px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
    }
    .title-badge {
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      margin-right: 4px;
    }
    .group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .group-label {
      color: var(--muted);
      font-weight: 600;
    }
    input[type="number"] {
      width: 62px;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 4px 6px;
      font-size: 12px;
      text-align: right;
    }
    .strategy-select {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .toggle-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--bg);
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      font-size: 12px;
    }
    .toggle-pill input { display: none; }
    .toggle-pill.checked {
      border-color: var(--accent);
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
    }
    .divider {
      width: 1px;
      height: 20px;
      background: var(--border);
    }
    .live-cockpit {
      margin: 12px 20px 6px;
      padding: 14px 18px;
      background: linear-gradient(135deg, rgba(22, 27, 34, 0.96), rgba(15, 21, 30, 0.98));
      border: 1px solid var(--border);
      border-radius: 10px;
      display: grid;
      grid-template-columns: minmax(290px, 1.05fr) minmax(360px, 1.45fr);
      gap: 16px;
      align-items: stretch;
    }
    @media (max-width: 960px) {
      .live-cockpit {
        grid-template-columns: 1fr;
      }
    }
    .cockpit-main {
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 12px;
    }
    .cockpit-meta {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      font-size: 13.5px;
      color: var(--text);
    }
    .fresh-pill {
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(88, 166, 255, 0.16);
      color: #79c0ff;
      border: 1px solid rgba(88, 166, 255, 0.40);
      font-weight: 700;
      font-size: 13px;
    }
    .warn-pill {
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(210, 153, 34, 0.20);
      color: #f0b840;
      border: 1px solid rgba(210, 153, 34, 0.45);
      font-weight: 700;
      font-size: 12.5px;
    }
    .action-box {
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 16px 18px;
      border-radius: 10px;
      border: 1px solid var(--border);
      border-left-width: 5px;
      background: rgba(13, 17, 23, 0.78);
    }
    .action-box.tone-buy {
      border-color: #3fb950;
      background: rgba(46, 160, 67, 0.18);
    }
    .action-box.tone-sell {
      border-color: #f85149;
      background: rgba(248, 81, 73, 0.18);
    }
    .action-box.tone-hold {
      border-color: #58a6ff;
      background: rgba(56, 139, 253, 0.16);
    }
    .action-box.tone-wait {
      border-color: #d29922;
      background: rgba(210, 153, 34, 0.13);
    }
    .action-badge {
      font-size: 22px;
      font-weight: 800;
      padding: 10px 16px;
      border-radius: 8px;
      white-space: nowrap;
      letter-spacing: 0.4px;
    }
    .tone-buy .action-badge {
      background: #238636;
      color: #ffffff;
      box-shadow: 0 0 14px rgba(63, 185, 80, 0.42);
    }
    .tone-sell .action-badge {
      background: #da3633;
      color: #ffffff;
      box-shadow: 0 0 14px rgba(248, 81, 73, 0.42);
    }
    .tone-hold .action-badge {
      background: #1f6feb;
      color: #ffffff;
      box-shadow: 0 0 12px rgba(88, 166, 255, 0.35);
    }
    .tone-wait .action-badge {
      background: #9e6a03;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(210, 153, 34, 0.28);
    }
    .action-detail {
      display: flex;
      flex-direction: column;
      gap: 5px;
      min-width: 0;
    }
    .action-pos {
      font-size: 15px;
      font-weight: 700;
      color: #f0f6fc;
    }
    .action-reason {
      font-size: 13.5px;
      color: #c9d1d9;
      line-height: 1.5;
    }
    .cockpit-holding-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      font-size: 13.5px;
      color: #c9d1d9;
    }
    .cockpit-holding-strip b { color: #f0f6fc; }
    .cockpit-radar {
      display: flex;
      flex-direction: column;
      gap: 10px;
      border-left: 1px solid var(--border);
      padding-left: 16px;
    }
    @media (max-width: 960px) {
      .cockpit-radar {
        border-left: none;
        border-top: 1px solid var(--border);
        padding-left: 0;
        padding-top: 12px;
      }
    }
    .radar-title {
      font-size: 12.5px;
      font-weight: 600;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .radar-groups {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .radar-group {
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid rgba(48, 54, 61, 0.9);
      background: rgba(13, 17, 23, 0.45);
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .radar-group-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 700;
      color: #c9d1d9;
    }
    .radar-group-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 1px 8px;
      border-radius: 999px;
      background: rgba(139, 148, 158, 0.16);
      color: var(--muted);
      white-space: nowrap;
    }
    .radar-group-tag.ready {
      background: rgba(63, 185, 80, 0.18);
      color: var(--green);
    }
    .radar-group-tag.alert {
      background: rgba(248, 81, 73, 0.18);
      color: var(--red);
    }
    .radar-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
      gap: 8px;
    }
    .radar-item {
      padding: 7px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: rgba(13, 17, 23, 0.72);
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .radar-item.ok {
      border-color: rgba(63, 185, 80, 0.45);
      background: rgba(63, 185, 80, 0.08);
    }
    .radar-item.warn {
      border-color: rgba(210, 153, 34, 0.45);
      background: rgba(210, 153, 34, 0.08);
    }
    .radar-item-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      color: var(--text);
    }
    .radar-flag {
      white-space: nowrap;
      flex-shrink: 0;
    }
    .radar-item.ok .radar-flag { color: var(--green); }
    .radar-item.warn .radar-flag { color: var(--amber); }
    .radar-item.miss .radar-flag { color: var(--muted); }
    .radar-item-val {
      font-size: 11px;
      color: var(--muted);
    }
    #chart {
      width: 100%;
      height: 68vh;
      min-height: 460px;
    }
    .table-wrap {
      padding: 12px 16px 24px;
      background: var(--bg);
      border-top: 1px solid var(--border);
      overflow-x: auto;
    }
    .table-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 6px 10px;
      border-bottom: 1px solid #21262d;
      text-align: left;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: var(--panel);
    }
    .tag-top {
      color: var(--red);
      font-weight: 600;
    }
    .tag-bot {
      color: var(--green);
      font-weight: 600;
    }
    .tag-warn {
      color: #f0b90b;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <span class="title-badge">SOXL 月线顶底区域</span>

    <label class="toggle-pill checked" id="logLabel">
      <input type="checkbox" id="logToggle" checked />
      <span>LOG 对数坐标</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showZigZag" checked />
      <span>ZigZag 连线</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showSma" checked />
      <span>30周 / 200周均线</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showFisher" checked />
      <span>月线 Fisher(10)</span>
    </label>

    <label class="toggle-pill">
      <input type="checkbox" class="opt-check" id="showRsi" />
      <span>月线 RSI(14)</span>
    </label>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">策略标注</span>
      <select id="strategySelect" class="strategy-select">
        <option value="C2_F1_or_MFishLin_TP3h" selected>SOXL高容错旗舰: C2+Deep85_HL+TP3h(50%+10%) (12年$1.61亿 / 15.5年$2.89亿, 0踏空)</option>
        <option value="F1_TP3h">SOXL极致大周期: F1+TP3h (5笔全胜, 12年$9931万, 回撤-49.0%)</option>
        <option value="F0_TP3h">通用大周期止盈: F0+TP3h (6笔, 12年$6681万, 回撤-49.0%)</option>
        <option value="D1_X1a_WTSL35">策略3 极简大周期: D1_X1a_WTSL35 (6笔, 12年$3804万)</option>
        <option value="D1_X1a_WTSL35_Split50">策略3 分批建仓版: D1_X1a_WTSL35_Split50 (6笔, 12年$3012万, 回撤-54.4%)</option>
        <option value="H5_E2_Promote_S1_AdaptHV">策略4 双引擎旗舰: H5_E2_Promote_S1_AdaptHV (32笔, 12年$3791万, 回撤-52.0%)</option>
        <option value="C7_Vol">仅大底入口: C7_Vol (5次, 0误报)</option>
        <option value="NONE">不显示策略标注</option>
      </select>
    </div>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">上涨确认(%)</span>
      <input type="number" id="upPct" value="120" min="20" max="500" step="10" />
    </div>
    <div class="group">
      <span class="group-label">回撤确认(%)</span>
      <input type="number" id="downPct" value="55" min="15" max="90" step="5" />
    </div>
    <div class="group">
      <span class="group-label">区域容差(±月 / 价格%)</span>
      <input type="number" id="zoneMonths" value="2" min="0" max="5" step="1" />
      <input type="number" id="zoneTolPct" value="20" min="5" max="50" step="5" />
    </div>
  </div>

  <div id="liveCockpit" class="live-cockpit"></div>

  <div id="chart"></div>

  <div class="table-wrap">
    <div class="table-title" id="summaryTitle">识别到的月线顶部与底部区域</div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>类型</th>
          <th>拐点月份</th>
          <th>区域范围</th>
          <th>极值价格 (前复权)</th>
          <th>相对上一拐点涨跌幅</th>
        </tr>
      </thead>
      <tbody id="zoneTableBody"></tbody>
    </table>

    <div id="strategyTableWrap">
      <div class="table-title" id="entryTableTitle" style="margin-top:20px;">策略交易明细</div>
      <table id="entryTable">
        <thead id="entryTableHead"></thead>
        <tbody id="entryTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const MONTHLY = __MONTHLY_JSON__;
    const WEEKLY_SMA = __WEEKLY_SMA_JSON__;
    const C7_ENTRIES = __C7_ENTRIES_JSON__;
    const STRATEGIES = __STRATEGIES_JSON__;
    const LIVE_MARKET = __LIVE_MARKET_JSON__;

    function detectZigZag(bars, upTh, downTh) {
      const pivots = [];
      let trend = null;
      let lastHIdx = 0, lastH = bars[0].h;
      let lastLIdx = 0, lastL = bars[0].l;

      for (let i = 1; i < bars.length; i++) {
        const h = bars[i].h;
        const l = bars[i].l;

        if (trend === null) {
          if (h >= lastL * (1 + upTh)) {
            pivots.push({ idx: lastLIdx, type: "L", price: lastL });
            trend = "UP";
            lastHIdx = i;
            lastH = h;
          } else if (l <= lastH * (1 - downTh)) {
            pivots.push({ idx: lastHIdx, type: "H", price: lastH });
            trend = "DOWN";
            lastLIdx = i;
            lastL = l;
          } else {
            if (h > lastH) { lastHIdx = i; lastH = h; }
            if (l < lastL) { lastLIdx = i; lastL = l; }
          }
        } else if (trend === "UP") {
          if (h >= lastH) {
            lastHIdx = i;
            lastH = h;
          } else if (l <= lastH * (1 - downTh)) {
            pivots.push({ idx: lastHIdx, type: "H", price: lastH });
            trend = "DOWN";
            lastLIdx = i;
            lastL = l;
          }
        } else if (trend === "DOWN") {
          if (l <= lastL) {
            lastLIdx = i;
            lastL = l;
          } else if (h >= lastL * (1 + upTh)) {
            pivots.push({ idx: lastLIdx, type: "L", price: lastL });
            trend = "UP";
            lastHIdx = i;
            lastH = h;
          }
        }
      }
      return pivots;
    }

    function expandZones(bars, pivots, maxOffset, tolRatio) {
      return pivots.map((p, k) => {
        let start = p.idx;
        let end = p.idx;
        if (p.type === "H") {
          const minPrice = p.price * (1 - tolRatio);
          for (let i = p.idx - 1; i >= Math.max(0, p.idx - maxOffset); i--) {
            if (bars[i].h >= minPrice || bars[i].c >= minPrice) start = i;
            else break;
          }
          for (let i = p.idx + 1; i <= Math.min(bars.length - 1, p.idx + maxOffset); i++) {
            if (bars[i].h >= minPrice || bars[i].c >= minPrice) end = i;
            else break;
          }
        } else {
          const maxPrice = p.price * (1 + tolRatio);
          for (let i = p.idx - 1; i >= Math.max(0, p.idx - maxOffset); i--) {
            if (bars[i].l <= maxPrice || bars[i].c <= maxPrice) start = i;
            else break;
          }
          for (let i = p.idx + 1; i <= Math.min(bars.length - 1, p.idx + maxOffset); i++) {
            if (bars[i].l <= maxPrice || bars[i].c <= maxPrice) end = i;
            else break;
          }
        }
        const prev = k > 0 ? pivots[k - 1] : null;
        const pctChange = prev ? ((p.price - prev.price) / prev.price) * 100 : null;
        return {
          ...p,
          startIdx: start,
          endIdx: end,
          startMonth: bars[start].m,
          endMonth: bars[end].m,
          pivotMonth: bars[p.idx].m,
          pctChange,
        };
      });
    }

    function calcFisher(highs, lows, period = 10) {
      const n = highs.length;
      const fisher = new Array(n).fill(null);
      const signal = new Array(n).fill(null);
      const med = highs.map((h, i) => (h + lows[i]) / 2);
      let prevX = 0, prevF = 0;
      for (let i = period - 1; i < n; i++) {
        let minV = Infinity, maxV = -Infinity;
        for (let j = i - period + 1; j <= i; j++) {
          if (med[j] < minV) minV = med[j];
          if (med[j] > maxV) maxV = med[j];
        }
        const span = maxV - minV;
        const norm = span === 0 ? 0 : (med[i] - minV) / span - 0.5;
        let x = 0.33 * 2 * norm + 0.67 * prevX;
        if (x > 0.999) x = 0.999;
        if (x < -0.999) x = -0.999;
        const f = 0.5 * Math.log((1 + x) / (1 - x)) + 0.5 * prevF;
        fisher[i] = f;
        signal[i] = prevF;
        prevX = x;
        prevF = f;
      }
      return { fisher, signal };
    }

    function calcRSI(arr, period = 14) {
      const out = new Array(arr.length).fill(null);
      const alpha = 1 / period;
      let avgGain = 0, avgLoss = 0;
      for (let i = 1; i < arr.length; i++) {
        const delta = arr[i] - arr[i - 1];
        const gain = delta > 0 ? delta : 0;
        const loss = delta < 0 ? -delta : 0;
        if (i === 1) { avgGain = gain; avgLoss = loss; }
        else {
          avgGain = alpha * gain + (1 - alpha) * avgGain;
          avgLoss = alpha * loss + (1 - alpha) * avgLoss;
        }
        if (i >= period) {
          out[i] = avgLoss === 0 ? (avgGain > 0 ? 100 : 50) : 100 - 100 / (1 + avgGain / avgLoss);
        }
      }
      return out;
    }

    function fmtWan(val) {
      if (val >= 100000000) return `$${(val / 100000000).toFixed(2)}亿`;
      if (val >= 10000) return `$${(val / 10000).toFixed(1)}万`;
      return `$${val.toFixed(2)}`;
    }

    function addStrategyTraces(traces, stratId) {
      if (stratId === "NONE") return;

      if (stratId === "C7_Vol" && C7_ENTRIES.length > 0) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "大底入口 (C7_Vol 次日开盘)",
          x: C7_ENTRIES.map(e => e.entry_m),
          y: C7_ENTRIES.map(e => e.entry_px),
          text: C7_ENTRIES.map(e => {
            const f6 = e.fwd_6m === null ? "—" : `${e.fwd_6m >= 0 ? "+" : ""}${e.fwd_6m.toFixed(1)}%`;
            const f12 = e.fwd_12m === null ? "—" : `${e.fwd_12m >= 0 ? "+" : ""}${e.fwd_12m.toFixed(1)}%`;
            return `大底入口 [C7_Vol]<br>信号周: ${e.sig_week}<br>买入日: ${e.entry_date} (开盘 $${e.entry_px.toFixed(2)})<br>近4周峰值量: ${e.max4w_vol_r.toFixed(2)}x<br>6月最大再跌: ${e.mae_6m.toFixed(1)}%<br>6个月收益: ${f6} | 12个月收益: ${f12}`;
          }),
          hoverinfo: "text",
          marker: { symbol: "triangle-up", size: 13, color: "#00e676", line: { color: "#0d1117", width: 1.5 } },
          yaxis: "y",
        });
        return;
      }

      const strat = STRATEGIES[stratId];
      if (!strat || !strat.trades || strat.trades.length === 0) return;

      const botBuys = [];
      const split2Buys = [];
      const trendBuys = [];
      const altBuys = [];
      const normalSells = [];
      const slSells = [];

      strat.trades.forEach((t, idx) => {
        const num = idx + 1;
        const retStr = `${t.ret_pct >= 0 ? "+" : ""}${t.ret_pct.toFixed(1)}%`;
        const avgInfo = t.tranche2_m ? ` | 均价 $${t.avg_entry_px.toFixed(2)}` : "";
        const entryTip = `#${num} 买入 [${t.mode}]<br>买入日: ${t.entry_date} (开盘 $${t.entry_px.toFixed(2)}${avgInfo})<br>对应卖出: ${t.exit_date} @ $${t.exit_px.toFixed(2)}<br>整笔收益: ${retStr} (${t.exit_reason})`;

        if (t.mode.startsWith("BOT")) {
          botBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        } else if (t.mode === "TREND") {
          trendBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        } else if (t.mode === "ALT") {
          altBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        } else {
          botBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        }

        if (t.tranche2_m && t.tranche2_px !== null) {
          split2Buys.push({
            x: t.tranche2_m,
            y: t.tranche2_px,
            text: `#${num} 二段补仓 (余50%仓)<br>补仓日: ${t.tranche2_date} (开盘 $${t.tranche2_px.toFixed(2)})<br>持仓均价: $${t.avg_entry_px.toFixed(2)}<br>整笔收益: ${retStr} (${t.exit_reason})`,
          });
        }

        if (!t.is_open) {
          const sellTip = `#${num} 卖出 [${t.mode}]<br>卖出日: ${t.exit_date} (开盘 $${t.exit_px.toFixed(2)})<br>入场: ${t.entry_date} (均价 $${t.avg_entry_px.toFixed(2)})<br>单笔收益: ${retStr}<br>原因: ${t.exit_reason}`;
          if (t.exit_reason.includes("SL")) {
            slSells.push({ x: t.exit_m, y: t.exit_px, text: sellTip });
          } else {
            normalSells.push({ x: t.exit_m, y: t.exit_px, text: sellTip });
          }
        }
      });

      const pushScatter = (pts, name, symbol, color, size) => {
        if (pts.length === 0) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          name,
          x: pts.map(p => p.x),
          y: pts.map(p => p.y),
          text: pts.map(p => p.text),
          hoverinfo: "text",
          marker: { symbol, size, color, line: { color: "#0d1117", width: 1.5 } },
          yaxis: "y",
        });
      };

      pushScatter(botBuys, stratId.includes("Split50") ? "大底首仓买入 (50% BOT)" : "大底买入 (BOT)", "triangle-up", "#00e676", 13);
      pushScatter(split2Buys, "二段补仓 (EMA10 补满50%)", "diamond", "#38bdf8", 11);
      pushScatter(trendBuys, "趋势买入 (TREND)", "triangle-up", "#22c55e", 10);
      pushScatter(altBuys, "恐慌底买入 (ALT)", "triangle-up", "#facc15", 10);
      pushScatter(normalSells, "大顶/趋势/止盈卖出", "triangle-down", "#ff5252", 12);
      pushScatter(slSells, "止损卖出 (SL)", "x", "#fb923c", 11);
    }

    function renderStrategyTable(stratId) {
      const wrap = document.getElementById("strategyTableWrap");
      const titleEl = document.getElementById("entryTableTitle");
      const headEl = document.getElementById("entryTableHead");
      const bodyEl = document.getElementById("entryTableBody");
      const fmtPct = v => (v === null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`);

      if (stratId === "NONE") {
        wrap.style.display = "none";
        return;
      }
      wrap.style.display = "block";

      if (stratId === "C7_Vol") {
        titleEl.textContent = `大底入口明细 [C7_Vol]（近8周Low≤1.10×200W均线 & 近4周Fisher<-1.5 & 近4周峰值量≥1.6x & 站上10周EMA，共 ${C7_ENTRIES.length} 次）`;
        headEl.innerHTML = `<tr>
          <th>#</th><th>信号周</th><th>买入日 (次日开盘)</th><th>开盘价 ($)</th>
          <th>近4周峰值量</th><th>6月最大再跌 (MAE)</th><th>3个月收益</th><th>6个月收益</th><th>12个月收益</th>
        </tr>`;
        bodyEl.innerHTML = C7_ENTRIES.map((e, i) => `<tr>
          <td>${i + 1}</td><td>${e.sig_week}</td><td>${e.entry_date}</td><td>${e.entry_px.toFixed(2)}</td>
          <td>${e.max4w_vol_r.toFixed(2)}x</td><td class="tag-top">${e.mae_6m.toFixed(1)}%</td>
          <td class="${e.fwd_3m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_3m)}</td>
          <td class="${e.fwd_6m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_6m)}</td>
          <td class="${e.fwd_12m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_12m)}</td>
        </tr>`).join("");
        return;
      }

      const strat = STRATEGIES[stratId];
      const st = strat.stats;
      titleEl.textContent = `${strat.name}（2015–2026：$1万 → ${fmtWan(st.final)}，CAGR ${st.cagr.toFixed(2)}%，MaxDD ${st.max_dd.toFixed(2)}%，Calmar ${st.calmar.toFixed(3)}，${st.trades} 笔，胜率 ${st.win_rate.toFixed(1)}% ｜ 规则：${strat.rule_desc}）`;
      headEl.innerHTML = `<tr>
        <th>#</th><th>仓位类型</th><th>首次买入日</th><th>补仓日</th><th>买入价 ($)</th>
        <th>持仓均价 ($)</th><th>卖出日</th><th>卖出价 ($)</th><th>单笔收益</th><th>离场原因</th>
      </tr>`;
      bodyEl.innerHTML = strat.trades.map((t, i) => {
        const retCls = t.ret_pct >= 0 ? "tag-bot" : "tag-top";
        const modeCls = t.mode.startsWith("BOT") ? "tag-bot" : (t.mode === "ALT" ? "tag-warn" : "");
        return `<tr>
          <td>${i + 1}</td>
          <td class="${modeCls}">${t.mode}</td>
          <td>${t.entry_date}</td>
          <td>${t.tranche2_date}</td>
          <td>${t.entry_px.toFixed(2)}</td>
          <td>${t.avg_entry_px.toFixed(2)}</td>
          <td>${t.exit_date}</td>
          <td>${t.exit_px.toFixed(2)}</td>
          <td class="${retCls}">${fmtPct(t.ret_pct)}</td>
          <td>${t.exit_reason}</td>
        </tr>`;
      }).join("");
    }

    function render() {
      const isLog = document.getElementById("logToggle").checked;
      const showZigZag = document.getElementById("showZigZag").checked;
      const showSma = document.getElementById("showSma").checked;
      const showFisher = document.getElementById("showFisher").checked;
      const showRsi = document.getElementById("showRsi").checked;
      const stratId = document.getElementById("strategySelect").value;

      const upTh = Math.max(0.1, Number(document.getElementById("upPct").value) / 100);
      const downTh = Math.min(0.95, Math.max(0.1, Number(document.getElementById("downPct").value) / 100));
      const zoneMonths = Math.max(0, Number(document.getElementById("zoneMonths").value));
      const zoneTol = Math.max(0.01, Number(document.getElementById("zoneTolPct").value) / 100);

      const dates = MONTHLY.map(b => b.m);
      const opens = MONTHLY.map(b => b.o);
      const highs = MONTHLY.map(b => b.h);
      const lows = MONTHLY.map(b => b.l);
      const closes = MONTHLY.map(b => b.c);

      const pivots = detectZigZag(MONTHLY, upTh, downTh);
      const zones = expandZones(MONTHLY, pivots, zoneMonths, zoneTol);

      const subPanels = [];
      if (showFisher) subPanels.push("fisher");
      if (showRsi) subPanels.push("rsi");
      const subH = subPanels.length === 0 ? 0 : Math.min(0.20, 0.38 / subPanels.length);
      const gap = 0.04;
      const mainBottom = subPanels.length * (subH + gap);

      const traces = [];
      const shapes = [];
      const annotations = [];

      zones.forEach(z => {
        const isTop = z.type === "H";
        shapes.push({
          type: "rect",
          xref: "x",
          x0: z.startIdx - 0.45,
          x1: z.endIdx + 0.45,
          yref: "paper",
          y0: mainBottom,
          y1: 1,
          fillcolor: isTop ? "rgba(239, 83, 80, 0.16)" : "rgba(38, 166, 154, 0.16)",
          line: {
            color: isTop ? "rgba(239, 83, 80, 0.45)" : "rgba(38, 166, 154, 0.45)",
            width: 1,
          },
          layer: "below",
        });

        annotations.push({
          xref: "x",
          x: z.pivotMonth,
          yref: "y",
          y: isLog ? Math.log10(z.price) : z.price,
          text: isTop ? `▼ 顶 ${z.price.toFixed(2)}` : `▲ 底 ${z.price.toFixed(2)}`,
          showarrow: true,
          arrowhead: 0,
          arrowcolor: isTop ? "#ef5350" : "#26a69a",
          ax: 0,
          ay: isTop ? -22 : 22,
          font: { color: isTop ? "#ff7b72" : "#3fb950", size: 10, weight: 700 },
          bgcolor: "rgba(13,17,23,0.75)",
        });
      });

      traces.push({
        type: "candlestick",
        name: "SOXL 月线",
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: { line: { color: "#26a69a" }, fillcolor: "#26a69a" },
        decreasing: { line: { color: "#ef5350" }, fillcolor: "#ef5350" },
        yaxis: "y",
      });

      if (showZigZag && zones.length > 0) {
        traces.push({
          type: "scatter",
          mode: "lines+markers",
          name: "ZigZag 拐点",
          x: zones.map(z => z.pivotMonth),
          y: zones.map(z => z.price),
          line: { width: 1.5, color: "#f0b90b", dash: "dot" },
          marker: { size: 7, color: zones.map(z => (z.type === "H" ? "#ef5350" : "#26a69a")) },
          yaxis: "y",
        });
      }

      if (showSma) {
        const smaMap = new Map(WEEKLY_SMA.map(w => [w.m, w]));
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "30周均线",
          x: dates,
          y: dates.map(d => (smaMap.get(d) ? smaMap.get(d).s30 : null)),
          line: { width: 1.4, color: "#58a6ff" },
          yaxis: "y",
        });
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "200周均线",
          x: dates,
          y: dates.map(d => (smaMap.get(d) ? smaMap.get(d).s200 : null)),
          line: { width: 1.5, color: "#d2a8ff" },
          yaxis: "y",
        });
      }

      addStrategyTraces(traces, stratId);

      const layout = {
        paper_bgcolor: "#0d1117",
        plot_bgcolor: "#0d1117",
        font: { color: "#e6edf3", size: 11 },
        margin: { l: 64, r: 24, t: 36, b: 36 },
        hovermode: "x unified",
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.04, bgcolor: "rgba(0,0,0,0)" },
        xaxis: {
          type: "category",
          rangeslider: { visible: false },
          gridcolor: "#21262d",
          nticks: 16,
        },
        yaxis: {
          domain: [mainBottom, 1],
          type: isLog ? "log" : "linear",
          gridcolor: "#21262d",
          title: isLog ? "SOXL 价格 (LOG)" : "SOXL 价格",
          autorange: true,
        },
        shapes,
        annotations,
      };

      subPanels.forEach((panel, idx) => {
        const axisNum = idx + 2;
        const yName = `yaxis${axisNum}`;
        const yRef = `y${axisNum}`;
        const top = mainBottom - gap - idx * (subH + gap);
        const bottom = Math.max(0, top - subH);

        if (panel === "fisher") {
          layout[yName] = { domain: [bottom, top], gridcolor: "#21262d", title: "月线 Fisher" };
          const f = calcFisher(highs, lows, 10);
          traces.push({
            type: "scatter", mode: "lines", name: "Fisher(10)",
            x: dates, y: f.fisher, line: { width: 1.4, color: "#2dd4bf" }, yaxis: yRef,
          });
          traces.push({
            type: "scatter", mode: "lines", name: "Fisher Signal",
            x: dates, y: f.signal, line: { width: 1.2, color: "#f97316", dash: "dot" }, yaxis: yRef,
          });
          layout.shapes.push(
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 0, y1: 0, line: { color: "#484f58", width: 1, dash: "dash" } }
          );
        } else if (panel === "rsi") {
          layout[yName] = { domain: [bottom, top], gridcolor: "#21262d", title: "月线 RSI(14)", range: [0, 100] };
          traces.push({
            type: "scatter", mode: "lines", name: "RSI(14)",
            x: dates, y: calcRSI(closes, 14), line: { width: 1.4, color: "#c084fc" }, yaxis: yRef,
          });
          layout.shapes.push(
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 70, y1: 70, line: { color: "#ef5350", width: 1, dash: "dot" } },
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 30, y1: 30, line: { color: "#26a69a", width: 1, dash: "dot" } }
          );
        }
      });

      Plotly.newPlot("chart", traces, layout, { responsive: true, scrollZoom: true });

      const tbody = document.getElementById("zoneTableBody");
      tbody.innerHTML = zones.map((z, i) => {
        const isTop = z.type === "H";
        const cls = isTop ? "tag-top" : "tag-bot";
        const label = isTop ? "顶部区域" : "底部区域";
        const pct = z.pctChange === null
          ? "—"
          : `${z.pctChange >= 0 ? "+" : ""}${z.pctChange.toFixed(1)}%`;
        return `<tr>
          <td>${i + 1}</td>
          <td class="${cls}">${label}</td>
          <td>${z.pivotMonth}</td>
          <td>${z.startMonth} ~ ${z.endMonth}</td>
          <td>${z.price.toFixed(2)}</td>
          <td class="${cls}">${pct}</td>
        </tr>`;
      }).join("");

      let stratInfo = "";
      if (stratId === "C7_Vol") {
        stratInfo = ` ｜ 大底入口 C7_Vol（共 ${C7_ENTRIES.length} 次触发，6月/12月胜率 100%）`;
      } else if (stratId !== "NONE" && STRATEGIES[stratId]) {
        const st = STRATEGIES[stratId].stats;
        stratInfo = ` ｜ ${STRATEGIES[stratId].short_label}（$1万→${fmtWan(st.final)}, CAGR ${st.cagr.toFixed(2)}%, MaxDD ${st.max_dd.toFixed(2)}%, ${st.trades}笔, 胜率 ${st.win_rate.toFixed(1)}%）`;
      }
      document.getElementById("summaryTitle").textContent =
        `共识别 ${zones.length} 个月线顶底区域（上涨 ≥ ${(upTh * 100).toFixed(0)}% / 回撤 ≥ ${(downTh * 100).toFixed(0)}%）${stratInfo}`;

      renderStrategyTable(stratId);
      renderLiveCockpit(stratId);
    }

    function fmtV(val, suffix = "", digits = 2) {
      if (val === null || val === undefined || Number.isNaN(val)) return "—";
      return `${Number(val).toFixed(digits)}${suffix}`;
    }

    function radarCell(title, ok, valueText, warn = false) {
      const cls = ok ? "ok" : (warn ? "warn" : "miss");
      const flag = ok ? "✓ 满足" : (warn ? "! 接近" : "○ 未满足");
      return {
        ok: Boolean(ok),
        html: `<div class="radar-item ${cls}">
          <div class="radar-item-head"><span>${title}</span><span class="radar-flag">${flag}</span></div>
          <div class="radar-item-val">${valueText}</div>
        </div>`,
      };
    }

    function renderRadarGroup(title, cells, mode = "all") {
      const done = cells.filter(c => c.ok).length;
      const total = cells.length;
      let tagText = `已满足 ${done}/${total}`;
      let tagCls = "";
      if (mode === "all") {
        if (done === total && total > 0) tagCls = "ready";
      } else if (mode === "any_exit") {
        tagText = done > 0 ? `已触发 ${done} 项` : `未触发 (0/${total})`;
        if (done > 0) tagCls = "alert";
      } else if (mode === "any_entry") {
        tagText = done > 0 ? `已触发 ${done} 项` : `未触发 (0/${total})`;
        if (done > 0) tagCls = "ready";
      }
      return `<div class="radar-group">
        <div class="radar-group-head">
          <span>${title}</span>
          <span class="radar-group-tag ${tagCls}">${tagText}</span>
        </div>
        <div class="radar-grid">${cells.map(c => c.html).join("")}</div>
      </div>`;
    }

    function renderLiveCockpit(stratId) {
      const box = document.getElementById("liveCockpit");
      if (!LIVE_MARKET) {
        box.style.display = "none";
        return;
      }
      box.style.display = "grid";
      const m = LIVE_MARKET;
      const st = STRATEGIES[stratId];
      const ls = st ? st.live_status : null;

      const freshHtml = `
        <span class="fresh-pill">最新交易日: ${m.last_date} (${m.weekday_cn})</span>
        ${m.is_friday ? "" : `<span class="warn-pill">非周五收盘（周中或节假日数据）</span>`}
        <span>SOXL <b>$${fmtV(m.soxl_close)}</b></span>
        <span>SMH <b>$${fmtV(m.smh_close)}</b> (200SMA $${fmtV(m.smh_sma200)})</span>
        <span>HV20 <b>${fmtV(m.hv20, "%", 1)}</b></span>
      `;

      let actionHtml = "";
      let holdingHtml = "";
      if (ls) {
        actionHtml = `
          <div class="action-box tone-${ls.tone}">
            <div class="action-badge">${ls.badge}</div>
            <div class="action-detail">
              <div class="action-pos">当前仓位: ${ls.position_label} ｜ 下个交易日开盘执行</div>
              <div class="action-reason">${ls.reason}</div>
            </div>
          </div>
        `;
        if (ls.entry_date) {
          const retCls = ls.ret_pct >= 0 ? "pos" : "neg";
          const retTxt = `${ls.ret_pct >= 0 ? "+" : ""}${ls.ret_pct.toFixed(1)}%`;
          holdingHtml = `<div class="cockpit-holding-strip">
            <span>入场日: <b>${ls.entry_date}</b></span>
            <span>成本均价: <b>$${ls.entry_px.toFixed(2)}</b></span>
            <span>当前浮盈: <b class="${retCls}">${retTxt}</b></span>
            ${ls.hold_weeks !== null ? `<span>已持仓: <b>${ls.hold_weeks} 周</b></span>` : ""}
            ${ls.sl_price !== null ? `<span>假底止损线: <b>$${ls.sl_price.toFixed(2)}</b></span>` : ""}
            ${ls.wtsl_price !== null ? `<span>周回撤35%线: <b>$${ls.wtsl_price.toFixed(2)}</b></span>` : ""}
          </div>`;
        }
      } else {
        actionHtml = `
          <div class="action-box tone-wait">
            <div class="action-badge">仅观察 / 信号叠加</div>
            <div class="action-detail">
              <div class="action-pos">当前未选定自动仓位策略 (${stratId})</div>
              <div class="action-reason">请在顶部策略下拉框选择 C2 / F1 / 策略3 / 策略4 查看完整买卖指令。</div>
            </div>
          </div>
        `;
      }

      const groups = [];
      if (stratId === "H5_E2_Promote_S1_AdaptHV" && ls) {
        const hvLim = (m.m_fish_prev === null || m.m_fish_prev < 1.5) ? 140 : 115;
        if (ls.pos_mode !== "FLAT") {
          const breakPx = m.smh_sma200 ? m.smh_sma200 * 0.99 : null;
          const g1 = [
            radarCell("趋势跌破: SMH < 200SMA×0.99", m.smh_close && breakPx ? m.smh_close < breakPx : false,
              `SMH $${fmtV(m.smh_close)} vs 跌破线 $${fmtV(breakPx)}`),
            radarCell(`波动率刹车: HV20 > ${hvLim}%`, m.hv20 !== null ? m.hv20 > hvLim : false,
              `当前 HV20 ${fmtV(m.hv20, "%", 1)} (M_Fish=${fmtV(m.m_fish_prev)})`),
          ];
          const g2 = [
            radarCell("假底止损 SL_W25 (仅BOT未成熟期)", ls.sl_price !== null && m.w_close !== null ? m.w_close < ls.sl_price : false,
              ls.sl_price !== null ? `W_Close $${fmtV(m.w_close)} vs 止损 $${fmtV(ls.sl_price)}` : "当前非BOT/ALT止损模式"),
            radarCell("月线热区 (M_Fish ≥ 1.5 切严格刹车)", m.m_fish_prev !== null ? m.m_fish_prev >= 1.5 : false,
              `当前 M_Fish_prev ${fmtV(m.m_fish_prev)}`, m.m_fish_prev !== null && m.m_fish_prev >= 1.0),
          ];
          groups.push(renderRadarGroup("出口规则 1 · 趋势 / 自适应波动率离场（任一触发即卖出）", g1, "any_exit"));
          groups.push(renderRadarGroup("出口规则 2 · 假底保护与热区状态", g2, "any_exit"));
        } else {
          const buyPx = m.smh_sma200 ? m.smh_sma200 * 1.01 : null;
          const g1 = [
            radarCell("SMH > 200SMA×1.01", m.smh_close && buyPx ? m.smh_close > buyPx : false,
              `SMH $${fmtV(m.smh_close)} vs 门槛 $${fmtV(buyPx)}`),
            radarCell("HV20 ≤ 105.8% (高波离场需≤95%)", m.hv20 !== null ? m.hv20 <= 105.8 : false,
              `当前 HV20 ${fmtV(m.hv20, "%", 1)}`),
          ];
          const g2 = [
            radarCell("BOT大底: pre_c7_vol + 站上周高", m.pre_c7_vol && m.w_close !== null && m.prev_w_high !== null && m.w_close > m.prev_w_high,
              `8W/200W=${fmtV(m.min8w_low_200w, "x", 2)} | 4W_WF=${fmtV(m.min4w_wf)} | 上周高=$${fmtV(m.prev_w_high)}`),
            radarCell("ALT恐慌: W_Fish < -1.8 & 拐头/放量", m.w_fisher !== null && m.w_fisher < -1.8 && ((m.dw_fish !== null && m.dw_fish > 0) || (m.w_vol_r !== null && m.w_vol_r > 1.3)),
              `W_Fish=${fmtV(m.w_fisher)} | 量比=${fmtV(m.w_vol_r, "x")}`),
          ];
          groups.push(renderRadarGroup("入口规则 1 · SMH 趋势仓 (TREND，满足全部 → 买入)", g1, "all"));
          groups.push(renderRadarGroup("入口规则 2 · 大底早买 (BOT) / 恐慌底 (ALT，任一 → 买入)", g2, "any_entry"));
        }
      } else if (ls && ls.pos_mode !== "FLAT") {
        const isTp3h = stratId.includes("TP3h");
        if (isTp3h) {
          const wfCrossDown = m.w_fisher !== null && m.w_fisher_sig !== null && m.w_fisher < m.w_fisher_sig;
          const g1 = [
            radarCell("持仓满 50 周", ls.hold_weeks !== null && ls.hold_weeks >= 50,
              `当前已持仓 ${ls.hold_weeks ?? 0} / 50 周`),
            radarCell("月线 M_Fish ≥ 2.0", m.m_fish_prev !== null && m.m_fish_prev >= 2.0,
              `当前 M_Fish_prev ${fmtV(m.m_fish_prev)}`, m.m_fish_prev !== null && m.m_fish_prev >= 1.5),
            radarCell("周收盘乖离30W ≥ +40%", m.close_vs_30w_pct !== null && m.close_vs_30w_pct >= 40.0,
              `当前乖离 ${fmtV(m.close_vs_30w_pct, "%", 1)} (30W=$${fmtV(m.w_sma30)})`),
            radarCell("高位激活 & 周Fisher死叉", ls.high_zone_armed && wfCrossDown,
              `${ls.high_zone_armed ? "高位区已激活" : "未激活"} | WF=${fmtV(m.w_fisher)} vs Sig=${fmtV(m.w_fisher_sig)}`, ls.high_zone_armed),
          ];
          const mfOver2 = m.m_fish_prev !== null && m.m_fish_prev >= 2.0;
          const g2 = [
            radarCell("假底止损 SL_W25%", ls.sl_price !== null && m.w_close !== null ? m.w_close < ls.sl_price : false,
              ls.sl_price !== null ? `W_Close $${fmtV(m.w_close)} < 止损 $${fmtV(ls.sl_price)}` : "已脱离未成熟期，SL_W25 失效"),
            radarCell("M_Fish≥2 后跌破 30周SMA", mfOver2 && m.w_close !== null && m.w_sma30 !== null && m.w_close < m.w_sma30,
              `W_Close $${fmtV(m.w_close)} vs 30W $${fmtV(m.w_sma30)}`),
            radarCell("M_Fish≥2 后周回撤 ≥ 35%", mfOver2 && ls.wtsl_price !== null && m.w_close !== null && m.w_close <= ls.wtsl_price,
              `W_Close $${fmtV(m.w_close)} vs 35%线 $${fmtV(ls.wtsl_price)}`),
          ];
          groups.push(renderRadarGroup("出口规则 1 · TP3h 主动高位止盈（前3项激活后，死叉 → 卖出）", g1, "all"));
          groups.push(renderRadarGroup("出口规则 2 · 假底止损 & X1a 兜底逃顶（任一触发 → 卖出）", g2, "any_exit"));
        } else {
          const mfOver2 = m.m_fish_prev !== null && m.m_fish_prev >= 2.0;
          const g1 = [
            radarCell("假底止损 SL_W25%", ls.sl_price !== null && m.w_close !== null ? m.w_close < ls.sl_price : false,
              ls.sl_price !== null ? `W_Close $${fmtV(m.w_close)} < 止损 $${fmtV(ls.sl_price)}` : "已脱离未成熟期，SL_W25 失效"),
          ];
          if (stratId === "D1_X1a_WTSL35_Split50" && ls.position_label.includes("50%")) {
            g1.push(radarCell("补仓: 周收盘 > 10周EMA", m.w_close !== null && m.w_ema10 !== null ? m.w_close > m.w_ema10 : false,
              `W_Close $${fmtV(m.w_close)} vs 10W_EMA $${fmtV(m.w_ema10)}`));
          }
          const g2 = [
            radarCell("M_Fish≥2 后跌破 30周SMA", mfOver2 && m.w_close !== null && m.w_sma30 !== null && m.w_close < m.w_sma30,
              `M_Fish=${fmtV(m.m_fish_prev)} | W_Close $${fmtV(m.w_close)} vs 30W $${fmtV(m.w_sma30)}`),
            radarCell("M_Fish≥2 后周回撤 ≥ 35%", mfOver2 && ls.wtsl_price !== null && m.w_close !== null && m.w_close <= ls.wtsl_price,
              `W_Close $${fmtV(m.w_close)} vs 35%线 $${fmtV(ls.wtsl_price)}`),
          ];
          groups.push(renderRadarGroup("出口规则 1 · 假底保护（未成熟期有效）", g1, "any_exit"));
          groups.push(renderRadarGroup("出口规则 2 · X1a 大顶逃顶（M_Fish≥2 后任一触发 → 卖出）", g2, "any_exit"));
        }
      } else {
        const useF1 = stratId === "C2_F1_or_MFishLin_TP3h" || stratId === "F1_TP3h";
        const useC2 = stratId === "C2_F1_or_MFishLin_TP3h";
        const isSplit = stratId === "D1_X1a_WTSL35_Split50";

        const g1 = [
          radarCell("200W极值: 8W Low/200W ≤ 1.10", m.min8w_low_200w !== null && m.min8w_low_200w <= 1.10,
            `当前 ${fmtV(m.min8w_low_200w, "x", 2)} (200W=$${fmtV(m.w_sma200)})`),
          radarCell("周恐慌: 4W min W_Fish < -1.5", m.min4w_wf !== null && m.min4w_wf < -1.5,
            `近4周最低 ${fmtV(m.min4w_wf)} (本周 ${fmtV(m.w_fisher)})`),
          radarCell("放量见底: 4W max 量比 ≥ 1.6x", m.max4w_vol_r !== null && m.max4w_vol_r >= 1.6,
            `近4周最高 ${fmtV(m.max4w_vol_r, "x")} (本周 ${fmtV(m.w_vol_r, "x")})`),
        ];
        if (useF1) {
          const f1Ok = (m.wks_52h !== null && m.wks_52h >= 30) || (m.dd52_8wl !== null && m.dd52_8wl <= -80.0);
          g1.push(radarCell("F1门控: 距52W高≥30w 或 跌≤-80%", f1Ok,
            `当前距高点 ${fmtV(m.wks_52h, "周", 0)} / 8W低回撤 ${fmtV(m.dd52_8wl, "%", 1)}`));
        }
        g1.push(radarCell("右侧确认: 周收盘 > 上周高点", m.w_close !== null && m.prev_w_high !== null && m.w_close > m.prev_w_high,
          `W_Close $${fmtV(m.w_close)} vs 上周高 $${fmtV(m.prev_w_high)}`));

        if (useC2) {
          const g_deep = [
            radarCell("前置大底: pre_c7_vol 满足", m.pre_c7_vol,
              `200W=${fmtV(m.min8w_low_200w, "x", 2)} | 4W_WF=${fmtV(m.min4w_wf)} | 量比=${fmtV(m.max4w_vol_r, "x")}`),
            radarCell("极深回撤: 周收盘较52W高 ≤ -85%", m.close_vs_52h_pct !== null && m.close_vs_52h_pct <= -85.0,
              `当前 ${fmtV(m.close_vs_52h_pct, "%", 1)}`),
            radarCell("结构止跌: 周线 Higher Low (本周低 > 上周低)", Boolean(m.w_higher_low),
              `本周低 $${fmtV(m.w_low)} vs 上周低 $${fmtV(m.prev_w_low)}`),
          ];
          const g2 = [
            radarCell("月线处于低位: M_Fish < 0.8", m.m_fish_prev !== null && m.m_fish_prev < 0.8,
              `当前 M_Fish=${fmtV(m.m_fish_prev)} (< 0.8 允许触发)`),
            radarCell("月线金叉: M_Fish 上穿 Signal", m.mf_lin_buy,
              `M_Fish=${fmtV(m.m_fish_prev)} vs Sig=${fmtV(m.m_fish_sig_prev)}`),
          ];
          groups.push(renderRadarGroup("入口规则 1 · F1 主入口（满足全部 5 项 → 买入 100%）", g1, "all"));
          groups.push(renderRadarGroup("入口规则 2 · Deep85 + 周线 Higher Low（满足全部 3 项 → 买入 100%）", g_deep, "all"));
          groups.push(renderRadarGroup("入口规则 3 · 月线 Fisher 金叉兜底（满足全部 2 项 → 买入 100%）", g2, "all"));
        } else if (isSplit) {
          const g2 = [
            radarCell("前置大底: pre_c7_vol 满足", m.pre_c7_vol,
              `200W=${fmtV(m.min8w_low_200w, "x", 2)} | 4W_WF=${fmtV(m.min4w_wf)} | 量比=${fmtV(m.max4w_vol_r, "x")}`),
            radarCell("E1 强确认: 周收盘 > 10周EMA", m.w_close !== null && m.w_ema10 !== null && m.w_close > m.w_ema10,
              `W_Close $${fmtV(m.w_close)} vs 10W_EMA $${fmtV(m.w_ema10)}`),
          ];
          groups.push(renderRadarGroup("入口规则 1 · E2 先买50%（满足全部 4 项 → 买入 50%）", g1, "all"));
          groups.push(renderRadarGroup("入口规则 2 · E1 全仓/补满（满足全部 2 项 → 买入 100%）", g2, "all"));
        } else {
          const preCells = g1.slice(0, -1);
          const trigCells = [g1[g1.length - 1]];
          groups.push(renderRadarGroup(`入口规则 1 · 前置大底区域 (${useF1 ? "pre_c7_vol + F1门控" : "pre_c7_vol"})`, preCells, "all"));
          groups.push(renderRadarGroup("入口规则 2 · 右侧触发确认（满足后 → 买入 100%）", trigCells, "all"));
        }
      }

      box.innerHTML = `
        <div class="cockpit-main">
          <div class="cockpit-meta">${freshHtml}</div>
          ${actionHtml}
          ${holdingHtml}
        </div>
        <div class="cockpit-radar">
          <div class="radar-title">
            <span>${ls && ls.pos_mode !== "FLAT" ? "出口条件雷达 · 持仓退出监控" : "入口条件雷达 · 分组规则监控"}</span>
            <span>周线基准: ${m.w_date}</span>
          </div>
          <div class="radar-groups">${groups.join("")}</div>
        </div>
      `;
    }

    document.getElementById("logToggle").addEventListener("change", e => {
      document.getElementById("logLabel").classList.toggle("checked", e.target.checked);
      render();
    });
    document.querySelectorAll(".opt-check").forEach(el => {
      el.addEventListener("change", () => {
        el.parentElement.classList.toggle("checked", el.checked);
        render();
      });
    });
    document.getElementById("strategySelect").addEventListener("change", render);
    ["upPct", "downPct", "zoneMonths", "zoneTolPct"].forEach(id => {
      document.getElementById(id).addEventListener("change", render);
    });

    render();
  </script>
</body>
</html>
"""


def render_html(
    monthly: list[dict],
    weekly_sma: list[dict],
    c7_entries: list[dict],
    strategies: dict[str, dict],
    live_market: dict | None = None,
) -> str:
    """Render standalone HTML with Plotly.js."""
    m_json = json.dumps(monthly, separators=(",", ":"), allow_nan=False)
    w_json = json.dumps(weekly_sma, separators=(",", ":"), allow_nan=False)
    c7_json = json.dumps(c7_entries, separators=(",", ":"), allow_nan=False)
    s_json = json.dumps(strategies, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    lm_json = json.dumps(live_market, separators=(",", ":"), allow_nan=False, ensure_ascii=False)

    return (
        HTML_TEMPLATE.replace("__MONTHLY_JSON__", m_json)
        .replace("__WEEKLY_SMA_JSON__", w_json)
        .replace("__C7_ENTRIES_JSON__", c7_json)
        .replace("__STRATEGIES_JSON__", s_json)
        .replace("__LIVE_MARKET_JSON__", lm_json)
    )


def format_tg_summary(live_market: dict, strategies: dict[str, dict]) -> str:
    """Format concise SOXL weekly action summary for Telegram."""
    m = live_market
    fr_tag = "" if m["is_friday"] else " [非周五收盘]"
    lines = [
        f"SOXL 策略周报 | {m['last_date']} ({m['weekday_cn']}){fr_tag}",
        f"SOXL: ${m['soxl_close']:.2f} | SMH: ${m['smh_close']:.2f} (200SMA ${m['smh_sma200']:.2f}) | HV20: {m['hv20']:.1f}%",
        "",
    ]

    c2 = strategies["C2_F1_or_MFishLin_TP3h"]["live_status"]
    lines.append(f"★ 主策略 C2+Deep85_HL+TP3h: {c2['badge']}")
    if c2["entry_date"]:
        lines.append(
            f"• 仓位: {c2['position_label']} | 入场 {c2['entry_date']} @ ${c2['entry_px']:.2f} | 浮盈 {c2['ret_pct']:+.1f}% | {c2['hold_weeks']}周"
        )
        lines.append(f"• 说明: {c2['reason']}")
    else:
        f1_ok = (m["wks_52h"] is not None and m["wks_52h"] >= 30) or (
            m["dd52_8wl"] is not None and m["dd52_8wl"] <= -80.0
        )
        e1_checks = [
            ("200W极值", m["min8w_low_200w"] is not None and m["min8w_low_200w"] <= 1.10),
            ("周恐慌", m["min4w_wf"] is not None and m["min4w_wf"] < -1.5),
            ("放量1.6x", m["max4w_vol_r"] is not None and m["max4w_vol_r"] >= 1.6),
            ("F1门控", f1_ok),
            (f"站上周高(${m['prev_w_high']:.2f})", m["w_close"] is not None and m["prev_w_high"] is not None and m["w_close"] > m["prev_w_high"]),
        ]
        e1_done = sum(1 for _, ok in e1_checks if ok)
        e1_miss = [name for name, ok in e1_checks if not ok]
        d_checks = [
            ("pre_c7_vol", bool(m["pre_c7_vol"])),
            ("52W跌幅≤-85%", m["close_vs_52h_pct"] is not None and m["close_vs_52h_pct"] <= -85.0),
            (f"HigherLow(>${m['prev_w_low']:.2f})", bool(m["w_higher_low"])),
        ]
        d_done = sum(1 for _, ok in d_checks if ok)
        d_miss = [name for name, ok in d_checks if not ok]
        e2_checks = [
            ("M_Fish<0.8", m["m_fish_prev"] is not None and m["m_fish_prev"] < 0.8),
            ("月线金叉", bool(m["mf_lin_buy"])),
        ]
        e2_done = sum(1 for _, ok in e2_checks if ok)
        lines.append(f"• 入口1 (F1主入口): {e1_done}/5 满足" + (f" (缺: {', '.join(e1_miss)})" if e1_miss else " (全部满足!)"))
        lines.append(f"• 入口2 (Deep85+HL): {d_done}/3 满足" + (f" (缺: {', '.join(d_miss)})" if d_miss else " (全部满足!)"))
        lines.append(
            f"• 入口3 (月Fisher兜底): {e2_done}/2 满足 (M_Fish={m['m_fish_prev']} vs Sig={m['m_fish_sig_prev']})"
        )
        t = m.get("soxl_gap_tracker")
        if t:
            if t["is_armed"]:
                trig_s = (
                    f" | 已触发 {t['latest_trigger']['date']}@${t['latest_trigger']['open']:.2f}"
                    if t.get("latest_trigger")
                    else ""
                )
                lines.append(
                    f"• Deep85亮灯(跳空接针): 🔥已亮灯(锚${t['anchor_w_close']:.2f}, {t['armed_start_date']}) | 明日开盘≤${t['effective_open_max']:.2f}触发真破位跳空{trig_s}"
                )
            else:
                lines.append("• Deep85亮灯(跳空接针): 熄灭")

    lines.append("")
    for key, label in (
        ("F1_TP3h", "F1+TP3h"),
        ("D1_X1a_WTSL35", "策略3 (D1)"),
        ("H5_E2_Promote_S1_AdaptHV", "策略4 (H5)"),
    ):
        ls = strategies[key]["live_status"]
        extra = f" — {ls['reason']}" if ls["action"] in ("BUY", "SELL", "BUY_HALF", "BUY_TOPUP") else ""
        lines.append(f"• {label}: {ls['badge']}{extra}")

    return "\n".join(lines)


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def send_tg_notification(message: str, timeout: int = 10) -> bool:
    """Send Telegram notification using env/.env or sibling apex/utils/notifier.py."""
    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT.parent / "apex" / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        apex_notifier = ROOT.parent / "apex" / "utils" / "notifier.py"
        if apex_notifier.is_file():
            spec = importlib.util.spec_from_file_location("apex_notifier", apex_notifier)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return bool(mod.send_telegram_message(message, timeout=timeout))
        print("[Telegram] Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID and apex/utils/notifier.py not found.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe_msg = f"<b>[APEXSTOCK]</b>\n{html_lib.escape(message)}"[:4000]
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": safe_msg, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError as e:
            if hasattr(e, "reason") and isinstance(e.reason, ssl.SSLCertVerificationError):
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[Telegram] Send attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return False


def send_pushover_notification(
    title: str,
    message: str,
    priority: int = 2,
    sound: str = "siren",
    retry: int = 30,
    expire: int = 600,
    timeout: int = 10,
) -> bool:
    """Send Pushover notification (priority=2 triggers emergency siren alert until acknowledged)."""
    _load_env_file(ROOT / ".env")
    user_key = os.getenv("PUSHOVER_USER_KEY", "").strip()
    api_token = os.getenv("PUSHOVER_API_TOKEN", "").strip()
    if not user_key or not api_token:
        print("[Pushover] Missing PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN.")
        return False

    payload = {
        "token": api_token,
        "user": user_key,
        "title": title[:250],
        "message": message[:1024],
        "priority": str(priority),
        "sound": sound,
    }
    if priority == 2:
        payload["retry"] = str(max(30, retry))
        payload["expire"] = str(min(10800, expire))

    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=data)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError as e:
            if hasattr(e, "reason") and isinstance(e.reason, ssl.SSLCertVerificationError):
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[Pushover] Send attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe_msg = f"<b>[APEXSTOCK]</b>\n{html_lib.escape(message)}"[:4000]
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": safe_msg, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError as e:
            if hasattr(e, "reason") and isinstance(e.reason, ssl.SSLCertVerificationError):
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[Telegram] Send attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return False


def send_pushover_notification(
    title: str,
    message: str,
    priority: int = 2,
    sound: str = "siren",
    retry: int = 30,
    expire: int = 600,
    timeout: int = 10,
) -> bool:
    """Send Pushover notification (priority=2 triggers emergency siren alert until acknowledged)."""
    _load_env_file(ROOT / ".env")
    user_key = os.getenv("PUSHOVER_USER_KEY", "").strip()
    api_token = os.getenv("PUSHOVER_API_TOKEN", "").strip()
    if not user_key or not api_token:
        print("[Pushover] Missing PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN.")
        return False

    payload = {
        "token": api_token,
        "user": user_key,
        "title": title[:250],
        "message": message[:1024],
        "priority": str(priority),
        "sound": sound,
    }
    if priority == 2:
        payload["retry"] = str(max(30, retry))
        payload["expire"] = str(min(10800, expire))

    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request("https://api.pushover.net/1/messages.json", data=data)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError as e:
            if hasattr(e, "reason") and isinstance(e.reason, ssl.SSLCertVerificationError):
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[Pushover] Send attempt {attempt + 1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SOXL monthly zones HTML and optional Telegram alert.")
    parser.add_argument("--notify-tg", action="store_true", help="Send weekly status summary to Telegram.")
    args = parser.parse_args()
    df_soxl = load_daily_adjusted(DATA_PATH)
    df_smh = load_daily_adjusted(SMH_PATH)

    monthly, weekly_sma = build_monthly_and_weekly(df_soxl)
    c7_entries = detect_c7_vol_entries(df_soxl)

    feats = build_causal_features(df_soxl, df_smh)
    latest_date = str(df_soxl["Date"].max().date())
    c2_tp3h = simulate_tp3h(
        feats,
        use_f1_filter=True,
        end_date=latest_date,
        entry_col_override="C2_Deep85HL",
        mode_tag_override="BOT_C2",
        strategy_id="C2_F1_or_MFishLin_TP3h",
        tp_split_lottery_pct=0.10,
        consec_sl_weeks=2,
    )
    f1_tp3h = simulate_tp3h(feats, use_f1_filter=True, end_date=latest_date)
    f0_tp3h = simulate_tp3h(feats, use_f1_filter=False, end_date=latest_date)
    d1_full = simulate_d1_x1a_wtsl35(feats, use_split_50_50=False, end_date=latest_date)
    d1_split = simulate_d1_x1a_wtsl35(feats, use_split_50_50=True, end_date=latest_date)
    h5 = simulate_h5_adapt_hv(feats, end_date=latest_date)

    if latest_date == BT_END:
        c2_15yr = simulate_tp3h(
            feats,
            use_f1_filter=True,
            start_date="2011-06-01",
            end_date=BT_END,
            entry_col_override="C2_F1_or_MFishLin",
            mode_tag_override="BOT_C2",
            strategy_id="C2_F1_or_MFishLin_TP3h",
        )
        assert len(c7_entries) == 5, f"Expected 5 C7_Vol entries, got {len(c7_entries)}"
        assert c2_tp3h["stats"]["trades"] == 5 and abs(c2_tp3h["stats"]["final"] - 99311369.63) < 1.0, c2_tp3h["stats"]
        assert c2_15yr["stats"]["trades"] == 7 and abs(c2_15yr["stats"]["final"] - 182231668.40) < 1.0, c2_15yr["stats"]
        assert f1_tp3h["stats"]["trades"] == 5 and abs(f1_tp3h["stats"]["final"] - 99311369.63) < 1.0 and f1_tp3h["stats"]["win_rate"] == 100.0, f1_tp3h["stats"]
        assert f0_tp3h["stats"]["trades"] == 6 and abs(f0_tp3h["stats"]["final"] - 66812281.30) < 1.0 and f0_tp3h["stats"]["win_rate"] == 83.3, f0_tp3h["stats"]
        assert d1_full["stats"]["trades"] == 6 and abs(d1_full["stats"]["final"] - 38035375.82) < 1.0, d1_full["stats"]
        assert d1_split["stats"]["trades"] == 6 and abs(d1_split["stats"]["final"] - 30120328.33) < 1.0, d1_split["stats"]
        assert h5["stats"]["trades"] == 32 and abs(h5["stats"]["final"] - 37913860.48) < 1.0, h5["stats"]

    strategies = {
        "C2_F1_or_MFishLin_TP3h": c2_tp3h,
        "F1_TP3h": f1_tp3h,
        "F0_TP3h": f0_tp3h,
        "D1_X1a_WTSL35": d1_full,
        "D1_X1a_WTSL35_Split50": d1_split,
        "H5_E2_Promote_S1_AdaptHV": h5,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    live_market = build_live_market_snapshot(feats)
    live_market["soxl_gap_tracker"] = build_soxl_deep85_gap_tracker(feats)
    html = render_html(monthly, weekly_sma, c7_entries, strategies, live_market=live_market)

    d85_css = """
    .d85-light { margin: 12px 24px 0; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }
    .d85-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid #21262d; }
    .d85-title-wrap { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .d85-lamp { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .d85-lamp.on { background: #3fb950; box-shadow: 0 0 10px rgba(63,185,80,0.85); }
    .d85-lamp.warn { background: #f0883e; box-shadow: 0 0 10px rgba(240,136,62,0.85); }
    .d85-lamp.off { background: #484f58; }
    .d85-title { font-size: 14px; font-weight: 700; color: #f0f6fc; }
    .d85-sub { font-size: 11px; color: #8b949e; background: #0d1117; border: 1px solid #30363d; border-radius: 999px; padding: 2px 8px; }
    .d85-stage { font-size: 12px; font-weight: 600; border-radius: 6px; padding: 4px 10px; }
    .d85-stage.muted { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
    .d85-stage.warn { background: rgba(240,136,62,0.14); color: #f0883e; border: 1px solid rgba(240,136,62,0.45); }
    .d85-stage.ready { background: rgba(63,185,80,0.16); color: #3fb950; border: 1px solid rgba(63,185,80,0.45); }
    .d85-cols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
    @media (max-width: 980px) { .d85-cols { grid-template-columns: 1fr; } }
    .d85-card { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 11px 12px; }
    .d85-card-h { font-size: 12px; font-weight: 600; color: #c9d1d9; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .d85-item { font-size: 12px; padding: 6px 8px; border-radius: 6px; margin-bottom: 6px; border: 1px solid #21262d; background: #161b22; }
    .d85-item.ok { border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.08); }
    .d85-item.warn { border-color: rgba(240,136,62,0.45); background: rgba(240,136,62,0.10); }
    .d85-item.miss { border-color: #21262d; color: #8b949e; }
    .d85-item-top { display: flex; justify-content: space-between; gap: 6px; font-weight: 600; }
    .d85-item.ok .d85-item-top { color: #3fb950; }
    .d85-item.warn .d85-item-top { color: #f0883e; }
    .d85-item-val { font-size: 11px; color: #8b949e; margin-top: 3px; line-height: 1.45; }
    .d85-bigpx { font-size: 20px; font-weight: 800; color: #3fb950; margin: 4px 0; }
    .d85-bigpx.off { color: #8b949e; font-size: 15px; }
    .d85-note { font-size: 11px; color: #8b949e; margin-top: 6px; line-height: 1.45; }
    .d85-table-wrap { margin-top: 12px; overflow-x: auto; border: 1px solid #21262d; border-radius: 8px; }
    .d85-table { width: 100%; border-collapse: collapse; font-size: 12px; background: #0d1117; }
    .d85-table th, .d85-table td { padding: 6px 8px; text-align: right; border-bottom: 1px solid #21262d; white-space: nowrap; }
    .d85-table th:first-child, .d85-table td:first-child, .d85-table th:last-child, .d85-table td:last-child { text-align: left; }
    .d85-table th { color: #8b949e; font-weight: 600; background: #161b22; }
    .d85-badge-trig { color: #3fb950; font-weight: 700; }
    .d85-badge-trap { color: #f0883e; font-weight: 600; }
    .d85-badge-wait { color: #8b949e; }
    .d85-hist { margin-top: 8px; font-size: 11px; color: #8b949e; display: flex; flex-wrap: wrap; gap: 14px; }
    .d85-hist b { color: #c9d1d9; }
  </style>"""
    assert "</style>" in html, "Missing </style> anchor"
    html = html.replace("</style>", d85_css, 1)

    cockpit_div = '<div id="liveCockpit" class="live-cockpit"></div>'
    assert cockpit_div in html, "Missing liveCockpit anchor"
    html = html.replace(cockpit_div, cockpit_div + '\n\n  <div id="soxlDeep85Light" class="d85-light"></div>', 1)

    d85_js = """
    function renderSoxlDeep85Light() {
      const el = document.getElementById("soxlDeep85Light");
      const m = LIVE_MARKET;
      if (!el || !m || !m.soxl_gap_tracker) {
        if (el) el.style.display = "none";
        return;
      }
      const t = m.soxl_gap_tracker;
      const lampCls = !t.is_armed ? "off" : (t.stage_code === "TRIGGERED" ? "on" : "warn");

      const col1 = `
        <div class="d85-card">
          <div class="d85-card-h">
            <span>周线亮灯状态（每周五收盘结算）</span>
            <span>基准周: ${t.w_date}</span>
          </div>
          <div class="d85-item ${t.pre_c7_vol ? "ok" : "miss"}">
            <div class="d85-item-top">
              <span>1. 前置大底 pre_c7_vol</span>
              <span>${t.pre_c7_vol ? "✓ 满足" : "○ 未满足"}</span>
            </div>
            <div class="d85-item-val">200W=${fmtV(t.min8w_low_200w, "x")} | 4W_WF=${fmtV(t.min4w_wf)} | 量比=${fmtV(t.max4w_vol_r, "x")}</div>
          </div>
          <div class="d85-item ${t.dd52_ok ? "ok" : "miss"}">
            <div class="d85-item-top">
              <span>2. 极深回撤: 周收盘较52W高 ≤ -85%</span>
              <span>${t.dd52_ok ? "✓ 满足" : "○ 未满足"}</span>
            </div>
            <div class="d85-item-val">当前周收盘较52W高 ${fmtV(t.dd52_pct, "%", 1)}</div>
          </div>
          <div class="d85-item ${t.is_armed ? "ok" : "miss"}">
            <div class="d85-item-top">
              <span>锚点周五收盘价</span>
              <span>${t.is_armed ? `$${fmtV(t.anchor_w_close)}` : "熄灭"}</span>
            </div>
            <div class="d85-item-val">${t.is_armed ? `亮灯起始 ${t.armed_start_date} (已亮 ${t.armed_weeks_count} 周) · 最低周五收 ${t.anchor_w_date}` : "亮灯后连续每周五满足均保持常亮"}</div>
          </div>
        </div>`;

      const col2 = t.is_armed ? `
        <div class="d85-card">
          <div class="d85-card-h">
            <span>明日 9:30 开盘判定线（零心算直接对照）</span>
            <span>昨收 $${fmtV(t.last_close)}</span>
          </div>
          <div class="d85-item warn">
            <div class="d85-item-top">
              <span>明日生效买入门槛</span>
              <span>两项须同时满足</span>
            </div>
            <div class="d85-bigpx">开盘价 ≤ $${fmtV(t.effective_open_max)}</div>
            <div class="d85-item-val">① 跳空≤-8%: 昨收 $${fmtV(t.last_close)}×0.92 = <b>$${fmtV(t.gap8_trigger_px)}</b><br>② 防诱多护栏: 须 ≤ 亮灯周五收 <b>$${fmtV(t.anchor_w_close)}</b></div>
          </div>
          <div class="d85-note">
            <b>一旦开盘触发（O ≤ $${fmtV(t.effective_open_max)}）：</b><br>
            • <b>稳健买法</b>：开盘价 O 直接买（2022/2025/2026 全部命中）<br>
            • <b>接针买法</b>：开盘后挂 <code>O×0.97 (-3%)</code> 或 <code>O×0.95 (-5%)</code>
          </div>
        </div>` : `
        <div class="d85-card">
          <div class="d85-card-h">
            <span>明日 9:30 开盘判定线</span>
            <span>昨收 $${fmtV(t.last_close)}</span>
          </div>
          <div class="d85-item miss">
            <div class="d85-item-top"><span>Deep85 灯未亮</span><span>○ 锁定</span></div>
            <div class="d85-bigpx off">灯未亮，明日开盘无需操作</div>
            <div class="d85-item-val">仅当周五收盘同时满足 pre_c7_vol 且较52W高 ≤ -85% 时亮灯解锁。</div>
          </div>
        </div>`;

      const trigHtml = t.latest_trigger ? `
        <div class="d85-item ok">
          <div class="d85-item-top">
            <span>最新真破位跳空命中</span>
            <span>${t.latest_trigger.date}</span>
          </div>
          <div class="d85-item-val">
            开盘 <b>$${fmtV(t.latest_trigger.open)}</b> (跳空 ${fmtV(t.latest_trigger.gap_pct, "%", 1)} ≤ 锚 $${fmtV(t.latest_trigger.anchor)})<br>
            盘中低 $${fmtV(t.latest_trigger.low)} | -3% $${fmtV(t.latest_trigger.dip3_px)} ${t.latest_trigger.dip3_hit ? "✓" : "✗"} / -5% $${fmtV(t.latest_trigger.dip5_px)} ${t.latest_trigger.dip5_hit ? "✓" : "✗"} | 收 $${fmtV(t.latest_trigger.close)}
          </div>
        </div>` : `
        <div class="d85-item miss">
          <div class="d85-item-top">
            <span>本轮真破位跳空命中</span>
            <span>${t.is_armed ? "等待中" : "未亮灯"}</span>
          </div>
          <div class="d85-item-val">亮灯后不限次日，只要后续任一交易日开盘同时满足跳空≤-8% 且 ≤ 锚点收盘价即触发。</div>
        </div>`;

      const col3 = `
        <div class="d85-card">
          <div class="d85-card-h">
            <span>本轮触发摘要 & 熄灭规则</span>
            <span>最新 ${t.last_date}</span>
          </div>
          ${trigHtml}
          <div class="d85-note">
            • <b>熄灭条件</b>：完成买入，或某周五收盘反弹回到 52W 高 -85% 以上。<br>
            • <b>防踏空兜底</b>：上方原策略 <b>入口1 (F1)</b> 与 <b>入口3 (月线金叉)</b> 仍全程保留；未满仓时仍受 2 周 -25% 假底止损保护。
          </div>
        </div>`;

      let tableHtml = "";
      if (t.daily_rows && t.daily_rows.length > 0) {
        tableHtml = `
          <div class="d85-table-wrap">
            <table class="d85-table">
              <thead>
                <tr>
                  <th>日期</th><th>Open</th><th>相对昨收跳空</th><th>≤ 锚点($)</th><th>Low</th><th>开盘再跌</th><th>-3% / -5% 接针</th><th>Close</th><th>判定</th>
                </tr>
              </thead>
              <tbody>
                ${t.daily_rows.map(r => {
                  const badgeCls = r.status_code === "TRIGGERED" ? "d85-badge-trig" : (r.status_code === "BLOCKED_TRAP" ? "d85-badge-trap" : "d85-badge-wait");
                  const dipNote = r.triggered
                    ? `$${fmtV(r.dip3_px)}(${r.dip3_hit ? "✓" : "✗"}) / $${fmtV(r.dip5_px)}(${r.dip5_hit ? "✓" : "✗"})`
                    : "—";
                  return `<tr>
                    <td>${r.date}</td>
                    <td>$${fmtV(r.open)}</td>
                    <td>${r.gap_pct >= 0 ? "+" : ""}${fmtV(r.gap_pct, "%", 1)}</td>
                    <td>${r.below_anchor ? `✓ ≤$${fmtV(r.anchor)}` : `✗ >$${fmtV(r.anchor)}`}</td>
                    <td>$${fmtV(r.low)}</td>
                    <td>${fmtV(r.dip_open_pct, "%", 1)}</td>
                    <td>${dipNote}</td>
                    <td>$${fmtV(r.close)}</td>
                    <td><span class="${badgeCls}">${r.status_text}</span></td>
                  </tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>`;
      }

      const histHtml = (t.history_episodes && t.history_episodes.length > 0)
        ? `<div class="d85-hist"><span>历史 ${t.history_episodes.length} 次 Deep85 亮灯对照：</span>${t.history_episodes.map(h =>
            `<span><b>${h.armed_date}</b> (锚$${fmtV(h.anchor_close)}) → ${h.first_trigger_date ? `${h.first_trigger_date} 开$${fmtV(h.trigger_open)} (${fmtV(h.trigger_gap_pct, "%", 1)}, -3%${h.dip3_hit ? "✓" : "✗"}/-5%${h.dip5_hit ? "✓" : "✗"})` : "无真破位跳空"}</span>`
          ).join("")}</div>`
        : "";

      el.innerHTML = `
        <div class="d85-head">
          <div class="d85-title-wrap">
            <span class="d85-lamp ${lampCls}"></span>
            <span class="d85-title">SOXL Deep85 亮灯 & 跳空接针监视器</span>
            <span class="d85-sub">独立参考 · 不改变原有策略信号</span>
          </div>
          <span class="d85-stage ${t.stage_tone}">${t.stage_label}</span>
        </div>
        <div class="d85-cols">${col1}${col2}${col3}</div>
        ${tableHtml}
        ${histHtml}
      `;
    }"""

    render_anchor = "\n\n    render();\n  </script>"
    assert render_anchor in html, "Missing top-level render() anchor"
    html = html.replace(render_anchor, f"\n{d85_js}\n\n    render();\n    renderSoxlDeep85Light();\n  </script>", 1)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")

    if args.notify_tg:
        tg_msg = format_tg_summary(live_market, strategies)
        ok = send_tg_notification(tg_msg)
        print(f"[Telegram] sent={ok}")
        c2_act = strategies["C2_F1_or_MFishLin_TP3h"]["live_status"]["action"]
        h5_act = strategies["H5_E2_Promote_S1_AdaptHV"]["live_status"]["action"]
        if c2_act in ("BUY", "SELL") or h5_act in ("BUY", "SELL"):
            po_ok = send_pushover_notification(
                title=f"🚨 SOXL 交易信号 ({live_market['last_date']})",
                message=f"C2: {strategies['C2_F1_or_MFishLin_TP3h']['live_status']['badge']}\nH5: {strategies['H5_E2_Promote_S1_AdaptHV']['live_status']['badge']}\nSOXL=${live_market['soxl_close']:.2f}",
            )
            print(f"[Pushover] sent={po_ok}")


if __name__ == "__main__":
    main()
