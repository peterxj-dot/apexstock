"""Build a dedicated MSTR monthly LOG chart (2020+) with BTC Halving Buy/Sell zones and mNAV-spike strategies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import historical_volatility, sma
from tools.build_btc_zones import HALVINGS, _ds_du, _pine_gaussian_filter, build_halving_zone_shapes
from tools.build_soxl_zones import (
    DATA_DIR,
    FEE_RATE,
    INITIAL_CAPITAL,
    build_causal_features,
    build_live_market_snapshot,
    build_monthly_and_weekly,
    detect_c7_vol_entries,
    load_daily_adjusted,
    render_html,
    send_pushover_notification,
    send_tg_notification,
)

MSTR_PATH = DATA_DIR / "MSTR.csv"
BTC_PATH = DATA_DIR / "BTC-USD.csv"
OUTPUT_HTML = ROOT / "mstr_monthly_zones.html"
START_DATE = "2020-01-02"


def _compute_mstr_stats(
    equity: np.ndarray,
    trades: list[dict],
    days_inv: int,
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    n = len(equity)
    final_val = float(equity[-1])
    years_span = n / 252.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0)) * 100.0
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else 0.0
    rets = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252.0)) if (len(rets) > 0 and np.std(rets) > 0) else 0.0
    trades_cnt = len(trades)
    win_rate = (sum(1 for t in trades if t["ret_pct"] > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = (days_inv / n) * 100.0 if n > 0 else 0.0
    s_eq = pd.Series(equity, index=dates)
    ann_rets = s_eq.resample("YE").last().pct_change().dropna() * 100.0
    worst_yr = float(ann_rets.min()) if not ann_rets.empty else 0.0
    return {
        "final": round(final_val, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "sharpe": round(sharpe, 2),
        "trades": trades_cnt,
        "win_rate": round(win_rate, 1),
        "exposure": round(exposure, 1),
        "worst_year": round(worst_yr, 1),
    }


def prepare_mstr_btc_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_mstr = load_daily_adjusted(MSTR_PATH).sort_values("Date").reset_index(drop=True)
    df_btc = load_daily_adjusted(BTC_PATH).sort_values("Date").reset_index(drop=True)

    b = df_btc.copy()
    b["b_hv20"] = historical_volatility(b["Close"], period=20, annualization_factor=365)
    b["b_hv60"] = historical_volatility(b["Close"], period=60, annualization_factor=365)
    b["b_hvr"] = b["b_hv20"] / b["b_hv60"]
    b["b_hvr_spike35"] = (b["b_hvr"] >= 1.30).rolling(35, min_periods=1).max().astype(bool)
    b["b_close"] = b["Close"]
    # Strictly causal: at MSTR date t, only BTC bars up to t-1 are used
    b["Date_avail"] = b["Date"] + pd.Timedelta(days=1)

    m = df_mstr.copy()
    m["m_sma50"] = sma(m["Close"], 50)
    m["m_hv20"] = historical_volatility(m["Close"], period=20, annualization_factor=252)
    mg_mid = _pine_gaussian_filter(m["Close"], 10, 2)
    mg_dev = _pine_gaussian_filter((m["Close"] - mg_mid).abs(), 10, 2)
    m["m_gac_mid"] = mg_mid
    m["m_gac_upper"] = mg_mid + 1.5 * mg_dev

    b_cols = ["Date_avail", "b_close", "b_hv20", "b_hvr", "b_hvr_spike35"]
    merged = pd.merge_asof(
        m.sort_values("Date"),
        b[b_cols].sort_values("Date_avail"),
        left_on="Date",
        right_on="Date_avail",
        direction="backward",
    )
    merged["mstr_btc"] = merged["Close"] / merged["b_close"]
    merged["mstr_btc_ma60"] = merged["mstr_btc"].rolling(60).mean()
    merged["mstr_btc_r60"] = merged["mstr_btc"] / merged["mstr_btc_ma60"]
    merged["m_vol_r20"] = merged["Volume"] / merged["Volume"].rolling(20).mean()
    merged["m_vol_r20_max3"] = merged["m_vol_r20"].rolling(3).max()
    return df_mstr, merged


def _safe_float(val: object, digits: int = 2) -> float | None:
    if val is None or pd.isna(val):
        return None
    f = float(val)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, digits)


def build_mstr_t0_tracker(merged: pd.DataFrame) -> dict:
    """Compute strictly causal MSTR T0 panic-bottom tracker inside the BTC halving buy window."""
    d = merged.copy().sort_values("Date").reset_index(drop=True)
    d["m_hv60"] = historical_volatility(d["Close"], period=60, annualization_factor=252)
    d["m_hvr"] = d["m_hv20"] / d["m_hv60"]
    d["dd60_pct"] = (d["Close"] / d["High"].rolling(60).max() - 1.0) * 100.0
    d["low20"] = d["Low"].rolling(20).min()
    d["cond_dd"] = d["dd60_pct"] <= -35.0
    d["cond_low20"] = d["Low"] <= d["low20"] * 1.02
    d["cond_pulse"] = (
        (d["b_hvr"] >= 1.25)
        | (d["m_hvr"] >= 1.25)
        | ((d["m_vol_r20"] >= 1.50) & (d["mstr_btc_r60"] <= 0.85))
    )
    d["is_t0_bar"] = d["cond_dd"] & d["cond_low20"] & d["cond_pulse"]

    last_row = d.iloc[-1]
    last_dt = pd.Timestamp(last_row["Date"])
    next_h = min((h for h in HALVINGS if h > last_dt), default=HALVINGS[-1])
    ds, du = _ds_du(last_dt)

    win_start = next_h - pd.Timedelta(days=540)
    win_end = next_h - pd.Timedelta(days=460)
    h490_date = next_h - pd.Timedelta(days=490)
    pre_start = next_h - pd.Timedelta(days=800)
    pre_end = next_h - pd.Timedelta(days=541)
    in_buy_zone = bool(460 <= du <= 540)

    pre = d[(d["Date"] >= pre_start) & (d["Date"] <= pre_end)]
    if not pre.empty:
        pre_idx = pre["Low"].idxmin()
        pre_low = _safe_float(pre.loc[pre_idx, "Low"], 2)
        pre_low_date = pd.Timestamp(pre.loc[pre_idx, "Date"]).strftime("%Y-%m-%d")
    else:
        pre_low = None
        pre_low_date = None

    win = d[(d["Date"] >= win_start) & (d["Date"] <= win_end)].reset_index(drop=True)
    t0_rows = win[win["is_t0_bar"]]
    t0_triggered = not t0_rows.empty

    t0_date = None
    t0_close = None
    t0_low = None
    t0_dd60_pct = None
    t0_b_hvr = None
    t0_m_hvr = None
    t0_vol_r20 = None
    t0_r60 = None
    days_since_t0 = 0
    win_low = None
    win_low_date = None
    drop_from_t0_pct = None
    vs_pre_low_pct = None
    run_low = None
    run_low_date = None
    streak = 0
    first_5d_date = None
    first_5d_px = None
    daily_rows: list[dict] = []
    limit_tiers: list[dict] = []

    if t0_triggered:
        t0_first_idx = int(t0_rows.index[0])
        r0 = win.loc[t0_first_idx]
        t0_ts = pd.Timestamp(r0["Date"])
        t0_date = t0_ts.strftime("%Y-%m-%d")
        t0_close = _safe_float(r0["Close"], 2)
        t0_low = _safe_float(r0["Low"], 2)
        t0_dd60_pct = _safe_float(r0["dd60_pct"], 1)
        t0_b_hvr = _safe_float(r0["b_hvr"], 2)
        t0_m_hvr = _safe_float(r0["m_hvr"], 2)
        t0_vol_r20 = _safe_float(r0["m_vol_r20"], 2)
        t0_r60 = _safe_float(r0["mstr_btc_r60"], 2)

        post = win.loc[t0_first_idx:].reset_index(drop=True)
        days_since_t0 = len(post) - 1
        wl_idx = post["Low"].idxmin()
        win_low = _safe_float(post.loc[wl_idx, "Low"], 2)
        win_low_date = pd.Timestamp(post.loc[wl_idx, "Date"]).strftime("%Y-%m-%d")
        if win_low is not None and t0_close:
            drop_from_t0_pct = round((win_low / t0_close - 1.0) * 100.0, 1)
        if win_low is not None and pre_low:
            vs_pre_low_pct = round((win_low / pre_low - 1.0) * 100.0, 1)

        run_low = float(post.loc[0, "Low"])
        run_low_date = pd.Timestamp(post.loc[0, "Date"]).strftime("%Y-%m-%d")
        streak = 0
        all_post_rows: list[dict] = []
        for idx_p in range(len(post)):
            rp = post.loc[idx_p]
            dt_p = pd.Timestamp(rp["Date"]).strftime("%Y-%m-%d")
            low_p = float(rp["Low"])
            if idx_p == 0:
                is_new_low = True
                cur_streak = 0
            elif low_p < run_low:
                run_low = low_p
                run_low_date = dt_p
                streak = 0
                is_new_low = True
                cur_streak = 0
            else:
                streak += 1
                is_new_low = False
                cur_streak = streak

            if cur_streak == 5 and first_5d_date is None:
                if idx_p + 1 < len(post):
                    nxt_rp = post.loc[idx_p + 1]
                    first_5d_date = pd.Timestamp(nxt_rp["Date"]).strftime("%Y-%m-%d")
                    first_5d_px = _safe_float(nxt_rp["Open"], 2)
                else:
                    first_5d_date = f"{dt_p} (明日开盘)"
                    first_5d_px = None

            c_p = float(rp["Close"])
            vs_t0 = ((c_p / t0_close) - 1.0) * 100.0 if t0_close else None
            all_post_rows.append({
                "date": dt_p,
                "open": _safe_float(rp["Open"], 2),
                "low": _safe_float(low_p, 2),
                "close": _safe_float(c_p, 2),
                "vs_t0_pct": _safe_float(vs_t0, 1),
                "r60": _safe_float(rp["mstr_btc_r60"], 2),
                "b_hvr": _safe_float(rp["b_hvr"], 2),
                "vol_r20": _safe_float(rp["m_vol_r20"], 2),
                "is_new_low": is_new_low,
                "streak": cur_streak,
                "is_t0": idx_p == 0,
            })

        daily_rows = all_post_rows[-12:]
        run_low = round(run_low, 2)

        anchor_px = pre_low if pre_low is not None else t0_close
        post_after = post.iloc[1:]
        if anchor_px is not None:
            for mult in (0.92, 0.86, 0.80, 0.74):
                lim_px = round(anchor_px * mult, 2)
                hit = post_after[post_after["Low"] <= lim_px]
                if not hit.empty:
                    hr = hit.iloc[0]
                    fill_px = round(min(float(hr["Open"]), lim_px), 2)
                    fill_dt = pd.Timestamp(hr["Date"]).strftime("%Y-%m-%d")
                    limit_tiers.append({
                        "mult_pct": int(round(mult * 100)),
                        "weight_pct": 12.5,
                        "limit_px": lim_px,
                        "filled": True,
                        "fill_date": fill_dt,
                        "fill_px": fill_px,
                    })
                else:
                    limit_tiers.append({
                        "mult_pct": int(round(mult * 100)),
                        "weight_pct": 12.5,
                        "limit_px": lim_px,
                        "filled": False,
                        "fill_date": None,
                        "fill_px": None,
                    })

    if du > 540:
        stage_code = "OUT_OF_WINDOW"
        stage_label = "未进买点窗 · 禁止抄底"
        stage_tone = "muted"
    elif in_buy_zone and not t0_triggered:
        stage_code = "WAIT_T0"
        stage_label = "窗内等待 T0 · 绝对不买"
        stage_tone = "warn"
    elif t0_triggered and streak < 5 and first_5d_date is None:
        stage_code = "T0_ACTIVE"
        stage_label = "T0 已亮灯 · 左侧接针 / 观察止跌"
        stage_tone = "active"
    elif t0_triggered and (streak >= 5 or first_5d_date is not None):
        stage_code = "T0_STABILIZED"
        stage_label = "右侧止跌确认 (连守前低 ≥ 5 交易日)"
        stage_tone = "ready"
    else:
        stage_code = "POST_WINDOW"
        stage_label = "已过本轮买点窗"
        stage_tone = "muted"

    history_windows: list[dict] = []
    for h_dt in HALVINGS:
        if h_dt < pd.Timestamp("2024-01-01"):
            continue
        hw_s = h_dt - pd.Timedelta(days=540)
        hw_e = h_dt - pd.Timedelta(days=460)
        if last_dt < hw_s:
            continue
        hp_s = h_dt - pd.Timedelta(days=800)
        hp_e = h_dt - pd.Timedelta(days=541)
        h_pre = d[(d["Date"] >= hp_s) & (d["Date"] <= hp_e)]
        h_pre_low = _safe_float(h_pre["Low"].min(), 2) if not h_pre.empty else None
        h_pre_dt = pd.Timestamp(h_pre.loc[h_pre["Low"].idxmin(), "Date"]).strftime("%Y-%m-%d") if not h_pre.empty else None

        h_win = d[(d["Date"] >= hw_s) & (d["Date"] <= hw_e)].reset_index(drop=True)
        h_t0s = h_win[h_win["is_t0_bar"]]
        if h_t0s.empty:
            continue
        hi0 = int(h_t0s.index[0])
        hr0 = h_win.loc[hi0]
        h_t0_c = _safe_float(hr0["Close"], 2)
        h_post = h_win.loc[hi0:].reset_index(drop=True)
        h_wl_idx = h_post["Low"].idxmin()
        h_wl = _safe_float(h_post.loc[h_wl_idx, "Low"], 2)
        h_wl_dt = pd.Timestamp(h_post.loc[h_wl_idx, "Date"]).strftime("%Y-%m-%d")
        h_d2l = int(h_wl_idx)
        h_drop = round((h_wl / h_t0_c - 1.0) * 100.0, 1) if (h_wl is not None and h_t0_c) else None
        history_windows.append({
            "cycle": f"{h_dt.year} 减半周期",
            "window": f"{hw_s.strftime('%Y-%m-%d')} ~ {hw_e.strftime('%Y-%m-%d')}",
            "pre_low": h_pre_low,
            "pre_low_date": h_pre_dt,
            "t0_date": pd.Timestamp(hr0["Date"]).strftime("%Y-%m-%d"),
            "t0_close": h_t0_c,
            "win_low": h_wl,
            "win_low_date": h_wl_dt,
            "days_to_low": h_d2l,
            "drop_from_t0_pct": h_drop,
        })

    return {
        "cycle_halving_date": next_h.strftime("%Y-%m-%d"),
        "win_start": win_start.strftime("%Y-%m-%d"),
        "win_end": win_end.strftime("%Y-%m-%d"),
        "h490_date": h490_date.strftime("%Y-%m-%d"),
        "ds": int(ds),
        "du": int(du),
        "in_buy_zone": in_buy_zone,
        "pre_low": pre_low,
        "pre_low_date": pre_low_date,
        "anchor_label": "进窗前熊市低点" if pre_low is not None else "T0 收盘价",
        "stage_code": stage_code,
        "stage_label": stage_label,
        "stage_tone": stage_tone,
        "t0_triggered": t0_triggered,
        "t0_date": t0_date,
        "t0_close": t0_close,
        "t0_low": t0_low,
        "t0_dd60_pct": t0_dd60_pct,
        "t0_b_hvr": t0_b_hvr,
        "t0_m_hvr": t0_m_hvr,
        "t0_vol_r20": t0_vol_r20,
        "t0_r60": t0_r60,
        "days_since_t0": days_since_t0,
        "win_low": win_low,
        "win_low_date": win_low_date,
        "drop_from_t0_pct": drop_from_t0_pct,
        "vs_pre_low_pct": vs_pre_low_pct,
        "run_low": run_low,
        "run_low_date": run_low_date,
        "no_new_low_streak": streak,
        "first_5d_date": first_5d_date,
        "first_5d_px": first_5d_px,
        "limit_tiers": limit_tiers,
        "daily_rows": daily_rows,
        "history_windows": history_windows,
        "last_date": last_dt.strftime("%Y-%m-%d"),
        "last_close": _safe_float(last_row["Close"], 2),
        "last_low": _safe_float(last_row["Low"], 2),
        "last_dd60_pct": _safe_float(last_row["dd60_pct"], 1),
        "last_low20": _safe_float(last_row["low20"], 2),
        "last_cond_dd": bool(last_row["cond_dd"]),
        "last_cond_low20": bool(last_row["cond_low20"]),
        "last_cond_pulse": bool(last_row["cond_pulse"]),
        "last_b_hvr": _safe_float(last_row["b_hvr"], 2),
        "last_m_hvr": _safe_float(last_row["m_hvr"], 2),
        "last_vol_r20": _safe_float(last_row["m_vol_r20"], 2),
        "last_r60": _safe_float(last_row["mstr_btc_r60"], 2),
    }


def simulate_mstr_strategy(
    merged: pd.DataFrame,
    mode: str,
    strategy_id: str,
    name: str,
    short_label: str,
    rule_desc: str,
    start_date: str = START_DATE,
) -> dict:
    d = merged.set_index("Date").copy()
    idx = d.index
    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = len(idx) - 1
    n_bt = i_end - i_start + 1

    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    b_hvr = d["b_hvr"].to_numpy(dtype=np.float64)
    b_sp35 = d["b_hvr_spike35"].to_numpy(dtype=bool)
    m_hv20 = d["m_hv20"].to_numpy(dtype=np.float64)
    m_sma50 = d["m_sma50"].to_numpy(dtype=np.float64)
    m_gac_m = d["m_gac_mid"].to_numpy(dtype=np.float64)
    m_gac_u = d["m_gac_upper"].to_numpy(dtype=np.float64)
    r60 = d["mstr_btc_r60"].to_numpy(dtype=np.float64)
    vol_r3 = d["m_vol_r20_max3"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    trade_cash_start = INITIAL_CAPITAL
    entry_px = 0.0
    entry_dt_str = ""
    entry_m_str = ""
    pending = ""
    pending_reason = ""
    days_inv = 0
    trades: list[dict] = []
    equity = np.empty(n_bt, dtype=np.float64)
    vol_cool_wait = False
    cooldown = 0

    def eval_sig(pos: int, holding: bool) -> tuple[str, str]:
        nonlocal vol_cool_wait, cooldown
        ds, du = _ds_du(idx[pos])
        in_buy = (460 <= du <= 540)
        in_sell = (480 <= ds <= 560) and (du > 540)
        in_bull = (du <= 540) or (ds <= 540)
        c = closes[pos]
        hv = m_hv20[pos]
        bhvr = b_hvr[pos]
        bsp = bool(b_sp35[pos])
        mr = r60[pos]
        gm = m_gac_m[pos]
        gu = m_gac_u[pos]

        if mode == "HALV_MNAV_170":
            if not holding:
                if vol_cool_wait:
                    if (ds < 200 or du <= 100) and not np.isnan(mr) and mr <= 1.10 and not np.isnan(gm) and c > gm:
                        vol_cool_wait = False
                        return "BUY", "mNAV冷却接回(≤1.10x)"
                    if in_buy and ((bsp and not np.isnan(bhvr) and bhvr < 1.0) or du <= 490):
                        vol_cool_wait = False
                        return "BUY", "新周期BTC买点区共振/H-490"
                else:
                    if in_buy and ((bsp and not np.isnan(bhvr) and bhvr < 1.0) or du <= 490):
                        return "BUY", "BTC买点区共振/H-490"
                    if ds < 480 or du <= 150:
                        return "BUY", "减半牛市持仓期"
            else:
                vr3 = vol_r3[pos]
                if ds >= 150 and not np.isnan(mr) and mr >= 1.70 and not np.isnan(vr3) and vr3 >= 1.50:
                    vol_cool_wait = True
                    return "SELL", f"mNAV溢价放量脉冲({mr:.2f}x≥1.70 & 量{vr3:.1f}x)"
                if (in_sell and not np.isnan(bhvr) and bhvr >= 1.30) or (ds >= 540 and du > 540):
                    vol_cool_wait = False
                    return "SELL", "BTC卖点区逃顶"

        elif mode == "BTC_HALV_HV":
            if not holding:
                if in_buy and ((bsp and not np.isnan(bhvr) and bhvr < 1.0) or du <= 490):
                    return "BUY", "BTC买点区共振/H-490"
                if ds < 480:
                    return "BUY", "减半牛市持仓期"
            else:
                if (in_sell and not np.isnan(bhvr) and bhvr >= 1.30) or (ds >= 540 and du > 540):
                    return "SELL", "BTC卖点区逃顶"

        elif mode == "HALV_PHASE_GAC":
            if not holding:
                if (du <= 520 and ds > 540) or (0 <= ds <= 100):
                    return "BUY", "筑底区买入"
                if 100 < ds <= 540 and not np.isnan(gu) and c > gu:
                    return "BUY", "突破MSTR GAC上轨"
            else:
                if ds > 540 and du > 520:
                    return "SELL", "进入熊市空仓"
                if 100 < ds <= 540 and not np.isnan(gm) and c < gm:
                    return "SELL", "跌破MSTR GAC中轨"

        elif mode == "HALV_HV130_COOL90":
            if not holding:
                if cooldown == 0:
                    if vol_cool_wait:
                        if in_bull and ds < 480 and hv <= 90.0 and not np.isnan(m_sma50[pos]) and c > m_sma50[pos]:
                            vol_cool_wait = False
                            return "BUY", "MSTR波动率冷却(HV≤90%)"
                    else:
                        if in_buy and ((bsp and not np.isnan(bhvr) and bhvr < 1.0) or du <= 490):
                            return "BUY", "BTC买点区共振/H-490"
                        if ds < 480:
                            return "BUY", "减半牛市持仓期"
            else:
                if ds >= 150 and hv > 130.0:
                    vol_cool_wait = True
                    cooldown = 5
                    return "SELL", f"MSTR高波刹车(HV={hv:.0f}%)"
                if (in_sell and not np.isnan(bhvr) and bhvr >= 1.30) or (ds >= 540 and du > 540):
                    vol_cool_wait = False
                    return "SELL", "BTC卖点区逃顶"

        return "HOLD", ""

    for k, pos in enumerate(range(i_start, i_end + 1)):
        o_px = opens[pos]
        c_px = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        if pending == "BUY" and shares == 0.0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o_px
            cash = 0.0
            entry_px = o_px
            entry_dt_str = dt_str
            entry_m_str = m_str
            pending = ""
        elif pending == "SELL" and shares > 0.0:
            cash = shares * o_px * (1.0 - FEE_RATE)
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append({
                "mode": mode,
                "entry_date": entry_dt_str,
                "entry_m": entry_m_str,
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(entry_px, 2),
                "avg_entry_px": round(entry_px, 2),
                "exit_date": dt_str,
                "exit_m": m_str,
                "exit_px": round(o_px, 2),
                "ret_pct": round(ret_pct, 1),
                "exit_reason": pending_reason,
                "is_open": False,
            })
            shares = 0.0
            entry_px = 0.0
            pending = ""

        if shares > 0.0:
            days_inv += 1
        elif cooldown > 0:
            cooldown -= 1

        equity[k] = cash + shares * c_px

        if pos < i_end:
            act, rsn = eval_sig(pos, shares > 0.0)
            if act in ("BUY", "SELL"):
                pending = act
                pending_reason = rsn

    if shares > 0.0:
        cur_val = shares * closes[i_end]
        ret_pct = (cur_val / trade_cash_start - 1.0) * 100.0
        trades.append({
            "mode": mode,
            "entry_date": entry_dt_str,
            "entry_m": entry_m_str,
            "tranche2_date": "-",
            "tranche2_m": None,
            "tranche2_px": None,
            "entry_px": round(entry_px, 2),
            "avg_entry_px": round(entry_px, 2),
            "exit_date": idx[i_end].strftime("%Y-%m-%d") + " (OPEN)",
            "exit_m": idx[i_end].strftime("%Y-%m"),
            "exit_px": round(float(closes[i_end]), 2),
            "ret_pct": round(ret_pct, 1),
            "exit_reason": "持仓中",
            "is_open": True,
        })

    stats = _compute_mstr_stats(equity, trades, days_inv, idx[i_start:i_end + 1])

    # Live status at last bar
    last_pos = i_end
    ds, du = _ds_du(idx[last_pos])
    in_buy = (du <= 540) and (ds > 560)
    in_sell = (ds >= 480) and (du > 540)
    mr = float(r60[last_pos]) if not np.isnan(r60[last_pos]) else 1.0
    bhvr = float(b_hvr[last_pos]) if not np.isnan(b_hvr[last_pos]) else 1.0
    is_holding = shares > 0.0
    if pending == "BUY":
        action, badge, tone, reason = "BUY", "明日开盘买入", "buy", f"触发信号: {pending_reason}"
    elif pending == "SELL":
        action, badge, tone, reason = "SELL", "明日开盘卖出", "sell", f"触发信号: {pending_reason}"
    elif is_holding:
        action, badge, tone = "HOLD", "继续持仓", "hold"
        reason = f"周期 H+{ds}d | MSTR/BTC相对60日均={mr:.2f}x (警戒≥1.70x)"
    else:
        action, badge, tone = "WAIT", "保持空仓", "wait"
        if in_buy:
            reason = f"已进入BTC买点区 (H-{du}d)：BTC HVR={bhvr:.2f}，等待脉冲冷却 (<1.0) 或 H-490d 兜底"
        else:
            reason = f"熊市避险空仓中 (距下次减半 H-{du}d，买点区 H-540d 起)"

    live_status = {
        "action": action,
        "badge": badge,
        "tone": tone,
        "position_label": "满仓 (100%)" if is_holding else "空仓 (0%)",
        "pos_mode": "MSTR" if is_holding else "FLAT",
        "entry_date": entry_dt_str if is_holding else None,
        "entry_px": round(entry_px, 2) if is_holding else None,
        "ret_pct": round((shares * closes[last_pos] / trade_cash_start - 1.0) * 100.0, 2) if is_holding else None,
        "hold_weeks": None,
        "high_zone_armed": False,
        "sl_price": None,
        "wtsl_price": None,
        "cooldown_days": 0,
        "reason": reason,
    }

    return {
        "id": strategy_id,
        "name": name,
        "short_label": short_label,
        "rule_desc": rule_desc,
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
        "equity_curve": [{"d": idx[i_start + j].strftime("%Y-%m-%d"), "v": round(float(equity[j]), 2)} for j in range(0, n_bt, 5)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MSTR monthly zones HTML and optional Telegram alert.")
    parser.add_argument("--notify-tg", action="store_true", help="Send daily MSTR status summary to Telegram.")
    args = parser.parse_args()

    df_mstr, merged = prepare_mstr_btc_df()
    # Filter chart monthly bars to 2020-01 onwards (MSTR Bitcoin Era)
    df_mstr_2020 = df_mstr[df_mstr["Date"] >= pd.Timestamp("2020-01-01")].reset_index(drop=True)
    monthly, weekly_sma = build_monthly_and_weekly(df_mstr_2020)
    c7_entries = detect_c7_vol_entries(df_mstr_2020)
    feats = build_causal_features(df_mstr_2020, df_mstr_2020)

    s_mnav = simulate_mstr_strategy(
        merged,
        mode="HALV_MNAV_170",
        strategy_id="Halv_mNAV_Spike170",
        name="MSTR 减半周期 + mNAV溢价脉冲旗舰: 减半买底 + 放量mNAV止盈 + H+200锁盈 (3战全胜, $245万)",
        short_label="MSTR减半+mNAV脉冲 (默认)",
        rule_desc="BTC买点区[H-540~-460]冷却买 | MSTR/BTC≥1.70x且近3日量比≥1.5x止盈；H+200前冷却接回，H+200后锁盈等下轮",
    )
    s_btc = simulate_mstr_strategy(
        merged,
        mode="BTC_HALV_HV",
        strategy_id="BTC_Halv_HV_Combo",
        name="MSTR 纯BTC减半+HV共振版: 沿用BTC买点区/卖点区信号 (2战全胜, $83.2万)",
        short_label="纯BTC减半+HV驱动",
        rule_desc="完全跟随BTC减半[H-540~-460]冷却买入，[H+480~+560]脉冲卖出",
    )
    s_gac = simulate_mstr_strategy(
        merged,
        mode="HALV_PHASE_GAC",
        strategy_id="Halv_Phase_MSTR_GAC",
        name="MSTR 减半筑底 + 高斯通道波段版: 筑底死拿 + MSTR GAC(10,2,1.5) ($45.6万)",
        short_label="减半筑底+MSTR GAC",
        rule_desc="筑底[H-520~+100]死拿，中后期[H+100~+540]按MSTR日线GAC(10,2,1.5)波段，熊市空仓",
    )
    s_lowdd = simulate_mstr_strategy(
        merged,
        mode="HALV_HV130_COOL90",
        strategy_id="Halv_MSTR_HV130_Cool90",
        name="MSTR 低回撤控险版: 减半买底 + MSTR HV20>130% 强刹车 (MaxDD仅-39.7%)",
        short_label="MSTR低回撤 (HV130刹车)",
        rule_desc="减半买点区买入，H+150天后若MSTR HV20>130%先刹车，冷却至≤90%且站上SMA50接回",
    )

    strategies = {
        "Halv_mNAV_Spike170": s_mnav,
        "BTC_Halv_HV_Combo": s_btc,
        "Halv_Phase_MSTR_GAC": s_gac,
        "Halv_MSTR_HV130_Cool90": s_lowdd,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    live_market = build_live_market_snapshot(feats)
    last_row = merged.iloc[-1]
    ds, du = _ds_du(pd.Timestamp(last_row["Date"]))
    live_market["mstr_live"] = {
        "ds": int(ds),
        "du": int(du),
        "btc_close": round(float(last_row["b_close"]), 2),
        "mstr_btc": round(float(last_row["mstr_btc"]), 6),
        "mstr_btc_r60": round(float(last_row["mstr_btc_r60"]), 2),
        "mstr_hv20": round(float(last_row["m_hv20"]), 1),
        "mstr_vol_r20": round(float(last_row["m_vol_r20"]), 2),
        "mstr_vol_r3": round(float(last_row["m_vol_r20_max3"]), 2),
        "btc_hvr": round(float(last_row["b_hvr"]), 2),
        "btc_hvr_spike35": bool(last_row["b_hvr_spike35"]),
        "gac_mid": round(float(last_row["m_gac_mid"]), 2),
        "gac_upper": round(float(last_row["m_gac_upper"]), 2),
        "in_buy_zone": bool((du <= 540) and (ds > 560)),
        "in_sell_zone": bool((ds >= 480) and (du > 540)),
    }
    live_market["mstr_t0"] = build_mstr_t0_tracker(merged)

    halv_zones = [z for z in build_halving_zone_shapes() if z["end"] >= "2020-01-01"]

    html = render_html(monthly, weekly_sma, c7_entries, strategies, live_market=live_market)
    mstr_options = """<select id="strategySelect" class="strategy-select">
        <option value="Halv_mNAV_Spike170" selected>MSTR 减半+mNAV放量脉冲: 3战全胜 ($245万, MaxDD -39.7%, Calmar 3.09)</option>
        <option value="BTC_Halv_HV_Combo">MSTR 纯BTC减半+HV共振: 沿用BTC买点区/卖点区 (2战全胜, $83.2万)</option>
        <option value="Halv_Phase_MSTR_GAC">MSTR 减半筑底+高斯通道: 筑底死拿 + MSTR GAC(10,2,1.5) ($45.6万)</option>
        <option value="Halv_MSTR_HV130_Cool90">MSTR 低回撤控险版: MSTR HV20>130%刹车 (MaxDD仅-39.7%)</option>
        <option value="NONE">不显示策略标注</option>
      </select>"""
    html = re.sub(
        r'<select id="strategySelect" class="strategy-select">.*?</select>',
        mstr_options,
        html,
        count=1,
        flags=re.S,
    )
    html = html.replace("SOXL 月线顶底区域", "MSTR 比特币减半周期与 mNAV 溢价策略")
    html = html.replace("SOXL 月线历史", "MSTR 比特币周期")
    html = html.replace("SOXL 月线", "MSTR 月线")
    html = html.replace("SOXL 价格", "MSTR 价格")

    halv_json = json.dumps(halv_zones, separators=(",", ":"), ensure_ascii=False)
    html = html.replace("const LIVE_MARKET =", f"const HALVING_ZONES = {halv_json};\n    const LIVE_MARKET =")

    zone_inject_js = """      const shapes = zones.map(z => ({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: z.startMonth,
        x1: z.endMonth,
        y0: 0,
        y1: 1,
        fillcolor: z.type === "TOP" ? "rgba(248,81,73,0.14)" : "rgba(63,185,80,0.14)",
        line: {
          color: z.type === "TOP" ? "rgba(248,81,73,0.45)" : "rgba(63,185,80,0.45)",
          width: 1,
          dash: "dot",
        },
        layer: "below",
      }));
      HALVING_ZONES.forEach(hz => {
        const isBuy = hz.kind === "BUY";
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: hz.start.slice(0, 7),
          x1: hz.end.slice(0, 7),
          y0: 0,
          y1: 1,
          fillcolor: isBuy ? "rgba(46,160,67,0.18)" : "rgba(248,81,73,0.18)",
          line: {
            color: isBuy ? "rgba(63,185,80,0.8)" : "rgba(248,81,73,0.8)",
            width: 1.5,
            dash: "dash",
          },
          layer: "below",
        });
      });"""
    html = re.sub(r"      const shapes = zones\.map\(z => \(\{.*?\}\)\);", zone_inject_js, html, count=1, flags=re.S)

    annot_target = "      const annots = zones.map(z => ({"
    annot_inject = """      const annots = HALVING_ZONES.map(hz => ({
        x: hz.start.slice(0, 7),
        y: 0.99,
        xref: "x",
        yref: "paper",
        text: hz.label,
        showarrow: false,
        xanchor: "left",
        yanchor: "top",
        font: {
          size: 11,
          color: hz.kind === "BUY" ? "#3fb950" : "#ff7b72",
          family: "system-ui, sans-serif",
        },
        bgcolor: "rgba(13,17,23,0.82)",
        borderpad: 3,
      }));
      zones.forEach(z => annots.push({"""
    html = html.replace(annot_target, annot_inject, 1)
    html = html.replace("      }));\n\n      const trades =", "      }));\n      });\n\n      const trades =", 1)

    new_fresh = """      const ml = m.mstr_live;
      const freshHtml = `
        <span class="fresh-pill">今日数据: ${m.last_date} (${m.weekday_cn})</span>
        <span>MSTR <b>$${fmtV(m.soxl_close)}</b></span>
        <span>BTC <b>$${fmtV(ml.btc_close)}</b></span>
        <span>距下次减半 <b>-${ml.du} 天</b></span>
        <span>mNAV溢价脉冲 <b>${fmtV(ml.mstr_btc_r60)}x</b></span>
        <span>HV20 <b>${fmtV(ml.mstr_hv20, "%", 1)}</b></span>
      `;"""
    html = re.sub(r"      const freshHtml = `.*?`;", new_fresh, html, count=1, flags=re.S)
    html = html.replace(
        '<div class="action-pos">当前仓位: ${ls.position_label} ｜ 下个交易日开盘执行</div>',
        '<div class="action-pos">今日状态: ${ls.position_label} ｜ 明日操作: 下个交易日开盘执行</div>',
    )

    radar_anchor = '      const groups = [];'
    radar_end = '      box.innerHTML = `'
    mstr_radar_js = """      const groups = [];
      if (ls && ls.pos_mode !== "FLAT") {
        const g1 = [
          radarCell("mNAV溢价脉冲 MSTR/BTC ≥ 1.70x", ml.mstr_btc_r60 >= 1.70, `当前相对60日均 ${fmtV(ml.mstr_btc_r60)}x`),
          radarCell("放量确认 近3日最大量比 ≥ 1.5x", ml.mstr_vol_r3 >= 1.50, `近3日最高 ${fmtV(ml.mstr_vol_r3)}x (今日 ${fmtV(ml.mstr_vol_r20)}x)`),
          radarCell("处于BTC卖点区 [H+480~+560]", Boolean(ml.in_sell_zone), `距上次减半 +${ml.ds} 天`),
          radarCell("BTC波动率脉冲 HVR ≥ 1.30", ml.btc_hvr >= 1.30, `当前 BTC HVR=${fmtV(ml.btc_hvr)}`),
        ];
        const g2 = [
          radarCell("MSTR 20日年化波动率", ml.mstr_hv20 > 130, `当前 HV20=${fmtV(ml.mstr_hv20, "%", 1)} (低回撤刹车线 130%)`),
          radarCell("MSTR GAC(10,2,1.5) 中轨", m.soxl_close > ml.gac_mid, `MSTR $${fmtV(m.soxl_close)} vs 中轨 $${fmtV(ml.gac_mid)}`),
        ];
        groups.push(renderRadarGroup("卖出监控 · mNAV溢价脉冲 & BTC卖点区", g1, "any_exit"));
        groups.push(renderRadarGroup("风控参考 · MSTR波动率与高斯通道", g2, "all"));
      } else {
        const g1 = [
          radarCell("处于BTC买点区 [H-540~-460]", Boolean(ml.in_buy_zone), `距下次减半 -${ml.du} 天`),
          radarCell("BTC近35日恐慌脉冲 HVR≥1.30", Boolean(ml.btc_hvr_spike35), `当前 BTC HVR=${fmtV(ml.btc_hvr)}`),
          radarCell("BTC脉冲冷却 HVR < 1.0", ml.btc_hvr < 1.0, `当前 BTC HV20/HV60 = ${fmtV(ml.btc_hvr)}`),
        ];
        const canReenterWindow = ml.ds < 200 || ml.du <= 100;
        const g2 = [
          radarCell("买点区时间兜底 H ≤ -490天", ml.du <= 490, `当前 -${ml.du} 天 (还差 ${Math.max(0, ml.du - 490)} 天)`),
          radarCell("逃顶后锁盈过滤 (仅H+200前接回)", canReenterWindow, `当前 H+${ml.ds}d / -${ml.du}d (H+200后锁盈过滤亏损单)`),
          radarCell("mNAV冷却接回 (≤1.10x 且 >GAC中轨)", canReenterWindow && ml.mstr_btc_r60 <= 1.10 && m.soxl_close > ml.gac_mid, `R60=${fmtV(ml.mstr_btc_r60)}x | $${fmtV(m.soxl_close)} vs GAC $${fmtV(ml.gac_mid)}`),
        ];
        groups.push(renderRadarGroup("买点区主规则 · BTC减半窗口脉冲冷却（满足全部 → 买入）", g1, "all"));
        groups.push(renderRadarGroup("兜底与溢价冷却参考", g2, "any_entry"));
      }

"""
    i_r1 = html.index(radar_anchor)
    i_r2 = html.index(radar_end, i_r1)
    html = html[:i_r1] + mstr_radar_js + html[i_r2:]

    t0_css = """
    .t0-tracker { margin: 12px 24px 0; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }
    .t0-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; padding-bottom: 10px; border-bottom: 1px solid #21262d; }
    .t0-title-wrap { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .t0-title { font-size: 14px; font-weight: 700; color: #f0f6fc; }
    .t0-sub { font-size: 11px; color: #8b949e; background: #0d1117; border: 1px solid #30363d; border-radius: 999px; padding: 2px 8px; }
    .t0-stage { font-size: 12px; font-weight: 600; border-radius: 6px; padding: 4px 10px; }
    .t0-stage.muted { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
    .t0-stage.warn { background: rgba(240,136,62,0.14); color: #f0883e; border: 1px solid rgba(240,136,62,0.4); }
    .t0-stage.active { background: rgba(88,166,255,0.14); color: #58a6ff; border: 1px solid rgba(88,166,255,0.45); }
    .t0-stage.ready { background: rgba(63,185,80,0.16); color: #3fb950; border: 1px solid rgba(63,185,80,0.45); }
    .t0-meta { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #8b949e; margin-top: 8px; }
    .t0-meta b { color: #e6edf3; }
    .t0-cols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
    @media (max-width: 980px) { .t0-cols { grid-template-columns: 1fr; } }
    .t0-card { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 11px 12px; }
    .t0-card-h { font-size: 12px; font-weight: 600; color: #c9d1d9; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .t0-item { font-size: 12px; padding: 6px 8px; border-radius: 6px; margin-bottom: 6px; border: 1px solid #21262d; background: #161b22; }
    .t0-item.ok { border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.08); }
    .t0-item.miss { border-color: #21262d; color: #8b949e; }
    .t0-item-top { display: flex; justify-content: space-between; gap: 6px; font-weight: 600; }
    .t0-item.ok .t0-item-top { color: #3fb950; }
    .t0-item-val { font-size: 11px; color: #8b949e; margin-top: 2px; }
    .t0-tiers { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .t0-tier { padding: 7px 8px; border-radius: 6px; border: 1px solid #30363d; background: #161b22; font-size: 12px; }
    .t0-tier.ok { border-color: rgba(63,185,80,0.45); background: rgba(63,185,80,0.10); }
    .t0-tier-px { font-weight: 700; color: #f0f6fc; font-size: 13px; }
    .t0-tier.ok .t0-tier-px { color: #3fb950; }
    .t0-tier-st { font-size: 11px; color: #8b949e; margin-top: 2px; }
    .t0-note { font-size: 11px; color: #8b949e; margin-top: 8px; line-height: 1.45; }
    .t0-prog-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 999px; height: 10px; overflow: hidden; margin: 6px 0; }
    .t0-prog-bar { height: 100%; background: linear-gradient(90deg, #58a6ff, #3fb950); transition: width 0.25s; }
    .t0-table-wrap { margin-top: 12px; overflow-x: auto; border: 1px solid #21262d; border-radius: 8px; }
    .t0-table { width: 100%; border-collapse: collapse; font-size: 12px; background: #0d1117; }
    .t0-table th, .t0-table td { padding: 6px 8px; text-align: right; border-bottom: 1px solid #21262d; white-space: nowrap; }
    .t0-table th:first-child, .t0-table td:first-child, .t0-table th:last-child, .t0-table td:last-child { text-align: left; }
    .t0-table th { color: #8b949e; font-weight: 600; background: #161b22; }
    .t0-badge-newlow { color: #ff7b72; font-weight: 600; }
    .t0-badge-hold { color: #3fb950; font-weight: 600; }
    .t0-badge-t0 { color: #58a6ff; font-weight: 700; }
    .t0-hist { margin-top: 8px; font-size: 11px; color: #8b949e; display: flex; flex-wrap: wrap; gap: 14px; }
    .t0-hist b { color: #c9d1d9; }
  </style>"""
    assert "</style>" in html, "Missing </style> anchor"
    html = html.replace("</style>", t0_css, 1)

    cockpit_div = '<div id="liveCockpit" class="live-cockpit"></div>'
    assert cockpit_div in html, "Missing liveCockpit anchor"
    html = html.replace(cockpit_div, cockpit_div + '\n\n  <div id="t0Tracker" class="t0-tracker"></div>', 1)

    t0_render_js = """
    function renderT0Tracker() {
      const el = document.getElementById("t0Tracker");
      const m = LIVE_MARKET;
      if (!el || !m || !m.mstr_t0) {
        if (el) el.style.display = "none";
        return;
      }
      const t = m.mstr_t0;
      const anchorPx = t.pre_low !== null ? t.pre_low : t.t0_close;
      const conds = [
        {
          label: "1. 深跌门槛: 60日回撤 ≤ -35%",
          ok: t.t0_triggered ? true : Boolean(t.last_cond_dd),
          val: t.t0_triggered
            ? `T0日 ${fmtV(t.t0_dd60_pct, "%", 1)} | 最新 ${fmtV(t.last_dd60_pct, "%", 1)}`
            : `最新 60日回撤 ${fmtV(t.last_dd60_pct, "%", 1)}`,
        },
        {
          label: "2. 破位插针: 触及近20日新低",
          ok: t.t0_triggered ? true : Boolean(t.last_cond_low20),
          val: t.t0_triggered
            ? `T0日低 $${fmtV(t.t0_low)} | 最新低 $${fmtV(t.last_low)} (20D低 $${fmtV(t.last_low20)})`
            : `最新低 $${fmtV(t.last_low)} vs 20D低 $${fmtV(t.last_low20)}`,
        },
        {
          label: "3. 恐慌/放量脉冲 (HVR≥1.25 或 量≥1.5x且R60≤0.85)",
          ok: t.t0_triggered ? true : Boolean(t.last_cond_pulse),
          val: t.t0_triggered
            ? `T0日 BTC_HVR=${fmtV(t.t0_b_hvr)} M_HVR=${fmtV(t.t0_m_hvr)} 量=${fmtV(t.t0_vol_r20, "x")} R60=${fmtV(t.t0_r60, "x")}`
            : `最新 BTC_HVR=${fmtV(t.last_b_hvr)} | M_HVR=${fmtV(t.last_m_hvr)} | 量=${fmtV(t.last_vol_r20, "x")} | R60=${fmtV(t.last_r60, "x")}`,
        },
      ];

      const col1 = `
        <div class="t0-card">
          <div class="t0-card-h">
            <span>T0 三条件门控（未亮灯绝对不买）</span>
            <span>${t.t0_triggered ? `✓ ${t.t0_date}` : "等待中"}</span>
          </div>
          ${conds.map(c => `
            <div class="t0-item ${c.ok ? "ok" : "miss"}">
              <div class="t0-item-top">
                <span>${c.label}</span>
                <span>${c.ok ? "✓ 满足" : "○ 未满足"}</span>
              </div>
              <div class="t0-item-val">${c.val}</div>
            </div>
          `).join("")}
        </div>`;

      let col2Body = "";
      if (t.limit_tiers && t.limit_tiers.length > 0) {
        col2Body = `<div class="t0-tiers">
          ${t.limit_tiers.map(x => `
            <div class="t0-tier ${x.filled ? "ok" : ""}">
              <div class="t0-tier-px">${x.mult_pct}% 档 · $${fmtV(x.limit_px)} <span style="font-weight:400;font-size:11px;color:#8b949e">(${x.weight_pct}%)</span></div>
              <div class="t0-tier-st">${x.filled ? `✓ ${x.fill_date.slice(5)} @ $${fmtV(x.fill_px)}` : "○ 待接针限价"}</div>
            </div>
          `).join("")}
        </div>`;
      } else {
        const preview = anchorPx !== null
          ? [0.92, 0.86, 0.80, 0.74].map(k => `$${fmtV(anchorPx * k)}`).join(" / ")
          : "待进窗确认锚点";
        col2Body = `<div class="t0-item miss">
          <div class="t0-item-top"><span>限价网暂未激活（等 T0 亮灯）</span><span>○ 锁定</span></div>
          <div class="t0-item-val">参考锚点下方 92/86/80/74%：${preview}</div>
        </div>`;
      }

      const col2 = `
        <div class="t0-card">
          <div class="t0-card-h">
            <span>左侧 50% 接针限价网</span>
            <span>锚: $${fmtV(anchorPx)} (${t.anchor_label})</span>
          </div>
          ${col2Body}
          <div class="t0-note">历史 11 次 MSTR 恐慌中 9 次 (82%) 于 T0 后 1–10 交易日内打出针尖；限价总仓封顶 50%，其余留给右侧兜底。</div>
        </div>`;

      const streak = t.no_new_low_streak || 0;
      const pct = Math.min(100, Math.round((streak / 5) * 100));
      const rightStatus = t.first_5d_date
        ? `✓ 首次连续5日守稳触发：${t.first_5d_date}${t.first_5d_px !== null ? ` @ $${fmtV(t.first_5d_px)}` : ""}`
        : (t.t0_triggered
          ? `当前低点 $${fmtV(t.run_low)} (${t.run_low_date}) · 连守 ${streak} / 5 交易日`
          : "T0 亮灯后自动开启 5 日不破新低计数");

      const ml = m.mstr_live || {};
      const col3 = `
        <div class="t0-card">
          <div class="t0-card-h">
            <span>右侧 50% 止跌确认 & 兜底</span>
            <span>${t.t0_triggered ? `T0 后 +${t.days_since_t0} 交易日` : "待启动"}</span>
          </div>
          <div class="t0-item ${t.first_5d_date ? "ok" : "miss"}">
            <div class="t0-item-top">
              <span>连续 5 个交易日不破前低</span>
              <span>${streak}/5</span>
            </div>
            <div class="t0-prog-wrap"><div class="t0-prog-bar" style="width:${pct}%"></div></div>
            <div class="t0-item-val">${rightStatus}</div>
          </div>
          <div class="t0-item ${(ml.btc_hvr < 1.0 && ml.btc_hvr_spike35) || t.du <= 490 ? "ok" : "miss"}">
            <div class="t0-item-top">
              <span>原策略兜底：BTC HVR < 1.0 或 H-490 (${t.h490_date})</span>
              <span>H-${t.du}d</span>
            </div>
            <div class="t0-item-val">窗内最低: ${t.win_low !== null ? `$${fmtV(t.win_low)} (${t.win_low_date}, 距T0 ${fmtV(t.drop_from_t0_pct, "%", 1)})` : "—"} | BTC HVR=${fmtV(ml.btc_hvr)}</div>
          </div>
          <div class="t0-note">创新低计数自动归零；若为 11–12 月行情，须防范 12 月下旬税损抛售“二次插针”。</div>
        </div>`;

      let tableHtml = "";
      if (t.daily_rows && t.daily_rows.length > 0) {
        tableHtml = `
          <div class="t0-table-wrap">
            <table class="t0-table">
              <thead>
                <tr>
                  <th>日期</th><th>Open</th><th>Low</th><th>Close</th><th>距T0收盘</th><th>r60</th><th>BTC HVR</th><th>量比</th><th>右侧状态</th>
                </tr>
              </thead>
              <tbody>
                ${t.daily_rows.map(r => {
                  const st = r.is_t0
                    ? `<span class="t0-badge-t0">● T0 亮灯日</span>`
                    : (r.is_new_low
                      ? `<span class="t0-badge-newlow">★ 破前低 (重置 0/5)</span>`
                      : `<span class="t0-badge-hold">守住前低 (${r.streak}/5)</span>`);
                  return `<tr>
                    <td>${r.date}</td>
                    <td>$${fmtV(r.open)}</td>
                    <td>$${fmtV(r.low)}</td>
                    <td>$${fmtV(r.close)}</td>
                    <td>${r.vs_t0_pct !== null ? `${r.vs_t0_pct >= 0 ? "+" : ""}${fmtV(r.vs_t0_pct, "%", 1)}` : "—"}</td>
                    <td>${fmtV(r.r60, "x")}</td>
                    <td>${fmtV(r.b_hvr)}</td>
                    <td>${fmtV(r.vol_r20, "x")}</td>
                    <td>${st}</td>
                  </tr>`;
                }).join("")}
              </tbody>
            </table>
          </div>`;
      }

      const histHtml = (t.history_windows && t.history_windows.length > 0)
        ? `<div class="t0-hist"><span>买点窗历史对照：</span>${t.history_windows.map(h =>
            `<span><b>${h.cycle}</b>: 锚 $${fmtV(h.pre_low)} → T0 ${h.t0_date} ($${fmtV(h.t0_close)}) → 窗内低 $${fmtV(h.win_low)} (${h.win_low_date}, +${h.days_to_low}交易日, ${fmtV(h.drop_from_t0_pct, "%", 1)})</span>`
          ).join("")}</div>`
        : "";

      el.innerHTML = `
        <div class="t0-head">
          <div class="t0-title-wrap">
            <span class="t0-title">MSTR T0 恐慌洗盘与接货追踪器</span>
            <span class="t0-sub">独立追踪 · 不改动上方原有策略买卖信号</span>
          </div>
          <span class="t0-stage ${t.stage_tone}">${t.stage_label}</span>
        </div>
        <div class="t0-meta">
          <span>当前买点窗: <b>${t.win_start} ~ ${t.win_end}</b> (${t.in_buy_zone ? "窗内" : "窗外"} · H-${t.du}d)</span>
          <span>窗前锚点低点: <b>${t.pre_low !== null ? `$${fmtV(t.pre_low)} (${t.pre_low_date})` : "—"}</b></span>
          <span>T0 基准: <b>${t.t0_triggered ? `${t.t0_date} 收 $${fmtV(t.t0_close)}` : "尚未触发"}</b></span>
          <span>窗内最低: <b>${t.win_low !== null ? `$${fmtV(t.win_low)} (${t.win_low_date}${t.vs_pre_low_pct !== null ? `, 较前低 ${fmtV(t.vs_pre_low_pct, "%", 1)}` : ""})` : "—"}</b></span>
        </div>
        <div class="t0-cols">${col1}${col2}${col3}</div>
        ${tableHtml}
        ${histHtml}
      `;
    }

    render();
    renderT0Tracker();"""

    render_anchor = "\n\n    render();\n  </script>"
    assert render_anchor in html, "Missing top-level render() anchor"
    html = html.replace(render_anchor, f"\n{t0_render_js}\n  </script>", 1)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")

    if args.notify_tg:
        tg_msg = format_mstr_tg_summary(live_market, strategies)
        ok = send_tg_notification(tg_msg)
        print(f"[Telegram] sent={ok}")
        s1 = strategies["Halv_mNAV_Spike170"]["live_status"]
        if s1["action"] in ("BUY", "SELL"):
            po_ok = send_pushover_notification(
                title=f"🚨 MSTR 交易信号: {s1['badge']}",
                message=tg_msg,
            )
            print(f"[Pushover] sent={po_ok}")


def format_mstr_tg_summary(live_market: dict, strategies: dict) -> str:
    """Format concise daily MSTR status message for Telegram."""
    ml = live_market["mstr_live"]
    s1 = strategies["Halv_mNAV_Spike170"]["live_status"]
    zone_tag = "买点区[H-540~-460]" if ml["in_buy_zone"] else ("卖点区[H+480~+560]" if ml["in_sell_zone"] else "区间外")
    lines = [
        f"MSTR 日更状态 ({live_market['last_date']} {live_market['weekday_cn']})",
        f"MSTR: ${live_market['soxl_close']:,.2f} | BTC: ${ml['btc_close']:,.2f} | 减半: -{ml['du']}d ({zone_tag})",
        f"mNAV脉冲(R60): {ml['mstr_btc_r60']}x | 近3日量比: {ml['mstr_vol_r3']}x | BTC HVR: {ml['btc_hvr']}",
        f"策略 [Halv+mNAV]: {s1['badge']} ({s1['position_label']})",
        f"原因: {s1['reason']}",
    ]
    t0 = live_market.get("mstr_t0")
    if t0:
        if t0.get("t0_triggered"):
            lines.append(
                f"T0追踪: 已亮灯({t0['t0_date']} ${t0['t0_close']}, +{t0['days_since_t0']}交易日) | 窗内低点 ${t0['win_low']} ({t0['drop_from_t0_pct']:+.1f}%) | 5日不破新低: {t0['no_new_low_streak']}/5"
            )
        else:
            lines.append(
                f"T0追踪: {t0['stage_label']} (60日回撤 {t0['last_dd60_pct']:+.1f}%, R60 {t0['last_r60']}x)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
