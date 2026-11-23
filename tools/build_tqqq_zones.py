"""Build a dedicated TQQQ monthly LOG chart with top/bottom zones and TQQQ-native strategy overlays."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import sma
from tools.build_soxl_zones import (
    BT_START,
    DATA_DIR,
    FEE_RATE,
    INITIAL_CAPITAL,
    _compute_stats,
    build_causal_features,
    build_live_market_snapshot,
    build_monthly_and_weekly,
    detect_c7_vol_entries,
    load_daily_adjusted,
    render_html,
    simulate_d1_x1a_wtsl35,
    simulate_h5_adapt_hv,
    simulate_tp3h,
)

TQQQ_PATH = DATA_DIR / "TQQQ.csv"
QQQ_PATH = DATA_DIR / "QQQ.csv"
SOXL_PATH = DATA_DIR / "SOXL.csv"
SMH_PATH = DATA_DIR / "SMH.csv"
OUTPUT_HTML = ROOT / "tqqq_monthly_zones.html"


def simulate_tqqq_dual(
    feats: dict,
    df_qqq: pd.DataFrame,
    ma_period: int = 200,
    band: float = 0.01,
    hv_exit: float = 100.0,
    hv_reentry: float = 80.0,
    tsl_pct: float | None = 0.25,
    strategy_id: str = "TQQQ_Dual_SMA200_1%_HV100_TSL25",
    name: str = "TQQQ 专属双引擎 (±1%): QQQ 200SMA + HV100/80 + TSL25%",
    short_label: str = "TQQQ双引擎 (200SMA±1%+TSL25%)",
    start_date: str = BT_START,
    end_date: str | None = None,
) -> dict:
    """Simulate TQQQ-native Dual-Engine strategy driven by QQQ trend + TQQQ HV20 + TSL + BOT/ALT."""
    d = feats["daily"]
    w = feats["weekly"]
    wed = feats["week_end_daily_pos"]
    idx = d.index

    if end_date is None:
        end_date = str(idx.max().date())

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    hv20 = d["HV20"].to_numpy(dtype=np.float64)

    bench = df_qqq.set_index("Date").reindex(idx, method="ffill")
    b_c = bench["Close"].to_numpy(dtype=np.float64)
    ma_arr = sma(bench["Close"], ma_period).to_numpy(dtype=np.float64)

    w_bot = (
        ((1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min() <= 1.15)
        & (w["W_Fisher"].rolling(4, min_periods=1).min() < -1.5)
        & (w["W_Vol_R"].rolling(4, min_periods=1).max() >= 1.5)
        & (w["Close"] > w["High"].shift(1))
    ).to_numpy(dtype=bool)
    w_alt = ((w["W_Fisher"] < -1.8) & ((w["dW_Fish"] > 0) | (w["W_Vol_R"] > 1.3))).to_numpy(dtype=bool)
    w_c = w["Close"].to_numpy(dtype=np.float64)
    w_mf = w["M_Fish_prev"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    trade_cash_start = INITIAL_CAPITAL
    entry_px = 0.0
    pos_mode = 0  # 0=FLAT, 1=TREND, 2=ALT, 3=BOT
    initial_mode_str = ""
    promoted_in_trade = False
    entry_idx = -1
    exit_idx = -9999
    last_exit = "INIT"
    peak_d = 0.0
    pending = 0
    pending_reason = ""

    days_inv = 0
    trades: list[dict] = []
    equity = np.empty(n_bt, dtype=np.float64)

    for step, i in enumerate(range(i_start, i_end + 1)):
        o = opens[i]
        c = closes[i]
        just_ex = False

        if pending > 0 and pos_mode == 0:
            trade_cash_start = cash
            shares = cash / (o * (1.0 + FEE_RATE))
            cash = 0.0
            entry_px = float(o)
            pos_mode = pending
            initial_mode_str = {1: "TREND", 2: "ALT", 3: "BOT"}[pos_mode]
            promoted_in_trade = False
            entry_idx = i
            peak_d = c
        elif pending == -1 and pos_mode != 0:
            cash = shares * o * (1.0 - FEE_RATE)
            shares = 0.0
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
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
            pos_mode = 0
            exit_idx = i
            last_exit = pending_reason
            just_ex = True

        pending = 0
        pending_reason = ""

        equity[step] = cash + shares * c
        if pos_mode != 0:
            days_inv += 1
            if c > peak_d:
                peak_d = c

        if i >= i_end:
            continue

        is_w = i in wed
        wp = wed[i] if is_w else -1
        tb_raw = (not np.isnan(ma_arr[i])) and (b_c[i] > ma_arr[i] * (1.0 + band))
        ma_brk = (not np.isnan(ma_arr[i])) and (b_c[i] < ma_arr[i] * (1.0 - band))
        hv = hv20[i]
        vol_ex = (not np.isnan(hv)) and (hv > hv_exit)
        tb_ok = tb_raw and (not np.isnan(hv)) and (hv <= hv_exit * 0.92)
        tb_re = tb_raw and (not np.isnan(hv)) and (hv <= hv_reentry) and (not vol_ex)

        if pos_mode == 3:
            if is_w and ((np.isnan(w_mf[wp]) or w_mf[wp] < 1.5) and peak_d < entry_px * 1.5) and w_c[wp] < entry_px * 0.75:
                pending, pending_reason = -1, "SL_W25"
            elif tb_raw and (not vol_ex) and (not ma_brk):
                pos_mode = 1
                promoted_in_trade = True
        elif pos_mode == 1:
            if vol_ex:
                pending, pending_reason = -1, "VOL_SPIKE"
            elif ma_brk:
                pending, pending_reason = -1, "TREND_BREAK"
            elif tsl_pct is not None and c <= peak_d * (1.0 - tsl_pct):
                pending, pending_reason = -1, f"TSL{int(tsl_pct * 100)}%"
        elif pos_mode == 2:
            if c <= entry_px * 0.80:
                pending, pending_reason = -1, "ALT_FSL"
            elif c >= entry_px * 1.25 or (i - entry_idx) >= 40:
                pending, pending_reason = -1, "ALT_TP"
        elif not just_ex:
            el = i - exit_idx
            cool_ok = (el >= 10) if last_exit.startswith(("SL_W25", "ALT_", "TSL")) else True
            if last_exit == "SL_W25":
                if el >= 10 and is_w and w_bot[wp]:
                    pending = 3
            elif cool_ok:
                if last_exit == "VOL_SPIKE" and tb_re:
                    pending = 1
                elif last_exit != "VOL_SPIKE" and tb_ok:
                    pending = 1
                elif is_w and w_bot[wp]:
                    pending = 3
                elif is_w and w_alt[wp]:
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

    # Causal live action evaluation at i_end
    hv_live = hv20[i_end]
    bc_live = b_c[i_end]
    ma_live = ma_arr[i_end]
    c_live = closes[i_end]
    is_w_live = i_end in wed
    wp_live = wed[i_end] if is_w_live else (len(w) - 1)
    tb_raw_live = (not np.isnan(ma_live)) and (bc_live > ma_live * (1.0 + band))
    ma_brk_live = (not np.isnan(ma_live)) and (bc_live < ma_live * (1.0 - band))
    vol_ex_live = (not np.isnan(hv_live)) and (hv_live > hv_exit)
    tb_ok_live = tb_raw_live and (not np.isnan(hv_live)) and (hv_live <= hv_exit * 0.92)
    tb_re_live = tb_raw_live and (not np.isnan(hv_live)) and (hv_live <= hv_reentry) and (not vol_ex_live)

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
            if ((np.isnan(w_mf[wp_live]) or w_mf[wp_live] < 1.5) and peak_d < entry_px * 1.5) and w_c[wp_live] < entry_px * 0.75:
                live_act, live_badge, live_tone, live_reason = "SELL", "卖出 (假底止损)", "sell", "触发 SL_W25"
            elif tb_raw_live and (not vol_ex_live) and (not ma_brk_live):
                live_reason = f"QQQ > {ma_period}SMA×{1+band:.2f}，BOT 晋升为 TREND 趋势仓"
        elif pos_mode == 1:
            if vol_ex_live:
                live_act, live_badge, live_tone, live_reason = "SELL", "卖出 (高波刹车)", "sell", f"TQQQ HV20={hv_live:.1f}% > {hv_exit:.0f}%"
            elif ma_brk_live:
                live_act, live_badge, live_tone, live_reason = "SELL", "卖出 (趋势跌破)", "sell", f"QQQ {bc_live:.2f} < {ma_period}SMA×{1-band:.2f} ({ma_live*(1-band):.2f})"
            elif tsl_pct is not None and c_live <= peak_d * (1.0 - tsl_pct):
                live_act, live_badge, live_tone, live_reason = "SELL", f"卖出 (回撤{int(tsl_pct*100)}%止损)", "sell", f"自持仓峰值 ${peak_d:.2f} 回撤 ≥ {int(tsl_pct*100)}%"
        elif pos_mode == 2:
            if c_live <= entry_px * 0.80:
                live_act, live_badge, live_tone, live_reason = "SELL", "卖出 (ALT止损-20%)", "sell", "触发 -20% 硬止损"
            elif c_live >= entry_px * 1.25 or (i_end - entry_idx) >= 40:
                live_act, live_badge, live_tone, live_reason = "SELL", "卖出 (ALT止盈/超时)", "sell", "触发 +25% 止盈或持仓满40日"
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
            "sl_price": round(peak_d * (1.0 - tsl_pct), 2) if (pos_mode == 1 and tsl_pct is not None) else (round(entry_px * 0.75, 2) if pos_mode == 3 else None),
            "wtsl_price": None,
            "cooldown_days": 0,
            "reason": live_reason,
        }
    else:
        el = i_end - exit_idx
        cool_rem = max(0, 10 - el) if last_exit.startswith(("SL_W25", "ALT_", "TSL")) else 0
        pending_live = 0
        live_reason = f"空仓观望（上次离场: {last_exit}）"
        if not just_ex:
            cool_ok = (el >= 10) if last_exit.startswith(("SL_W25", "ALT_", "TSL")) else True
            if last_exit == "SL_W25":
                if el >= 10 and w_bot[wp_live]:
                    pending_live = 3
            elif cool_ok:
                if last_exit == "VOL_SPIKE" and tb_re_live:
                    pending_live = 1
                elif last_exit != "VOL_SPIKE" and tb_ok_live:
                    pending_live = 1
                elif w_bot[wp_live]:
                    pending_live = 3
                elif w_alt[wp_live]:
                    pending_live = 2
        if pending_live == 1:
            live_act, live_badge, live_tone, live_reason = "BUY", "买入 100% (TREND趋势)", "buy", f"QQQ > {ma_period}SMA×{1+band:.2f} 且 HV20={hv_live:.1f}% 达标"
        elif pending_live == 3:
            live_act, live_badge, live_tone, live_reason = "BUY", "买入 100% (BOT大底)", "buy", "满足 TQQQ 大底早买信号"
        elif pending_live == 2:
            live_act, live_badge, live_tone, live_reason = "BUY", "买入 100% (ALT恐慌底)", "buy", "满足周线 Fisher < -1.8 恐慌抄底信号"
        else:
            live_act, live_badge, live_tone = "WAIT", "保持空仓", "wait"
            if cool_rem > 0:
                live_reason = f"冷静期剩余 {cool_rem} 个交易日（上次: {last_exit}）"
            elif last_exit == "VOL_SPIKE":
                live_reason = f"高波离场后需等 HV20 ≤ {hv_reentry:.0f}%（当前 {hv_live:.1f}%）"
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
            "cooldown_days": cool_rem,
            "reason": live_reason,
        }

    stats = _compute_stats(equity, trades, days_inv, idx[i_start : i_end + 1])
    tsl_desc = f"TSL{int(tsl_pct * 100)}%" if tsl_pct else "无TSL"
    return {
        "id": strategy_id,
        "name": name,
        "short_label": short_label,
        "rule_desc": f"QQQ {ma_period}SMA±{int(band*100)}% + TQQQ HV20>{int(hv_exit)}%刹车(重进≤{int(hv_reentry)}%) + {tsl_desc} + BOT大底/ALT恐慌",
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
    }


def simulate_tqqq_from_soxl_signal(
    df_tqqq: pd.DataFrame,
    soxl_res: dict,
    strategy_id: str,
    name: str,
    short_label: str,
    rule_desc: str,
    start_date: str = BT_START,
    end_date: str | None = None,
) -> dict:
    """Execute SOXL strategy entry/exit dates on TQQQ with strict next-day Open pricing."""
    d_t = df_tqqq.set_index("Date").copy()
    idx = d_t.index
    if end_date is None:
        end_date = str(idx.max().date())

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = d_t["Open"].to_numpy(dtype=np.float64)
    closes = d_t["Close"].to_numpy(dtype=np.float64)

    buy_map = {pd.Timestamp(t["entry_date"]): t for t in soxl_res["trades"]}
    sell_map = {pd.Timestamp(t["exit_date"].replace("*", "")): t for t in soxl_res["trades"] if not t["is_open"]}

    cash = INITIAL_CAPITAL
    shares = 0.0
    trade_cash_start = INITIAL_CAPITAL
    avg_px = 0.0
    cur_entry_date = ""
    cur_entry_m = ""
    cur_mode = ""
    days_inv = 0
    trades: list[dict] = []
    equity = np.empty(n_bt, dtype=np.float64)

    for step, pos in enumerate(range(i_start, i_end + 1)):
        dt = idx[pos]
        o = opens[pos]
        c = closes[pos]
        if dt in buy_map and shares == 0:
            trade_cash_start = cash
            shares = cash / (o * (1.0 + FEE_RATE))
            cash = 0.0
            avg_px = float(o)
            cur_entry_date = dt.strftime("%Y-%m-%d")
            cur_entry_m = dt.strftime("%Y-%m")
            cur_mode = buy_map[dt]["mode"]
        elif dt in sell_map and shares > 0:
            cash = shares * o * (1.0 - FEE_RATE)
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append(
                {
                    "mode": cur_mode,
                    "entry_date": cur_entry_date,
                    "entry_m": cur_entry_m,
                    "tranche2_date": "-",
                    "tranche2_m": None,
                    "tranche2_px": None,
                    "entry_px": round(avg_px, 2),
                    "avg_entry_px": round(avg_px, 2),
                    "exit_date": dt.strftime("%Y-%m-%d"),
                    "exit_m": dt.strftime("%Y-%m"),
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": sell_map[dt]["exit_reason"],
                    "is_open": False,
                }
            )
            shares = 0.0
            avg_px = 0.0
        equity[step] = cash + shares * c
        if shares > 0:
            days_inv += 1

    if shares > 0:
        ret_pct = (closes[i_end] / (avg_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        trades.append(
            {
                "mode": cur_mode,
                "entry_date": cur_entry_date,
                "entry_m": cur_entry_m,
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(avg_px, 2),
                "avg_entry_px": round(avg_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    ls = dict(soxl_res["live_status"])
    ls["reason"] = f"[SOXL 前置信号] {ls['reason']}"
    if shares > 0:
        ls["entry_px"] = round(avg_px, 2)
        ls["ret_pct"] = round(float((closes[i_end] / (avg_px * (1.0 + FEE_RATE)) - 1.0) * 100.0), 1)
        ls["sl_price"] = None
        ls["wtsl_price"] = None

    stats = _compute_stats(equity, trades, days_inv, idx[i_start : i_end + 1])
    return {
        "id": strategy_id,
        "name": name,
        "short_label": short_label,
        "rule_desc": rule_desc,
        "stats": stats,
        "trades": trades,
        "live_status": ls,
    }


def main() -> None:
    df_tqqq = load_daily_adjusted(TQQQ_PATH)
    df_qqq = load_daily_adjusted(QQQ_PATH)

    monthly, weekly_sma = build_monthly_and_weekly(df_tqqq)
    c7_entries = detect_c7_vol_entries(df_tqqq)
    feats = build_causal_features(df_tqqq, df_qqq)
    latest_date = str(df_tqqq["Date"].max().date())

    # TQQQ-native strategies
    s_dual_1 = simulate_tqqq_dual(
        feats, df_qqq, ma_period=200, band=0.01, hv_exit=100.0, hv_reentry=80.0, tsl_pct=0.25,
        strategy_id="TQQQ_Dual_SMA200_1%_HV100_TSL25",
        name="TQQQ 专属双引擎 ±1%: QQQ 200SMA±1% + HV100/80 + TSL25%",
        short_label="TQQQ双引擎 (200SMA±1%+TSL25%)",
        end_date=latest_date,
    )
    s_dual_2 = simulate_tqqq_dual(
        feats, df_qqq, ma_period=200, band=0.02, hv_exit=100.0, hv_reentry=80.0, tsl_pct=0.25,
        strategy_id="TQQQ_Dual_SMA200_2%_HV100_TSL25",
        name="TQQQ 专属双引擎 ±2%: QQQ 200SMA±2% + HV100/80 + TSL25% (15.5年最强)",
        short_label="TQQQ双引擎 (200SMA±2%+TSL25%)",
        end_date=latest_date,
    )
    s_dual_150 = simulate_tqqq_dual(
        feats, df_qqq, ma_period=150, band=0.00, hv_exit=100.0, hv_reentry=80.0, tsl_pct=None,
        strategy_id="TQQQ_Dual_SMA150_0%_HV100_NoTSL",
        name="TQQQ 双引擎高收益版: QQQ 150SMA + HV100/80 (无TSL)",
        short_label="TQQQ双引擎 (150SMA+HV100)",
        end_date=latest_date,
    )

    # TQQQ Major Cycle: F0 | MF_Lin_Buy + TP3h
    w = feats["weekly"]
    w["F0_or_MFLin"] = w["E2_PrevHigh"] | w["MF_Lin_Buy"]
    s_cycle = simulate_tp3h(
        feats,
        use_f1_filter=False,
        end_date=latest_date,
        entry_col_override="F0_or_MFLin",
        mode_tag_override="BOT_F0_MF",
        strategy_id="F0_or_MFLin_TP3h",
    )
    s_cycle["id"] = "F0_or_MFLin_TP3h"
    s_cycle["name"] = "TQQQ 通用大周期止盈: F0 + 月Fisher兜底 + TP3h"
    s_cycle["short_label"] = "TQQQ大周期 (F0+月Fisher+TP3h)"
    s_cycle["rule_desc"] = "pre_c7_vol&PrevHigh 或 月线M_Fish<0.8金叉 || 持满50w&M_Fish≥2&30W≥40%周Fisher死叉止盈 (X1a+SL_W25兜底)"

    s_d1 = simulate_d1_x1a_wtsl35(feats, use_split_50_50=False, end_date=latest_date)

    # Cross-asset: SOXL signals driving TQQQ
    df_soxl = load_daily_adjusted(SOXL_PATH)
    df_smh = load_daily_adjusted(SMH_PATH)
    feats_soxl = build_causal_features(df_soxl, df_smh)
    soxl_h5 = simulate_h5_adapt_hv(feats_soxl, end_date=latest_date)
    soxl_c2 = simulate_tp3h(
        feats_soxl,
        use_f1_filter=True,
        end_date=latest_date,
        entry_col_override="C2_Deep85HL",
        mode_tag_override="BOT_C2",
        strategy_id="C2_F1_or_MFishLin_TP3h",
        tp_split_lottery_pct=0.0,
    )
    s_soxl_h5_tqqq = simulate_tqqq_from_soxl_signal(
        df_tqqq,
        soxl_h5,
        strategy_id="SOXL_H5_to_TQQQ",
        name="跨标的旗舰: SOXL双引擎(H5)信号 → 买卖TQQQ (12年$713万 / 15.5年$3459万)",
        short_label="SOXL(H5)→TQQQ",
        rule_desc="由 SOXL H5 双引擎(E2大底晋升+SMH 200SMA+自适应HV+ALT恐慌)发出买卖信号，次日开盘全仓买卖 TQQQ",
        end_date=latest_date,
    )
    s_soxl_c2_tqqq = simulate_tqqq_from_soxl_signal(
        df_tqqq,
        soxl_c2,
        strategy_id="SOXL_C2_to_TQQQ",
        name="跨标的大周期: SOXL(C2+Deep85_HL+TP3h)信号 → 买卖TQQQ (12年$292万 / 回撤-37.0%)",
        short_label="SOXL(C2)→TQQQ",
        rule_desc="由 SOXL C2+Deep85_HL+TP3h 发出大底与逃顶信号，次日开盘全仓买卖 TQQQ（5战全胜，最大回撤仅-36.97%）",
        end_date=latest_date,
    )

    strategies = {
        "SOXL_H5_to_TQQQ": s_soxl_h5_tqqq,
        "SOXL_C2_to_TQQQ": s_soxl_c2_tqqq,
        "TQQQ_Dual_SMA200_1%_HV100_TSL25": s_dual_1,
        "TQQQ_Dual_SMA200_2%_HV100_TSL25": s_dual_2,
        "TQQQ_Dual_SMA150_0%_HV100_NoTSL": s_dual_150,
        "F0_or_MFLin_TP3h": s_cycle,
        "D1_X1a_WTSL35": s_d1,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    live_market = build_live_market_snapshot(feats)
    html = render_html(monthly, weekly_sma, c7_entries, strategies, live_market=live_market)

    # Adapt labels and strategy options cleanly for TQQQ
    tqqq_options = """<select id="strategySelect" class="strategy-select">
        <option value="SOXL_H5_to_TQQQ" selected>跨标的旗舰: SOXL双引擎(H5)信号 → 买TQQQ (12年$713万 / 15.5年$3459万, Calmar 1.58)</option>
        <option value="SOXL_C2_to_TQQQ">跨标的大周期: SOXL(C2+Deep85_HL+TP3h) → 买TQQQ (5笔全胜, 12年$292万, 回撤-37.0%)</option>
        <option value="TQQQ_Dual_SMA200_1%_HV100_TSL25">TQQQ 专属双引擎 ±1%: QQQ 200SMA±1% + HV100/80 + TSL25% (12年$183万 / 15.5年$431万)</option>
        <option value="TQQQ_Dual_SMA200_2%_HV100_TSL25">TQQQ 专属双引擎 ±2%: QQQ 200SMA±2% + HV100/80 + TSL25% (15.5年$446万, 回撤-47.8%)</option>
        <option value="TQQQ_Dual_SMA150_0%_HV100_NoTSL">TQQQ 双引擎高收益版: QQQ 150SMA + HV100/80 无TSL (12年$207万 / Calmar 1.36)</option>
        <option value="F0_or_MFLin_TP3h">TQQQ 通用大周期止盈: F0+月Fisher兜底+TP3h (7笔, 12年$153万 / 15.5年$367万)</option>
        <option value="D1_X1a_WTSL35">策略3 极简大周期: D1_X1a_WTSL35</option>
        <option value="NONE">不显示策略标注</option>
      </select>"""

    html = re.sub(
        r'<select id="strategySelect" class="strategy-select">.*?</select>',
        tqqq_options,
        html,
        count=1,
        flags=re.S,
    )
    html = html.replace("SOXL 月线顶底区域", "TQQQ 月线顶底区域")
    html = html.replace("SOXL 月线", "TQQQ 月线")
    html = html.replace("SOXL 价格", "TQQQ 价格")
    html = html.replace("<span>SOXL <b>", "<span>TQQQ <b>")
    html = html.replace("<span>SMH <b>", "<span>QQQ <b>")
    html = html.replace("SMH > 200SMA", "QQQ > 200SMA")
    html = html.replace("SMH < 200SMA", "QQQ < 200SMA")
    html = html.replace("SMH $", "QQQ $")
    html = html.replace("SMH 趋势仓", "QQQ 趋势仓")
    # Route TQQQ_Dual_* strategies to the Dual-Engine radar branch with TQQQ's HV100/80 rules
    html = html.replace(
        'if (stratId === "H5_E2_Promote_S1_AdaptHV" && ls) {',
        'if (stratId.startsWith("TQQQ_Dual_") && ls) {',
    )
    html = html.replace("HV20 ≤ 105.8% (高波离场需≤95%)", "TQQQ HV20 ≤ 92% (高波离场需≤80%)")
    html = html.replace(
        "const hvLim = (m.m_fish_prev === null || m.m_fish_prev < 1.5) ? 140 : 115;",
        "const hvLim = 100;",
    )
    html = html.replace("m.hv20 <= 105.8", "m.hv20 <= 92")

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")


if __name__ == "__main__":
    main()
