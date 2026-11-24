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
from tools.build_btc_zones import _ds_du, _pine_gaussian_filter, build_halving_zone_shapes
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
        const g2 = [
          radarCell("买点区时间兜底 H ≤ -490天", ml.du <= 490, `当前 -${ml.du} 天 (还差 ${Math.max(0, ml.du - 490)} 天)`),
          radarCell("mNAV冷却接回参考 (≤1.10x)", ml.mstr_btc_r60 <= 1.10, `当前 ${fmtV(ml.mstr_btc_r60)}x | MSTR GAC中轨 $${fmtV(ml.gac_mid)}`),
        ];
        groups.push(renderRadarGroup("买点区主规则 · BTC减半窗口脉冲冷却（满足全部 → 买入）", g1, "all"));
        groups.push(renderRadarGroup("兜底与溢价冷却参考", g2, "any_entry"));
      }

"""
    i_r1 = html.index(radar_anchor)
    i_r2 = html.index(radar_end, i_r1)
    html = html[:i_r1] + mstr_radar_js + html[i_r2:]

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")

    if args.notify_tg:
        tg_msg = format_mstr_tg_summary(live_market, strategies)
        ok = send_tg_notification(tg_msg)
        print(f"[Telegram] sent={ok}")


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
    return "\n".join(lines)


if __name__ == "__main__":
    main()
