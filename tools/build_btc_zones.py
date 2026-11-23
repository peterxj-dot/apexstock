"""Build a dedicated BTC-USD monthly LOG chart with Halving Buy/Sell zones and volatility-timed strategies."""

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

BTC_PATH = DATA_DIR / "BTC-USD.csv"
OUTPUT_HTML = ROOT / "btc_monthly_zones.html"

HALVINGS = [
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-04-10"),  # Projected Cycle 5 halving
]


def _ds_du(dt: pd.Timestamp) -> tuple[int, int]:
    prev_h = max((h for h in HALVINGS if h <= dt), default=None)
    next_h = min((h for h in HALVINGS if h > dt), default=None)
    ds = (dt - prev_h).days if prev_h is not None else 9999
    du = (next_h - dt).days if next_h is not None else 9999
    return ds, du


def _compute_btc_stats(
    equity: np.ndarray,
    trades: list[dict],
    days_invested: int,
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Compute 365-day/yr crypto performance stats matching SOXL/TQQQ schema."""
    n = len(equity)
    final_val = float(equity[-1])
    years_span = n / 365.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0)) * 100.0
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else 0.0
    rets = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365.0)) if (len(rets) > 0 and np.std(rets) > 0) else 0.0
    trades_cnt = len(trades)
    win_rate = (sum(1 for t in trades if t["ret_pct"] > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = (days_invested / n) * 100.0 if n > 0 else 0.0

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


def _pine_gaussian_filter(src: pd.Series, period: int, poles: int) -> pd.Series:
    beta = 0.5 * (1.0 - np.cos(2.0 * np.pi / period))
    g = src.copy()
    for _ in range(poles):
        g = g.ewm(alpha=beta, adjust=False).mean()
    return g


def simulate_btc_strategy(
    df_btc: pd.DataFrame,
    mode: str,
    strategy_id: str,
    name: str,
    short_label: str,
    rule_desc: str,
    start_date: str = "2015-01-01",
) -> dict:
    """Strictly causal daily BTC simulator supporting Halving+Volatility, PureHalv, Phase, and GAC."""
    d = df_btc.set_index("Date").copy()
    idx = d.index
    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)

    hv20 = historical_volatility(d["Close"], period=20, annualization_factor=365)
    hv60 = historical_volatility(d["Close"], period=60, annualization_factor=365)
    hv_ratio = (hv20 / hv60).to_numpy(dtype=np.float64)
    hv20_a = hv20.to_numpy(dtype=np.float64)
    hv60_a = hv60.to_numpy(dtype=np.float64)
    hv_spike_30d = (hv_ratio >= 1.30)
    hv_spike_recent = pd.Series(hv_spike_30d, index=idx).rolling(35, min_periods=1).max().to_numpy(dtype=bool)
    sma150 = sma(d["Close"], 150).to_numpy(dtype=np.float64)

    g_mid = _pine_gaussian_filter(d["Close"], 10, 2)
    g_dev = _pine_gaussian_filter((d["Close"] - g_mid).abs(), 10, 2)
    g_upper = (g_mid + 1.5 * g_dev).to_numpy(dtype=np.float64)
    g_mid_a = g_mid.to_numpy(dtype=np.float64)

    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = len(idx) - 1
    n_bt = i_end - i_start + 1

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

    def eval_signal(pos: int, is_holding: bool) -> tuple[str, str]:
        ds, du = _ds_du(idx[pos])
        hvr = hv_ratio[pos]
        hv2 = hv20_a[pos]
        c = closes[pos]
        if mode == "HALV_HV_COMBO":
            in_buy_zone = (du <= 540) and (ds > 560)
            in_sell_zone = (ds >= 480) and (du > 540)
            if not is_holding:
                if in_buy_zone:
                    if hv_spike_recent[pos] and (not np.isnan(hvr)) and hvr < 1.0:
                        return "BUY", "HV脉冲冷却(HVRatio<1.0)"
                    if du <= 490:
                        return "BUY", "买点区时间兜底(H-490d)"
                elif ds < 480:
                    return "BUY", "减半牛市持仓期"
            else:
                if in_sell_zone and (not np.isnan(hvr)) and hvr >= 1.30:
                    return "SELL", f"卖点区HV脉冲(HVR={hvr:.2f})"
                if ds >= 540 and du > 540:
                    return "SELL", f"卖点区时间兜底(H+{ds}d)"
        elif mode == "PURE_HALV":
            in_win = (du <= 500) or (ds < 530)
            if not is_holding and in_win:
                return "BUY", f"减半买点(H-{du}d)"
            if is_holding and not in_win:
                return "SELL", f"减半卖点(H+{ds}d)"
        elif mode == "PHASE_SMA150":
            if du <= 520 and du > 200:
                want = True
            elif du <= 200 or ds <= 250:
                want = (not np.isnan(sma150[pos])) and (c > sma150[pos]) and (np.isnan(hv2) or hv2 <= 95.0)
            elif ds < 535:
                want = (not np.isnan(sma150[pos])) and (c > sma150[pos] * 0.95)
            else:
                want = False
            if not is_holding and want:
                return "BUY", "阶段规则触发买入"
            if is_holding and not want:
                return "SELL", "阶段规则触发离场"
        elif mode == "HALV_GAC":
            in_win = (du <= 500) or (ds < 530)
            g_buy = (pos >= 1) and (c > g_upper[pos]) and (closes[pos - 1] <= g_upper[pos - 1])
            g_sell = (pos >= 1) and (c < g_mid_a[pos]) and (closes[pos - 1] >= g_mid_a[pos - 1])
            if not is_holding and in_win and (g_buy or c > g_upper[pos]):
                return "BUY", "减半窗+GAC突破上轨"
            if is_holding and ((not in_win) or g_sell):
                return "SELL", "跌破GAC中轨或出减半窗"
        return "", ""

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        if pending == "BUY" and shares == 0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            entry_px = float(o)
            entry_dt_str = dt_str
            entry_m_str = m_str
        elif pending == "SELL" and shares > 0:
            cash = shares * o * (1.0 - FEE_RATE)
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append(
                {
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
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            shares = 0.0
            entry_px = 0.0
        pending = ""
        pending_reason = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_inv += 1

        if pos >= i_end:
            continue
        pending, pending_reason = eval_signal(pos, shares > 0)

    if shares > 0:
        ret_pct = (closes[i_end] / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        trades.append(
            {
                "mode": mode,
                "entry_date": entry_dt_str,
                "entry_m": entry_m_str,
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

    # Live status at i_end
    sig_act, sig_rsn = eval_signal(i_end, shares > 0)
    ds_live, du_live = _ds_du(idx[i_end])
    if shares > 0:
        live_ret = round((closes[i_end] / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0, 1)
        if sig_act == "SELL":
            live_act, live_badge, live_tone, live_reason = "SELL", f"卖出 ({sig_rsn})", "sell", f"触发 {sig_rsn}，下个交易日开盘卖出"
        else:
            live_act, live_badge, live_tone, live_reason = "HOLD", "继续持有", "hold", f"减半牛市持仓中 (距上次减半 +{ds_live}d / 下次 -{du_live}d)"
        live_status = {
            "action": live_act,
            "badge": live_badge,
            "tone": live_tone,
            "position_label": f"持仓 100% ({mode})",
            "pos_mode": "BULL",
            "entry_date": entry_dt_str,
            "entry_px": round(entry_px, 2),
            "ret_pct": live_ret,
            "hold_weeks": None,
            "high_zone_armed": False,
            "sl_price": None,
            "wtsl_price": None,
            "cooldown_days": 0,
            "reason": live_reason,
        }
    else:
        if sig_act == "BUY":
            live_act, live_badge, live_tone, live_reason = "BUY", f"买入 100% ({sig_rsn})", "buy", f"触发 {sig_rsn}，下个交易日开盘全仓买入"
        else:
            live_act, live_badge, live_tone = "WAIT", "保持空仓", "wait"
            if du_live <= 540:
                live_reason = f"已进入买点区 (H-{du_live}d)：HV_Ratio={hv_ratio[i_end]:.2f}，等待脉冲冷却 (<1.0) 或 H-490d 兜底"
            else:
                live_reason = f"熊市空仓期 (距上次减半 +{ds_live}d / 距下次减半 -{du_live}d)"
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
            "cooldown_days": 0,
            "reason": live_reason,
        }

    stats = _compute_btc_stats(equity, trades, days_inv, idx[i_start : i_end + 1])
    return {
        "id": strategy_id,
        "name": name,
        "short_label": short_label,
        "rule_desc": rule_desc,
        "stats": stats,
        "trades": trades,
        "live_status": live_status,
        "btc_live": {
            "ds": ds_live,
            "du": du_live,
            "hv20": round(float(hv20_a[i_end]), 1),
            "hv60": round(float(hv60_a[i_end]), 1),
            "hv_ratio": round(float(hv_ratio[i_end]), 2),
            "hv_spike_recent": bool(hv_spike_recent[i_end]),
            "sma150": round(float(sma150[i_end]), 2) if not np.isnan(sma150[i_end]) else None,
            "in_buy_zone": bool((du_live <= 540) and (ds_live > 560)),
            "in_sell_zone": bool((ds_live >= 480) and (du_live > 540)),
        },
    }


def build_halving_zone_shapes() -> list[dict]:
    """Build green Buy-Zone (H-540~-460) and red Sell-Zone (H+480~+560) bands for Plotly."""
    zones = []
    for i, hdt in enumerate(HALVINGS, 1):
        b_s = (hdt - pd.Timedelta(days=540)).strftime("%Y-%m-%d")
        b_e = (hdt - pd.Timedelta(days=460)).strftime("%Y-%m-%d")
        s_s = (hdt + pd.Timedelta(days=480)).strftime("%Y-%m-%d")
        s_e = (hdt + pd.Timedelta(days=560)).strftime("%Y-%m-%d")
        if hdt >= pd.Timestamp("2016-01-01"):
            zones.append({"kind": "BUY", "start": b_s, "end": b_e, "label": f"买点区 (H-540~-460)"})
        if hdt <= pd.Timestamp("2025-01-01"):
            zones.append({"kind": "SELL", "start": s_s, "end": s_e, "label": f"卖点区 (H+480~+560)"})
    return zones


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BTC monthly zones HTML and optional Telegram alert.")
    parser.add_argument("--notify-tg", action="store_true", help="Send daily BTC status summary to Telegram.")
    args = parser.parse_args()

    df_btc = load_daily_adjusted(BTC_PATH)
    monthly, weekly_sma = build_monthly_and_weekly(df_btc)
    c7_entries = detect_c7_vol_entries(df_btc)
    feats = build_causal_features(df_btc, df_btc)

    s_combo = simulate_btc_strategy(
        df_btc,
        mode="HALV_HV_COMBO",
        strategy_id="BTC_Halv_HV_Combo",
        name="BTC 减半+波动率共振旗舰: [H-540~-460] HV冷却买 + [H+480~+560] HV脉冲卖",
        short_label="BTC减半+HV共振 (HVRatio)",
        rule_desc="买点区[H-540~-460]内等HV_Ratio≥1.3脉冲后冷却<1.0买(H-490兜底) | 卖点区[H+480~+560]内HV_Ratio≥1.3脉冲卖(H+540兜底)",
    )
    s_pure = simulate_btc_strategy(
        df_btc,
        mode="PURE_HALV",
        strategy_id="BTC_Pure_Halv_500_530",
        name="BTC 纯减半极简周期: H-500d买 / H+530d卖 (3战全胜, 12年$7464万)",
        short_label="BTC纯减半 (-500d~+530d)",
        rule_desc="固定减半前500天买入，减半后530天卖出，其余时间100%空仓避开大熊市",
    )
    s_phase = simulate_btc_strategy(
        df_btc,
        mode="PHASE_SMA150",
        strategy_id="BTC_Phase_Halv_SMA150",
        name="BTC 四阶段高夏普版: 筑底死拿 + 爬坡SMA150/HV≤95% + 冲顶SMA150×0.95 (Calmar 2.00)",
        short_label="BTC四阶段 (Calmar 2.00)",
        rule_desc="筑底[H-520~-200]死拿 | 爬坡[H-200~+250] SMA150&HV≤95% | 冲顶[H+250~+535] SMA150×0.95 | 熊市空仓",
    )
    s_gac = simulate_btc_strategy(
        df_btc,
        mode="HALV_GAC",
        strategy_id="BTC_Daily_GAC_10_2_15",
        name="BTC 减半窗 + 日线高斯通道 GAC(10,2,1.5) (MaxDD仅-36.2%, Calmar 2.04)",
        short_label="BTC减半+GAC(10,2,1.5)",
        rule_desc="仅在减半牛市窗[H-500~+530]内交易，突破GAC上轨买入、跌破GAC中轨卖出",
    )

    strategies = {
        "BTC_Halv_HV_Combo": s_combo,
        "BTC_Pure_Halv_500_530": s_pure,
        "BTC_Phase_Halv_SMA150": s_phase,
        "BTC_Daily_GAC_10_2_15": s_gac,
    }
    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    live_market = build_live_market_snapshot(feats)
    live_market["btc_live"] = s_combo["btc_live"]
    halv_zones = build_halving_zone_shapes()

    html = render_html(monthly, weekly_sma, c7_entries, strategies, live_market=live_market)

    btc_options = """<select id="strategySelect" class="strategy-select">
        <option value="BTC_Halv_HV_Combo" selected>BTC 减半+波动率共振: 买点区[H-540~-460] HV冷却 + 卖点区[H+480~+560] HV脉冲</option>
        <option value="BTC_Pure_Halv_500_530">BTC 纯减半极简周期: H-500d买 / H+530d卖 (3战全胜, 12年$7464万)</option>
        <option value="BTC_Phase_Halv_SMA150">BTC 四阶段高夏普版: 分阶段切换SMA150 (12年$3671万, MaxDD -49.6%, Calmar 2.00)</option>
        <option value="BTC_Daily_GAC_10_2_15">BTC 减半窗+高斯通道 GAC(10,2,1.5) (MaxDD仅-36.2%, Calmar 2.04)</option>
        <option value="NONE">不显示策略标注</option>
      </select>"""

    html = re.sub(
        r'<select id="strategySelect" class="strategy-select">.*?</select>',
        btc_options,
        html,
        count=1,
        flags=re.S,
    )
    html = html.replace("SOXL 月线顶底区域", "BTC 减半周期与波动率区域")
    html = html.replace("SOXL 月线", "BTC 月线")
    html = html.replace("SOXL 价格", "BTC 价格")

    # Inject Halving Buy/Sell Zone shapes & BTC-specific cockpit radar
    halv_json = json.dumps(halv_zones, separators=(",", ":"), ensure_ascii=False)
    html = html.replace(
        "const LIVE_MARKET =",
        f"const HALVING_ZONES = {halv_json};\n    const LIVE_MARKET =",
    )

    # Add Halving Buy/Sell Zone green/red vertical bands on top of chart shapes
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
      }));"""
    zone_inject_new = """      const shapes = HALVING_ZONES.map(hz => ({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: hz.start,
        x1: hz.end,
        y0: 0,
        y1: 1,
        fillcolor: hz.kind === "BUY" ? "rgba(63,185,80,0.20)" : "rgba(248,81,73,0.20)",
        line: {
          color: hz.kind === "BUY" ? "rgba(63,185,80,0.85)" : "rgba(248,81,73,0.85)",
          width: 1.5,
          dash: "dash",
        },
        layer: "below",
      }));"""
    html = html.replace(zone_inject_js, zone_inject_new)

    # Add text labels for Buy Zone / Sell Zone in annotations
    ann_target = "const annotations = zones.map(z => ({"
    ann_replace = """const hzAnnotations = HALVING_ZONES.map(hz => ({
        x: hz.start,
        y: 0.98,
        xref: "x",
        yref: "paper",
        text: hz.label,
        showarrow: false,
        xanchor: "left",
        font: { size: 11, color: hz.kind === "BUY" ? "#3fb950" : "#f85149" },
        bgcolor: "rgba(13,17,23,0.82)",
        bordercolor: hz.kind === "BUY" ? "#3fb950" : "#f85149",
        borderpad: 3,
      }));
      const annotations = [...hzAnnotations, ...zones.map(z => ({"""
    html = html.replace(ann_target, ann_replace)
    html = html.replace(
        "      }));\n\n      const xMin = MONTHLY[0].m;",
        "      }))];\n\n      const xMin = MONTHLY[0].m;",
    )

    # Customize BTC live cockpit meta & radar
    old_meta = """      const freshHtml = `
        <span class="fresh-pill">最新交易日: ${m.last_date} (${m.weekday_cn})</span>
        ${m.is_friday ? "" : `<span class="warn-pill">非周五收盘（周中或节假日数据）</span>`}
        <span>SOXL <b>$${fmtV(m.soxl_close)}</b></span>
        <span>SMH <b>$${fmtV(m.smh_close)}</b> (200SMA $${fmtV(m.smh_sma200)})</span>
        <span>HV20 <b>${fmtV(m.hv20, "%", 1)}</b></span>
      `;"""
    new_meta = """      const bl = m.btc_live || {};
      const freshHtml = `
        <span class="fresh-pill">最新日期: ${m.last_date} (${m.weekday_cn})</span>
        <span>BTC <b>$${fmtV(m.soxl_close)}</b></span>
        <span>距下次减半: <b>-${bl.du} 天</b> (上次 +${bl.ds} 天)</span>
        <span>HV20 <b>${fmtV(bl.hv20, "%", 1)}</b> / HV60 <b>${fmtV(bl.hv60, "%", 1)}</b> (比值 <b>${fmtV(bl.hv_ratio)}</b>)</span>
      `;"""
    html = html.replace(old_meta, new_meta)

    # Replace radar branch for BTC
    radar_anchor = '      const groups = [];'
    radar_end = '      box.innerHTML = `'
    btc_radar_js = """      const groups = [];
      if (ls && ls.pos_mode !== "FLAT") {
        const g1 = [
          radarCell("处于卖点时间窗 [H+480~+560]", Boolean(bl.in_sell_zone), `距上次减半 +${bl.ds} 天`),
          radarCell("波动率脉冲 HV_Ratio ≥ 1.30", bl.hv_ratio >= 1.30, `当前 HV20/HV60 = ${fmtV(bl.hv_ratio)}`),
          radarCell("时间兜底 H ≥ +540 天", bl.ds >= 540 && bl.du > 540, `当前 +${bl.ds} 天`),
        ];
        const g2 = [
          radarCell("150日均线参考", bl.sma150 ? m.soxl_close > bl.sma150 : false, `BTC $${fmtV(m.soxl_close)} vs SMA150 $${fmtV(bl.sma150)}`),
        ];
        groups.push(renderRadarGroup("卖点区监控 · 减半后期波动率逃顶", g1, "any_exit"));
        groups.push(renderRadarGroup("趋势参考 · 150日均线", g2, "all"));
      } else {
        const g1 = [
          radarCell("处于买点时间窗 [H-540~-460]", Boolean(bl.in_buy_zone), `距下次减半 -${bl.du} 天`),
          radarCell("近35日曾现恐慌脉冲 HVR≥1.30", Boolean(bl.hv_spike_recent), `当前 HVR=${fmtV(bl.hv_ratio)}`),
          radarCell("脉冲冷却 HV_Ratio < 1.0", bl.hv_ratio < 1.0, `当前 HV20/HV60 = ${fmtV(bl.hv_ratio)}`),
        ];
        const g2 = [
          radarCell("买点区时间兜底 H ≤ -490天", bl.du <= 490, `当前距减半 -${bl.du} 天 (还差 ${Math.max(0, bl.du - 490)} 天)`),
          radarCell("200周均线极值参考", m.min8w_low_200w !== null && m.min8w_low_200w <= 1.15, `8W Low/200W = ${fmtV(m.min8w_low_200w, "x", 2)} (200W=$${fmtV(m.w_sma200)})`),
        ];
        groups.push(renderRadarGroup("买点区主规则 · 减半前波动率脉冲冷却（满足全部 → 买入）", g1, "all"));
        groups.push(renderRadarGroup("买点区兜底与 200W 极值参考", g2, "any_entry"));
      }

"""
    i_r1 = html.index(radar_anchor)
    i_r2 = html.index(radar_end, i_r1)
    html = html[:i_r1] + btc_radar_js + html[i_r2:]

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")

    if args.notify_tg:
        tg_msg = format_btc_tg_summary(live_market, strategies)
        ok = send_tg_notification(tg_msg)
        print(f"[Telegram] sent={ok}")


def format_btc_tg_summary(live_market: dict, strategies: dict) -> str:
    """Format concise daily BTC status message for Telegram."""
    bl = live_market["btc_live"]
    s1 = strategies["BTC_Halv_HV_Combo"]["live_status"]
    zone_tag = "买点区[H-540~-460]" if bl["in_buy_zone"] else ("卖点区[H+480~+560]" if bl["in_sell_zone"] else "区间外")
    lines = [
        f"BTC 日更状态 ({live_market['last_date']} {live_market['weekday_cn']})",
        f"BTC: ${live_market['soxl_close']:,.2f} | 减半: -{bl['du']}d / +{bl['ds']}d ({zone_tag})",
        f"HV20={bl['hv20']}% | HV60={bl['hv60']}% | HVR={bl['hv_ratio']} | 35日脉冲={bl['hv_spike_recent']}",
        f"策略 [Halv+HV]: {s1['badge']} ({s1['position_label']})",
        f"原因: {s1['reason']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
