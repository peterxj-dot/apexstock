"""Backtest & build interactive HTML comparison: SOXL signals on VGT, VGT+20% TQQQ (margin), and VGT+20% TECL (margin)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import (
    DATA_DIR,
    FEE_RATE,
    INITIAL_CAPITAL,
    _compute_stats,
    build_causal_features,
    load_daily_adjusted,
    send_tg_notification,
    simulate_h5_adapt_hv,
    simulate_tp3h,
)

MARGIN_RATE_ANNUAL = 0.065  # 6.5% annual margin interest on borrowed 20%
OUTPUT_HTML = ROOT / "vgt_margin_comparison.html"


def _trade_windows_from_soxl(soxl_res: dict, idx: pd.DatetimeIndex) -> list[tuple[int, int, str]]:
    """Map SOXL trade entry/exit dates to target index positions."""
    windows = []
    for t in soxl_res["trades"]:
        e_dt = pd.Timestamp(t["entry_date"])
        e_pos = int(idx.searchsorted(e_dt, side="left"))
        if t.get("is_open", False):
            x_pos = len(idx) - 1
            is_open = True
        else:
            x_dt = pd.Timestamp(t["exit_date"].split(" ")[0])
            x_pos = int(idx.searchsorted(x_dt, side="left"))
            is_open = False
        if e_pos < len(idx):
            windows.append((e_pos, min(x_pos, len(idx) - 1), is_open))
    return windows


def simulate_spot_or_margin(
    df_base: pd.DataFrame,
    df_margin_leg: pd.DataFrame | None,
    margin_weight: float,
    soxl_res: dict,
    start_date: str = "2015-01-02",
    end_date: str = "2026-11-23",
    margin_rate_annual: float = MARGIN_RATE_ANNUAL,
) -> dict:
    """Strictly causal next-day Open simulator for 100% base + optional margin_weight leveraged leg."""
    d1 = df_base.set_index("Date").sort_index()
    idx = d1.index
    if df_margin_leg is not None:
        d2 = df_margin_leg.set_index("Date").reindex(idx, method="ffill")
    else:
        d2 = d1

    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = int(idx.searchsorted(pd.Timestamp(end_date), side="right")) - 1
    n_bt = i_end - i_start + 1

    o1 = d1["Open"].to_numpy(dtype=np.float64)
    c1 = d1["Close"].to_numpy(dtype=np.float64)
    o2 = d2["Open"].to_numpy(dtype=np.float64)
    c2 = d2["Close"].to_numpy(dtype=np.float64)

    windows = _trade_windows_from_soxl(soxl_res, idx)
    entry_map = {e: (x, is_op) for e, x, is_op in windows}
    exit_map = {x: e for e, x, is_op in windows if not is_op}

    cash = INITIAL_CAPITAL
    sh1 = 0.0
    sh2 = 0.0
    loan = 0.0
    total_interest = 0.0
    trade_eq_start = INITIAL_CAPITAL
    days_inv = 0
    max_debt_ratio = 0.0
    trades: list[dict] = []
    equity = np.empty(n_bt, dtype=np.float64)
    cur_entry_dt = ""

    for k, pos in enumerate(range(i_start, i_end + 1)):
        # Exit at today's Open if scheduled
        if pos in exit_map and (sh1 > 0.0 or sh2 > 0.0):
            proceeds1 = sh1 * o1[pos] * (1.0 - FEE_RATE)
            proceeds2 = sh2 * o2[pos] * (1.0 - FEE_RATE)
            cash = proceeds1 + proceeds2 - loan
            ret_pct = (cash / trade_eq_start - 1.0) * 100.0
            trades.append({
                "entry_date": cur_entry_dt,
                "exit_date": idx[pos].strftime("%Y-%m-%d"),
                "ret_pct": round(float(ret_pct), 2),
            })
            sh1 = sh2 = loan = 0.0

        # Entry at today's Open if scheduled
        if pos in entry_map and sh1 == 0.0:
            trade_eq_start = cash
            cur_entry_dt = idx[pos].strftime("%Y-%m-%d")
            # Base leg: 100% of equity
            sh1 = (cash * (1.0 - FEE_RATE)) / o1[pos]
            # Margin leg: margin_weight of equity borrowed
            if margin_weight > 0.0:
                loan = cash * margin_weight
                sh2 = (loan * (1.0 - FEE_RATE)) / o2[pos]
            cash = 0.0

        if sh1 > 0.0:
            days_inv += 1
            if loan > 0.0:
                daily_int = loan * (margin_rate_annual / 252.0)
                loan += daily_int
                total_interest += daily_int

        gross_assets = cash + sh1 * c1[pos] + sh2 * c2[pos]
        if loan > 0.0 and gross_assets > 0.0:
            dr = (loan / gross_assets) * 100.0
            if dr > max_debt_ratio:
                max_debt_ratio = dr

        eq_today = gross_assets - loan
        equity[k] = max(eq_today, 1.0)

    if sh1 > 0.0:
        ret_pct = (equity[-1] / trade_eq_start - 1.0) * 100.0
        trades.append({
            "entry_date": cur_entry_dt,
            "exit_date": idx[i_end].strftime("%Y-%m-%d") + " (OPEN)",
            "ret_pct": round(float(ret_pct), 2),
        })

    dates_bt = idx[i_start : i_end + 1]
    stats = _compute_stats(equity, trades, days_inv, dates_bt)
    rets = np.diff(equity) / equity[:-1]
    down = rets[rets < 0.0]
    sortino = float(np.mean(rets) / np.std(down) * np.sqrt(252.0)) if (len(down) > 0 and np.std(down) > 0) else 0.0
    stats["sortino"] = round(sortino, 2)
    stats["interest_paid"] = round(total_interest, 2)
    stats["max_debt_ratio"] = round(max_debt_ratio, 2)

    s_eq = pd.Series(equity, index=dates_bt)
    ann = s_eq.resample("YE").last().pct_change()
    # First partial/full year return from INITIAL_CAPITAL
    first_yr_end = s_eq.resample("YE").last().iloc[0]
    ann.iloc[0] = first_yr_end / INITIAL_CAPITAL - 1.0
    ann_dict = {str(d.year): round(float(v) * 100.0, 1) for d, v in ann.items()}

    peak = np.maximum.accumulate(equity)
    dd_series = (equity / peak - 1.0) * 100.0

    # Downsample weekly (every 5 trading days) + last point for crisp fast chart
    pts_idx = list(range(0, n_bt, 5))
    if pts_idx[-1] != n_bt - 1:
        pts_idx.append(n_bt - 1)
    curve = [{"d": dates_bt[j].strftime("%Y-%m-%d"), "v": round(float(equity[j]), 2), "dd": round(float(dd_series[j]), 2)} for j in pts_idx]

    return {"stats": stats, "annual": ann_dict, "curve": curve, "trades": trades}


def simulate_buy_hold(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    d = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    dates_bt = pd.DatetimeIndex(d["Date"])
    shares = (INITIAL_CAPITAL * (1.0 - FEE_RATE)) / opens[0]
    eq = shares * closes
    trades = [{"entry_date": start_date, "exit_date": end_date, "ret_pct": round(float((eq[-1] / INITIAL_CAPITAL - 1.0) * 100.0), 2)}]
    stats = _compute_stats(eq, trades, len(d), dates_bt)
    rets = np.diff(eq) / eq[:-1]
    down = rets[rets < 0.0]
    stats["sortino"] = round(float(np.mean(rets) / np.std(down) * np.sqrt(252.0)) if (len(down) > 0 and np.std(down) > 0) else 0.0, 2)
    stats["interest_paid"] = 0.0
    stats["max_debt_ratio"] = 0.0

    s_eq = pd.Series(eq, index=dates_bt)
    ann = s_eq.resample("YE").last().pct_change()
    ann.iloc[0] = s_eq.resample("YE").last().iloc[0] / INITIAL_CAPITAL - 1.0
    ann_dict = {str(d.year): round(float(v) * 100.0, 1) for d, v in ann.items()}
    peak = np.maximum.accumulate(eq)
    dd_series = (eq / peak - 1.0) * 100.0
    pts_idx = list(range(0, len(d), 5))
    if pts_idx[-1] != len(d) - 1:
        pts_idx.append(len(d) - 1)
    curve = [{"d": dates_bt[j].strftime("%Y-%m-%d"), "v": round(float(eq[j]), 2), "dd": round(float(dd_series[j]), 2)} for j in pts_idx]
    return {"stats": stats, "annual": ann_dict, "curve": curve, "trades": trades}


def build_html(engines: dict, live_info: dict, start_date: str, end_date: str) -> str:
    payload_json = json.dumps(engines, separators=(",", ":"), ensure_ascii=False)
    live_json = json.dumps(live_info, separators=(",", ":"), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SOXL信号跨标的组合对比: VGT vs VGT+20%TQQQ(融资) vs VGT+20%TECL(融资)</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0d1117; --card: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
      --green: #3fb950; --red: #f85149; --gold: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 18px 24px 40px;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .header {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 12px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    select {{
      background: var(--card); color: var(--text); border: 1px solid var(--border);
      padding: 8px 12px; border-radius: 8px; font-size: 13px; cursor: pointer;
    }}
    .card {{
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 14px 16px; margin-bottom: 14px;
    }}
    .card-title {{ font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--text); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(255,255,255,0.02); }}
    tr.highlight {{ background: rgba(63, 185, 80, 0.09); }}
    .pos {{ color: var(--green); font-weight: 600; }}
    .neg {{ color: var(--red); font-weight: 600; }}
    .chart-box {{ height: 420px; width: 100%; }}
    .chart-box-sm {{ height: 280px; width: 100%; }}
    .live-grid {{
      display: grid; grid-template-columns: auto 1fr; gap: 14px; align-items: center;
    }}
    .live-badge {{
      padding: 10px 16px; border-radius: 8px; font-weight: 800; font-size: 18px; white-space: nowrap;
    }}
    .tone-buy {{ background: #238636; color: #fff; }}
    .tone-sell {{ background: #da3633; color: #fff; }}
    .tone-hold {{ background: #1f6feb; color: #fff; }}
    .tone-wait {{ background: #9e6a03; color: #fff; }}
    .live-meta {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 14px; }}
    .live-meta b {{ color: var(--text); }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>SOXL 信号跨标的组合回测对比（{start_date} ~ {end_date}）</h1>
      <div class="sub">初始本金 $10,000 | 严格因果次日开盘成交 | 双边 0.05% 手续费 | 融资部分按年化 6.5% 计息（空仓期 0 利息）</div>
    </div>
    <div>
      <span style="color:var(--muted);font-size:13px;margin-right:6px;">SOXL 信号引擎:</span>
      <select id="engineSelect">
        <option value="H5" selected>策略 4: H5 双引擎 (33笔, 全天候趋势+大底+恐慌底)</option>
        <option value="C2">旗舰策略: C2+Deep85_HL+TP3h(50%+10%) (5战全胜大周期)</option>
      </select>
    </div>
  </div>

  <div id="liveCard" class="card"></div>

  <div class="card">
    <div class="card-title">核心收益与风控指标对比</div>
    <table>
      <thead>
        <tr>
          <th>组合 / 策略</th>
          <th>终值 ($)</th>
          <th>CAGR</th>
          <th>MaxDD</th>
          <th>Calmar</th>
          <th>Sharpe</th>
          <th>Sortino</th>
          <th>最差年份</th>
          <th>交易 / 胜率</th>
          <th>最大负债率</th>
          <th>累计融资利息</th>
        </tr>
      </thead>
      <tbody id="metricsBody"></tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-title">净值曲线对比（LOG 对数坐标）</div>
    <div id="eqChart" class="chart-box"></div>
  </div>

  <div class="card">
    <div class="card-title">历史回撤曲线对比 (%)</div>
    <div id="ddChart" class="chart-box-sm"></div>
  </div>

  <div class="card">
    <div class="card-title">逐年收益率对比 (2015–2026)</div>
    <div style="overflow-x:auto;">
      <table>
        <thead id="annHead"></thead>
        <tbody id="annBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    const LIVE = {live_json};
    const COLORS = {{
      "VGT_Spot": "#58a6ff",
      "VGT_20TQQQ": "#d29922",
      "VGT_30TQQQ": "#e3b341",
      "VGT_40TQQQ": "#f0883e",
      "VGT_20TECL": "#3fb950",
      "VGT_30TECL": "#2ea043",
      "VGT_40TECL": "#00e676",
      "TQQQ_100": "#a371f7",
      "TECL_100": "#ff7b72",
      "BH_VGT": "#8b949e"
    }};

    function render() {{
      const eng = document.getElementById("engineSelect").value;
      const items = DATA[eng];
      const ls = LIVE.engines[eng];
      const px = LIVE.prices;
      const posTxt = ls.pos_mode === "FLAT"
        ? "空仓 (0% 现货 / 0% 融资)"
        : `持仓 100% VGT + 融资 20%~40% TECL/TQQQ (入场 ${{ls.entry_date}})`;
      let nextOp = ls.badge;
      if (ls.action === "BUY") nextOp = "下周/次日开盘：买入 100% VGT + 融资买入 TECL/TQQQ";
      else if (ls.action === "SELL") nextOp = "下周/次日开盘：全部平仓并还清融资";
      else if (ls.action === "HOLD") nextOp = "下周操作：继续持有 VGT + 融资仓位";
      else nextOp = "下周操作：保持空仓观望 (0 利息)";
      document.getElementById("liveCard").innerHTML = `
        <div class="live-meta">
          <span>最新数据: <b>${{LIVE.last_date}}</b></span>
          <span>VGT <b>$${{px.VGT.toFixed(2)}}</b></span>
          <span>TECL <b>$${{px.TECL.toFixed(2)}}</b></span>
          <span>TQQQ <b>$${{px.TQQQ.toFixed(2)}}</b></span>
          <span>SOXL <b>$${{px.SOXL.toFixed(2)}}</b></span>
        </div>
        <div class="live-grid">
          <div class="live-badge tone-${{ls.tone}}">${{ls.badge}}</div>
          <div>
            <div style="font-weight:700;font-size:15px;margin-bottom:4px;">当前持仓: ${{posTxt}} ｜ ${{nextOp}}</div>
            <div style="color:var(--muted);font-size:13px;">SOXL信号依据: ${{ls.reason}}</div>
          </div>
        </div>
      `;

      // Metrics Table
      const tbody = document.getElementById("metricsBody");
      tbody.innerHTML = items.map(it => {{
        const s = it.stats;
        const hl = it.key.includes("TECL") && it.key.startsWith("VGT_") ? "highlight" : "";
        return `<tr class="${{hl}}">
          <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${{COLORS[it.key]}};margin-right:6px;"></span><b>${{it.label}}</b></td>
          <td class="pos">$${{s.final.toLocaleString("en-US", {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</td>
          <td>${{s.cagr.toFixed(2)}}%</td>
          <td class="neg">${{s.max_dd.toFixed(2)}}%</td>
          <td><b>${{s.calmar.toFixed(3)}}</b></td>
          <td>${{s.sharpe.toFixed(2)}}</td>
          <td>${{s.sortino.toFixed(2)}}</td>
          <td class="${{s.worst_year >= 0 ? 'pos' : 'neg'}}">${{s.worst_year.toFixed(1)}}%</td>
          <td>${{s.trades}} / ${{s.win_rate.toFixed(1)}}%</td>
          <td>${{s.max_debt_ratio.toFixed(2)}}%</td>
          <td>$${{s.interest_paid.toLocaleString("en-US", {{minimumFractionDigits:0, maximumFractionDigits:0}})}}</td>
        </tr>`;
      }}).join("");

      // Equity Chart (LOG)
      const eqTraces = items.map(it => ({{
        type: "scatter", mode: "lines", name: it.label,
        x: it.curve.map(p => p.d), y: it.curve.map(p => p.v),
        line: {{ color: COLORS[it.key], width: (it.key.startsWith("VGT_20") || it.key.startsWith("VGT_30") || it.key.startsWith("VGT_40")) ? 2.4 : 1.6, dash: it.key === "BH_VGT" ? "dot" : "solid" }}
      }}));
      Plotly.newPlot("eqChart", eqTraces, {{
        paper_bgcolor: "#161b22", plot_bgcolor: "#0d1117", font: {{ color: "#e6edf3" }},
        margin: {{ l: 65, r: 20, t: 15, b: 35 }},
        yaxis: {{ type: "log", gridcolor: "#21262d", tickprefix: "$" }},
        xaxis: {{ gridcolor: "#21262d" }},
        legend: {{ orientation: "h", y: 1.08 }}
      }}, {{ responsive: true }});

      // Drawdown Chart
      const ddTraces = items.map(it => ({{
        type: "scatter", mode: "lines", name: it.label,
        x: it.curve.map(p => p.d), y: it.curve.map(p => p.dd),
        line: {{ color: COLORS[it.key], width: (it.key.startsWith("VGT_20") || it.key.startsWith("VGT_30") || it.key.startsWith("VGT_40")) ? 2.0 : 1.3, dash: it.key === "BH_VGT" ? "dot" : "solid" }}
      }}));
      Plotly.newPlot("ddChart", ddTraces, {{
        paper_bgcolor: "#161b22", plot_bgcolor: "#0d1117", font: {{ color: "#e6edf3" }},
        margin: {{ l: 55, r: 20, t: 10, b: 30 }},
        yaxis: {{ gridcolor: "#21262d", ticksuffix: "%" }},
        xaxis: {{ gridcolor: "#21262d" }},
        showlegend: false
      }}, {{ responsive: true }});

      // Annual Returns Table
      const years = Object.keys(items[0].annual);
      document.getElementById("annHead").innerHTML = "<tr><th>组合</th>" + years.map(y => `<th>${{y}}</th>`).join("") + "</tr>";
      document.getElementById("annBody").innerHTML = items.map(it => {{
        const hl = it.key.includes("TECL") && it.key.startsWith("VGT_") ? "highlight" : "";
        return `<tr class="${{hl}}"><td><b>${{it.label}}</b></td>` + years.map(y => {{
          const v = it.annual[y] ?? 0;
          return `<td class="${{v >= 0 ? 'pos' : 'neg'}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(1)}}%</td>`;
        }}).join("") + "</tr>";
      }}).join("");
    }}

    document.getElementById("engineSelect").addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""


def _action_emoji(action: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵"}.get(action, "🟡")


def format_vgt_tg_summary(live_info: dict, engines: dict) -> str:
    """Format vivid, scannable weekly VGT+TECL/TQQQ margin status for Telegram."""
    px = live_info["prices"]
    h5 = live_info["engines"]["H5"]
    c2 = live_info["engines"]["C2"]
    tecl_30 = next(x["stats"] for x in engines["H5"] if x["key"] == "VGT_30TECL")
    lines = [
        f"📊 VGT + TECL 融资组合周报 ({live_info['last_date']})",
        "",
        "💵 最新收盘价",
        f"• VGT: ${px['VGT']:.2f}  |  TECL: ${px['TECL']:.2f}",
        f"• TQQQ: ${px['TQQQ']:.2f}  |  SOXL: ${px['SOXL']:.2f}",
        "",
        "⚡ 【H5 双引擎信号】",
        f"• 当前持仓: {h5['position_label']}",
        f"• 下周操作: {_action_emoji(h5['action'])} {h5['badge']}",
        f"• 信号依据: {h5['reason']}",
        "",
        "🎯 【C2 大周期旗舰】",
        f"• 当前持仓: {c2['position_label']}",
        f"• 下周操作: {_action_emoji(c2['action'])} {c2['badge']}",
        f"• 信号依据: {c2['reason']}",
        "",
        "📈 回测参考 (H5 · VGT + 30% TECL 融资)",
        f"• 终值: ${tecl_30['final']:,.0f} | CAGR: {tecl_30['cagr']}% | MaxDD: {tecl_30['max_dd']}%",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VGT+TECL/TQQQ margin comparison HTML and optional TG alert.")
    parser.add_argument("--notify-tg", action="store_true", help="Send weekly status summary to Telegram.")
    args = parser.parse_args()

    df_soxl = load_daily_adjusted(DATA_DIR / "SOXL.csv")
    df_smh = load_daily_adjusted(DATA_DIR / "SMH.csv")
    df_vgt = load_daily_adjusted(DATA_DIR / "VGT.csv")
    df_tqqq = load_daily_adjusted(DATA_DIR / "TQQQ.csv")
    df_tecl = load_daily_adjusted(DATA_DIR / "TECL.csv")

    start_date = "2015-01-02"
    end_date = str(min(df_soxl["Date"].max(), df_vgt["Date"].max(), df_tqqq["Date"].max(), df_tecl["Date"].max()).date())
    feats = build_causal_features(df_soxl, df_smh)

    soxl_h5 = simulate_h5_adapt_hv(feats, start_date=start_date, end_date=end_date)
    soxl_c2 = simulate_tp3h(
        feats,
        use_f1_filter=True,
        start_date=start_date,
        end_date=end_date,
        entry_col_override="C2_Deep85HL",
        mode_tag_override="BOT_C2",
        strategy_id="C2_Deep85HL",
        tp_split_lottery_pct=0.0,
    )

    bh_vgt = simulate_buy_hold(df_vgt, start_date, end_date)
    engines = {}
    for eng_key, soxl_res in [("H5", soxl_h5), ("C2", soxl_c2)]:
        print(f"\n=== SOXL Signal Engine: {eng_key} ({start_date} ~ {end_date}) ===")
        rows = [
            ("VGT_40TECL", "VGT + 40% TECL (融资)", simulate_spot_or_margin(df_vgt, df_tecl, 0.40, soxl_res, start_date, end_date)),
            ("VGT_30TECL", "VGT + 30% TECL (融资)", simulate_spot_or_margin(df_vgt, df_tecl, 0.30, soxl_res, start_date, end_date)),
            ("VGT_20TECL", "VGT + 20% TECL (融资)", simulate_spot_or_margin(df_vgt, df_tecl, 0.20, soxl_res, start_date, end_date)),
            ("VGT_40TQQQ", "VGT + 40% TQQQ (融资)", simulate_spot_or_margin(df_vgt, df_tqqq, 0.40, soxl_res, start_date, end_date)),
            ("VGT_30TQQQ", "VGT + 30% TQQQ (融资)", simulate_spot_or_margin(df_vgt, df_tqqq, 0.30, soxl_res, start_date, end_date)),
            ("VGT_20TQQQ", "VGT + 20% TQQQ (融资)", simulate_spot_or_margin(df_vgt, df_tqqq, 0.20, soxl_res, start_date, end_date)),
            ("VGT_Spot", "SOXL信号 → 100% VGT 现货", simulate_spot_or_margin(df_vgt, None, 0.0, soxl_res, start_date, end_date)),
            ("TECL_100", "SOXL信号 → 100% TECL (参考)", simulate_spot_or_margin(df_tecl, None, 0.0, soxl_res, start_date, end_date)),
            ("TQQQ_100", "SOXL信号 → 100% TQQQ (参考)", simulate_spot_or_margin(df_tqqq, None, 0.0, soxl_res, start_date, end_date)),
            ("BH_VGT", "Buy & Hold VGT (死拿基准)", bh_vgt),
        ]
        engines[eng_key] = [{"key": k, "label": lbl, **res} for k, lbl, res in rows]
        for k, lbl, res in rows:
            s = res["stats"]
            print(
                f"  {lbl:28s} | Final=${s['final']:12,.2f} | CAGR={s['cagr']:6.2f}% | "
                f"MaxDD={s['max_dd']:6.2f}% | Calmar={s['calmar']:5.3f} | Sharpe={s['sharpe']:4.2f} | "
                f"Sortino={s['sortino']:4.2f} | WorstYr={s['worst_year']:6.1f}% | MaxDebt={s['max_debt_ratio']:5.2f}% | Int=${s['interest_paid']:,.0f}"
            )

    live_info = {
        "last_date": end_date,
        "prices": {
            "VGT": round(float(df_vgt[df_vgt["Date"] <= pd.Timestamp(end_date)]["Close"].iloc[-1]), 2),
            "TECL": round(float(df_tecl[df_tecl["Date"] <= pd.Timestamp(end_date)]["Close"].iloc[-1]), 2),
            "TQQQ": round(float(df_tqqq[df_tqqq["Date"] <= pd.Timestamp(end_date)]["Close"].iloc[-1]), 2),
            "SOXL": round(float(df_soxl[df_soxl["Date"] <= pd.Timestamp(end_date)]["Close"].iloc[-1]), 2),
        },
        "engines": {
            "H5": soxl_h5["live_status"],
            "C2": soxl_c2["live_status"],
        },
    }

    html = build_html(engines, live_info, start_date, end_date)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nGenerated {OUTPUT_HTML}")

    if args.notify_tg:
        tg_msg = format_vgt_tg_summary(live_info, engines)
        ok = send_tg_notification(tg_msg)
        print(f"[Telegram] sent={ok}")


if __name__ == "__main__":
    main()
