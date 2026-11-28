"""Build standalone Chip Structure (1-Year & 6-Month Profit Ratio) page (chip_structure.html)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import DATA_DIR, load_daily_adjusted

OUTPUT_HTML = ROOT / "chip_structure.html"

TICKER_SPECS = [
    ("SMH", "SMH · 1倍半导体基准", 252, 120, "2011-01-01", "1年获利盘≤5% 对应 SOXL 未来1年胜率90%，核心大底尺子"),
    ("SOXL", "SOXL · 3倍半导体", 252, 120, "2011-01-01", "3倍杠杆含波动损耗，大周期筹码须结合 SMH 同看"),
    ("MSTR", "MSTR · 比特币杠杆股", 252, 120, "2020-08-01", "熊市易早早归零，须结合减半买点窗与 T0 信号"),
    ("BTC-USD", "BTC-USD · 比特币现货", 365, 180, "2015-09-17", "7×24 现货成交额加权（365日=1年 / 180日=半年）"),
    ("QQQ", "QQQ · 纳指100基准", 252, 120, "2011-01-01", "美股大科技现货基准筹码"),
    ("TQQQ", "TQQQ · 3倍纳指", 252, 120, "2011-01-01", "3倍纳指杠杆筹码，结合 QQQ 同看"),
    ("VGT", "VGT · 美股科技ETF", 252, 120, "2011-01-01", "1倍科技现货仓位筹码参考"),
    ("TECL", "TECL · 3倍科技", 252, 120, "2011-01-01", "3倍科技杠杆筹码，结合 VGT/SMH 同看"),
]


def _profit_ratio_series(df: pd.DataFrame, lookback: int, decay: float) -> pd.Series:
    """Strictly causal rolling turnover-decayed dollar-volume profit ratio (%)."""
    tp = ((df["Low"] + df["High"] + 2.0 * df["Close"]) / 4.0).to_numpy(dtype=np.float64)
    dv = df["Volume"].to_numpy(dtype=np.float64) * tp
    closes = df["Close"].to_numpy(dtype=np.float64)
    n = len(df)
    w_base = decay ** np.arange(lookback - 1, -1, -1)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback - 1, n):
        sl_tp = tp[i - lookback + 1 : i + 1]
        sl_dv = dv[i - lookback + 1 : i + 1] * w_base
        tot = sl_dv.sum()
        if tot > 0.0:
            out[i] = float(sl_dv[sl_tp <= closes[i]].sum() / tot * 100.0)
    return pd.Series(out, index=df["Date"])


def _latest_profile_stats(df: pd.DataFrame, lookback: int, decay: float = 0.995, n_bins: int = 80) -> tuple[float, float]:
    """Compute latest 1Y triangular dollar-volume average cost and POC price."""
    sub = df.iloc[max(0, len(df) - lookback) :]
    lo_min = float(sub["Low"].min())
    hi_max = float(sub["High"].max())
    if hi_max <= lo_min:
        last_c = float(sub["Close"].iloc[-1])
        return round(last_c, 2), round(last_c, 2)

    edges = np.linspace(lo_min, hi_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    chips = np.zeros(n_bins, dtype=np.float64)

    highs = sub["High"].to_numpy(dtype=np.float64)
    lows = sub["Low"].to_numpy(dtype=np.float64)
    closes = sub["Close"].to_numpy(dtype=np.float64)
    vols = sub["Volume"].to_numpy(dtype=np.float64)
    n = len(sub)

    for k in range(n):
        chips *= decay
        lo, hi, c = lows[k], highs[k], closes[k]
        tp = (lo + hi + 2.0 * c) / 4.0
        dv = vols[k] * tp
        if dv <= 0.0:
            continue
        mask = (edges[1:] >= lo) & (edges[:-1] <= hi)
        if not np.any(mask):
            b = int(np.clip(np.searchsorted(edges, c) - 1, 0, n_bins - 1))
            chips[b] += dv
        else:
            bc = centers[mask]
            span = max(hi - lo, 1e-6)
            w = np.maximum(0.1, 1.0 - np.abs(bc - tp) / span)
            w /= w.sum()
            chips[mask] += dv * w

    tot = chips.sum()
    if tot <= 0.0:
        last_c = float(closes[-1])
        return round(last_c, 2), round(last_c, 2)

    avg_cost = float((chips * centers).sum() / tot)
    poc_px = float(centers[int(np.argmax(chips))])
    return round(avg_cost, 2), round(poc_px, 2)


def _status_meta(p1y: float) -> tuple[str, str]:
    if p1y <= 5.0:
        return "极端绝望 (≤5%)", "buy"
    if p1y <= 20.0:
        return "低位偏弱 (5-20%)", "hold"
    if p1y >= 85.0:
        return "高获利拥挤 (≥85%)", "sell"
    return "中性震荡", "neutral"


def build_chip_payload() -> dict:
    items: list[dict] = []
    for sym, title, w1y, w6m, start_date, note in TICKER_SPECS:
        df = load_daily_adjusted(DATA_DIR / f"{sym}.csv").sort_values("Date").reset_index(drop=True)
        p1y_s = _profit_ratio_series(df, lookback=w1y, decay=0.995)
        p6m_s = _profit_ratio_series(df, lookback=w6m, decay=0.990)

        avg_cost, poc_px = _latest_profile_stats(df, lookback=w1y, decay=0.995)

        work = pd.DataFrame({
            "Date": df["Date"],
            "Close": df["Close"],
            "p1y": p1y_s.to_numpy(dtype=np.float64),
            "p6m": p6m_s.to_numpy(dtype=np.float64),
        })
        work = work[work["Date"] >= pd.Timestamp(start_date)].dropna(subset=["p1y", "p6m"]).reset_index(drop=True)

        last_dt_str = str(work["Date"].iloc[-1].date())
        last_c = round(float(work["Close"].iloc[-1]), 2)
        last_p1y = round(float(work["p1y"].iloc[-1]), 1)
        last_p6m = round(float(work["p6m"].iloc[-1]), 1)
        vs_avg_pct = round((last_c / avg_cost - 1.0) * 100.0, 1) if avg_cost > 0 else 0.0
        st_label, st_tone = _status_meta(last_p1y)

        # Resample to monthly (ME) taking the last available bar of each month
        mon = (
            work.set_index("Date")
            .resample("ME")
            .last()
            .dropna(subset=["Close", "p1y", "p6m"])
        )

        items.append({
            "ticker": sym,
            "title": title,
            "note": note,
            "w1y": w1y,
            "w6m": w6m,
            "last_date": last_dt_str,
            "close": last_c,
            "p1y_last": last_p1y,
            "p6m_last": last_p6m,
            "avg_cost": avg_cost,
            "vs_avg_pct": vs_avg_pct,
            "poc": poc_px,
            "status_label": st_label,
            "status_tone": st_tone,
            "dates": [d.strftime("%Y-%m") for d in mon.index],
            "close_series": [round(float(x), 2) for x in mon["Close"]],
            "p1y_series": [round(float(x), 1) for x in mon["p1y"]],
            "p6m_series": [round(float(x), 1) for x in mon["p6m"]],
        })

    return {
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M %Z"),
        "items": items,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ApexStock · 筹码结构（1年 &amp; 半年获利盘比例）</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #081028;
      --surface: #0B1739;
      --border: #343B4F;
      --text: #FFFFFF;
      --text-sec: #D1DBF9;
      --muted: #7E89AC;
      --purple: #CB3CFF;
      --cyan: #00C2FF;
      --green: #14CA74;
      --amber: #FDB52A;
      --red: #FF5A79;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5;
    }
    .wrap { max-width: 1460px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
    .banner {
      background: linear-gradient(135deg, #0B1739 0%, #13224E 100%);
      border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px;
    }
    .banner h1 { font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .toolbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .btn-group { display: flex; gap: 8px; flex-wrap: wrap; }
    .rng-btn {
      background: rgba(8,16,40,0.7); color: var(--text-sec); border: 1px solid var(--border);
      padding: 6px 13px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer;
    }
    .rng-btn.active {
      background: rgba(203,60,255,0.18); color: var(--purple); border-color: var(--purple);
    }
    .toggle-lbl {
      display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-sec);
      background: rgba(8,16,40,0.7); border: 1px solid var(--border); padding: 6px 11px; border-radius: 8px; cursor: pointer;
    }
    .pill {
      display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px;
      border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap;
    }
    .pill-buy { background: rgba(20,202,116,0.16); color: var(--green); border: 1px solid rgba(20,202,116,0.35); }
    .pill-hold { background: rgba(253,181,42,0.16); color: var(--amber); border: 1px solid rgba(253,181,42,0.35); }
    .pill-sell { background: rgba(255,90,121,0.16); color: var(--red); border: 1px solid rgba(255,90,121,0.35); }
    .pill-neutral { background: rgba(126,137,172,0.14); color: var(--text-sec); border: 1px solid rgba(126,137,172,0.3); }

    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 15px 16px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(52,59,79,0.55); white-space: nowrap; }
    th:first-child, td:first-child, th:last-child, td:last-child { text-align: left; }
    th { color: var(--muted); font-weight: 600; font-size: 11px; }
    tbody tr { cursor: pointer; transition: background 0.15s; }
    tbody tr:hover { background: rgba(203,60,255,0.08); }

    .charts-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    @media (max-width: 1050px) { .charts-grid { grid-template-columns: 1fr; } }

    .chart-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
    .chart-title { font-size: 14px; font-weight: 700; color: var(--text); }
    .chart-note { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .badges { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .stat-chip {
      font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px;
      background: rgba(8,16,40,0.7); border: 1px solid var(--border); color: var(--text-sec);
    }
    .stat-chip.p1y { border-color: rgba(203,60,255,0.45); color: #E5A5FF; }
    .stat-chip.p6m { border-color: rgba(0,194,255,0.45); color: #7BE2FF; }
    .plot-box { width: 100%; height: 300px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="banner">
      <div>
        <h1>
          <span>筹码结构 · 月线 LOG &amp; 1年/半年获利盘比例监控</span>
          <span class="pill pill-neutral" id="genTag"></span>
        </h1>
        <div class="sub">
          月线 · LOG 对数价格（月末收盘抽样）· 换手衰减成交额加权模型（≤5% 极端绝望套牢区 / ≥90% 高获利拥挤区）· 点击汇总行可跳转到对应图表
        </div>
      </div>
      <div class="toolbar">
        <div class="btn-group" id="rangeBtns">
          <button class="rng-btn" data-days="730">近2年</button>
          <button class="rng-btn active" data-days="1095">近3年</button>
          <button class="rng-btn" data-days="1825">近5年</button>
          <button class="rng-btn" data-days="3650">近10年</button>
          <button class="rng-btn" data-days="0">全部</button>
        </div>
        <label class="toggle-lbl">
          <input type="checkbox" id="priceToggle" checked />
          <span>显示收盘价背景线</span>
        </label>
        <label class="toggle-lbl">
          <input type="checkbox" id="logToggle" checked />
          <span>LOG 对数价格</span>
        </label>
      </div>
    </div>

    <div class="card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>标的</th>
              <th>最新日期</th>
              <th>收盘价</th>
              <th>1年获利盘%</th>
              <th>半年获利盘%</th>
              <th>1年均成本 (现价偏离)</th>
              <th>主力筹码峰 POC</th>
              <th>筹码状态</th>
            </tr>
          </thead>
          <tbody id="summaryBody"></tbody>
        </table>
      </div>
    </div>

    <div class="charts-grid" id="chartsGrid"></div>
  </div>

  <script>
    const CHIP_DATA = __CHIP_DATA_JSON__;
    const ITEMS = CHIP_DATA.items || [];
    document.getElementById("genTag").textContent = `更新于 ${CHIP_DATA.generated_at}`;

    let activeDays = 1095;
    let showPrice = true;
    let useLog = true;

    function fmtNum(v, d = 2) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    }

    function renderSummaryTable() {
      const tbody = document.getElementById("summaryBody");
      tbody.innerHTML = ITEMS.map((it, idx) => `
        <tr onclick="document.getElementById('card_${idx}').scrollIntoView({behavior:'smooth', block:'center'})">
          <td><b>${it.title}</b></td>
          <td style="color:#7E89AC">${it.last_date}</td>
          <td><b>$${fmtNum(it.close)}</b></td>
          <td style="color:#E5A5FF;font-weight:700">${it.p1y_last.toFixed(1)}%</td>
          <td style="color:#7BE2FF;font-weight:700">${it.p6m_last.toFixed(1)}%</td>
          <td>$${fmtNum(it.avg_cost)} <span style="color:${it.vs_avg_pct >= 0 ? '#14CA74' : '#FF5A79'}">(${it.vs_avg_pct >= 0 ? '+' : ''}${it.vs_avg_pct.toFixed(1)}%)</span></td>
          <td>$${fmtNum(it.poc)}</td>
          <td><span class="pill pill-${it.status_tone}">${it.status_label}</span></td>
        </tr>
      `).join("");
    }

    function initCards() {
      const grid = document.getElementById("chartsGrid");
      grid.innerHTML = ITEMS.map((it, idx) => `
        <div class="card" id="card_${idx}">
          <div class="chart-head">
            <div>
              <div class="chart-title">${it.title}</div>
              <div class="chart-note">${it.note}</div>
            </div>
            <div class="badges">
              <span class="stat-chip p1y">1年获利 ${it.p1y_last.toFixed(1)}%</span>
              <span class="stat-chip p6m">半年获利 ${it.p6m_last.toFixed(1)}%</span>
              <span class="stat-chip">均成本 $${fmtNum(it.avg_cost)}</span>
              <span class="stat-chip">POC $${fmtNum(it.poc)}</span>
            </div>
          </div>
          <div id="plot_${idx}" class="plot-box"></div>
        </div>
      `).join("");
    }

    function sliceRange(it, days) {
      if (!days || days <= 0) {
        return { d: it.dates, p1: it.p1y_series, p6: it.p6m_series, c: it.close_series };
      }
      const parseMo = (s) => new Date(s.length === 7 ? `${s}-01` : s).getTime();
      const lastTs = parseMo(it.dates[it.dates.length - 1]);
      const cutoff = lastTs - days * 86400 * 1000;
      let startIdx = 0;
      for (let i = 0; i < it.dates.length; i++) {
        if (parseMo(it.dates[i]) >= cutoff) {
          startIdx = i;
          break;
        }
      }
      return {
        d: it.dates.slice(startIdx),
        p1: it.p1y_series.slice(startIdx),
        p6: it.p6m_series.slice(startIdx),
        c: it.close_series.slice(startIdx),
      };
    }

    function renderAllCharts() {
      ITEMS.forEach((it, idx) => {
        const sl = sliceRange(it, activeDays);
        const traces = [
          {
            type: "scatter",
            mode: "lines",
            name: `1年获利盘 (${it.w1y}日)`,
            x: sl.d,
            y: sl.p1,
            line: { color: "#CB3CFF", width: 2.0 },
            yaxis: "y",
            hovertemplate: "1年获利: %{y:.1f}%<extra></extra>",
          },
          {
            type: "scatter",
            mode: "lines",
            name: `半年获利盘 (${it.w6m}日)`,
            x: sl.d,
            y: sl.p6,
            line: { color: "#00C2FF", width: 1.6 },
            yaxis: "y",
            hovertemplate: "半年获利: %{y:.1f}%<extra></extra>",
          },
        ];
        if (showPrice) {
          traces.push({
            type: "scatter",
            mode: "lines",
            name: `月收盘 (${useLog ? "LOG" : "线性"}, 右轴)`,
            x: sl.d,
            y: sl.c,
            line: { color: "rgba(209,219,249,0.58)", width: 1.5 },
            yaxis: "y2",
            hovertemplate: "月收盘: $%{y:,.2f}<extra></extra>",
          });
        }
        const layout = {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(8,16,40,0.45)",
          font: { color: "#D1DBF9", size: 11 },
          margin: { l: 42, r: showPrice ? 58 : 14, t: 10, b: 28 },
          hovermode: "x unified",
          legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
          xaxis: { gridcolor: "rgba(52,59,79,0.45)", showspikes: true },
          yaxis: {
            range: [0, 102],
            ticksuffix: "%",
            gridcolor: "rgba(52,59,79,0.45)",
          },
          yaxis2: {
            type: useLog ? "log" : "linear",
            overlaying: "y",
            side: "right",
            showgrid: false,
            tickprefix: "$",
            visible: showPrice,
          },
          shapes: [
            { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 5, fillcolor: "rgba(20,202,116,0.12)", line: { width: 0 }, layer: "below" },
            { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 5, y1: 5, line: { color: "rgba(20,202,116,0.6)", width: 1, dash: "dot" } },
            { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 90, y1: 100, fillcolor: "rgba(255,90,121,0.10)", line: { width: 0 }, layer: "below" },
            { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 90, y1: 90, line: { color: "rgba(255,90,121,0.55)", width: 1, dash: "dot" } },
          ],
        };
        Plotly.react(`plot_${idx}`, traces, layout, { responsive: true, displayModeBar: false });
      });
    }

    document.getElementById("rangeBtns").addEventListener("click", e => {
      const btn = e.target.closest(".rng-btn");
      if (!btn) return;
      document.querySelectorAll(".rng-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeDays = Number(btn.getAttribute("data-days"));
      renderAllCharts();
    });

    document.getElementById("priceToggle").addEventListener("change", e => {
      showPrice = Boolean(e.target.checked);
      renderAllCharts();
    });

    document.getElementById("logToggle").addEventListener("change", e => {
      useLog = Boolean(e.target.checked);
      renderAllCharts();
    });

    renderSummaryTable();
    initCards();
    renderAllCharts();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_chip_payload()
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__CHIP_DATA_JSON__", data_json)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    for it in payload["items"]:
        print(
            f"[{it['ticker']:7s}] {it['last_date']} Close=${it['close']:,.2f} | 1Y={it['p1y_last']:4.1f}% | 6M={it['p6m_last']:4.1f}% | Avg=${it['avg_cost']:,.2f} | POC=${it['poc']:,.2f}"
        )
    print(f"Generated {OUTPUT_HTML} ({len(payload['items'])} tickers)")


if __name__ == "__main__":
    main()
