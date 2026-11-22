"""Build a standalone interactive HTML chart from local CSV stock data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "tickers.json"
DEFAULT_OUTPUT_HTML = Path(__file__).resolve().parent.parent / "chart.html"


def load_adjusted_ohlcv(csv_path: Path) -> list[dict]:
    """Load CSV and split-adjust OHLC using Adj Close / Close ratio."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Adjust OHLC for splits/dividends so historical charts (e.g. TQQQ/SOXL) are continuous.
    if "Adj Close" in df.columns and "Close" in df.columns:
        ratio = (df["Adj Close"] / df["Close"]).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = (df[col] * ratio).round(4)

    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "d": row.Date,
                "o": round(float(row.Open), 4),
                "h": round(float(row.High), 4),
                "l": round(float(row.Low), 4),
                "c": round(float(row.Close), 4),
                "v": int(row.Volume) if pd.notna(row.Volume) else 0,
            }
        )
    return records


def build_html(stock_data: dict[str, list[dict]]) -> str:
    """Render standalone HTML with embedded stock data and Plotly.js."""
    data_json = json.dumps(stock_data, separators=(",", ":"), allow_nan=False)
    tickers = list(stock_data.keys())
    options_html = "\n".join(
        f'          <option value="{t}">{t}</option>' for t in tickers
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ApexStock — 行情与技术指标分析</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-bg: rgba(88, 166, 255, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 14px;
      padding: 10px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      font-size: 13px;
    }}
    .group {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .group-label {{
      color: var(--muted);
      font-weight: 600;
      margin-right: 2px;
    }}
    select {{
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn-group {{
      display: inline-flex;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }}
    .btn-group button {{
      background: var(--bg);
      color: var(--muted);
      border: none;
      border-right: 1px solid var(--border);
      padding: 5px 11px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-group button:last-child {{ border-right: none; }}
    .btn-group button.active {{
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
    }}
    .toggle-pill {{
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
      transition: all 0.15s;
    }}
    .toggle-pill input {{
      display: none;
    }}
    .toggle-pill.checked {{
      border-color: var(--accent);
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
    }}
    .divider {{
      width: 1px;
      height: 22px;
      background: var(--border);
    }}
    #chart {{
      flex: 1;
      width: 100%;
      min-height: 0;
    }}
    .status-bar {{
      padding: 5px 16px;
      background: var(--panel);
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="group">
      <span class="group-label">标的</span>
      <select id="tickerSelect">
{options_html}
      </select>
    </div>

    <div class="group">
      <span class="group-label">周期</span>
      <div class="btn-group" id="tfGroup">
        <button data-tf="D">日线</button>
        <button data-tf="W" class="active">周线</button>
        <button data-tf="M">月线</button>
      </div>
    </div>

    <div class="group">
      <label class="toggle-pill" id="logLabel">
        <input type="checkbox" id="logToggle" />
        <span>LOG 对数坐标</span>
      </label>
    </div>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">主图叠加</span>
      <label class="toggle-pill checked"><input type="checkbox" class="ind-check" value="sma" checked /><span>SMA (30周/200周)</span></label>
      <label class="toggle-pill"><input type="checkbox" class="ind-check" value="ema" /><span>EMA (12/26)</span></label>
      <label class="toggle-pill"><input type="checkbox" class="ind-check" value="bb" /><span>Bollinger (20,2)</span></label>
      <label class="toggle-pill checked"><input type="checkbox" class="ind-check" value="cur_month" checked /><span>当前月份线</span></label>
    </div>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">副图指标</span>
      <label class="toggle-pill"><input type="checkbox" class="ind-check" value="volume" /><span>Volume</span></label>
      <label class="toggle-pill checked"><input type="checkbox" class="ind-check" value="macd" checked /><span>MACD</span></label>
      <label class="toggle-pill checked"><input type="checkbox" class="ind-check" value="rsi" checked /><span>RSI (14)</span></label>
      <label class="toggle-pill checked"><input type="checkbox" class="ind-check" value="fisher" checked /><span>Fisher (10)</span></label>
      <label class="toggle-pill"><input type="checkbox" class="ind-check" value="atr" /><span>ATR (14)</span></label>
      <label class="toggle-pill"><input type="checkbox" class="ind-check" value="vol" /><span>Volatility (20)</span></label>
    </div>
  </div>

  <div id="chart"></div>

  <div class="status-bar">
    <span id="statusLeft">数据已内嵌（前复权 OHLC）</span>
    <span>提示：更新 CSV 后运行 python3 tools/build_chart.py 即可同步最新行情</span>
  </div>

  <script>
    const RAW_DATA = {data_json};

    let currentTicker = document.getElementById("tickerSelect").value;
    let currentTF = "W";

    // --- Resampling ---
    function getPeriodKey(dateStr, tf) {{
      if (tf === "D") return dateStr;
      if (tf === "M") return dateStr.slice(0, 7);
      // Weekly: ISO week Monday key
      const [y, m, d] = dateStr.split("-").map(Number);
      const dt = new Date(Date.UTC(y, m - 1, d));
      const day = dt.getUTCDay() || 7;
      dt.setUTCDate(dt.getUTCDate() - day + 1);
      return dt.toISOString().slice(0, 10);
    }}

    function resampleBars(bars, tf) {{
      if (tf === "D") return bars;
      const grouped = [];
      let curKey = null;
      let cur = null;
      for (const b of bars) {{
        const key = getPeriodKey(b.d, tf);
        if (key !== curKey) {{
          if (cur) grouped.push(cur);
          curKey = key;
          cur = {{ d: b.d, o: b.o, h: b.h, l: b.l, c: b.c, v: b.v }};
        }} else {{
          cur.d = b.d;
          if (b.h > cur.h) cur.h = b.h;
          if (b.l < cur.l) cur.l = b.l;
          cur.c = b.c;
          cur.v += b.v;
        }}
      }}
      if (cur) grouped.push(cur);
      return grouped;
    }}

    // --- Indicator Calculations ---
    function calcSMA(arr, period) {{
      const out = new Array(arr.length).fill(null);
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {{
        sum += arr[i];
        if (i >= period) sum -= arr[i - period];
        if (i >= period - 1) out[i] = sum / period;
      }}
      return out;
    }}

    function calcEMA(arr, period) {{
      const out = new Array(arr.length).fill(null);
      const k = 2 / (period + 1);
      let prev = null;
      for (let i = 0; i < arr.length; i++) {{
        if (i === 0) {{
          prev = arr[i];
        }} else {{
          prev = arr[i] * k + prev * (1 - k);
        }}
        if (i >= period - 1) out[i] = prev;
      }}
      return out;
    }}

    function calcBollinger(arr, period = 20, numStd = 2) {{
      const mid = calcSMA(arr, period);
      const upper = new Array(arr.length).fill(null);
      const lower = new Array(arr.length).fill(null);
      for (let i = period - 1; i < arr.length; i++) {{
        const m = mid[i];
        let sq = 0;
        for (let j = i - period + 1; j <= i; j++) {{
          const diff = arr[j] - m;
          sq += diff * diff;
        }}
        const std = Math.sqrt(sq / period);
        upper[i] = m + numStd * std;
        lower[i] = m - numStd * std;
      }}
      return {{ mid, upper, lower }};
    }}

    function calcMACD(arr, fast = 12, slow = 26, signal = 9) {{
      const emaFast = [];
      const emaSlow = [];
      const kF = 2 / (fast + 1), kS = 2 / (slow + 1), kSig = 2 / (signal + 1);
      let pf = arr[0], ps = arr[0];
      for (let i = 0; i < arr.length; i++) {{
        pf = i === 0 ? arr[0] : arr[i] * kF + pf * (1 - kF);
        ps = i === 0 ? arr[0] : arr[i] * kS + ps * (1 - kS);
        emaFast.push(pf);
        emaSlow.push(ps);
      }}
      const macdLine = arr.map((_, i) => (i >= slow - 1 ? emaFast[i] - emaSlow[i] : null));
      const sigLine = new Array(arr.length).fill(null);
      const hist = new Array(arr.length).fill(null);
      let pSig = null;
      let validCount = 0;
      for (let i = 0; i < arr.length; i++) {{
        if (macdLine[i] === null) continue;
        validCount++;
        pSig = pSig === null ? macdLine[i] : macdLine[i] * kSig + pSig * (1 - kSig);
        if (validCount >= signal) {{
          sigLine[i] = pSig;
          hist[i] = macdLine[i] - pSig;
        }}
      }}
      return {{ macdLine, sigLine, hist }};
    }}

    function calcRSI(arr, period = 14) {{
      const out = new Array(arr.length).fill(null);
      const alpha = 1 / period;
      let avgGain = 0, avgLoss = 0;
      for (let i = 1; i < arr.length; i++) {{
        const delta = arr[i] - arr[i - 1];
        const gain = delta > 0 ? delta : 0;
        const loss = delta < 0 ? -delta : 0;
        if (i === 1) {{
          avgGain = gain;
          avgLoss = loss;
        }} else {{
          avgGain = alpha * gain + (1 - alpha) * avgGain;
          avgLoss = alpha * loss + (1 - alpha) * avgLoss;
        }}
        if (i >= period) {{
          if (avgLoss === 0) {{
            out[i] = avgGain > 0 ? 100 : 50;
          }} else {{
            const rs = avgGain / avgLoss;
            out[i] = 100 - 100 / (1 + rs);
          }}
        }}
      }}
      return out;
    }}

    function calcFisher(highs, lows, period = 10) {{
      const n = highs.length;
      const fisher = new Array(n).fill(null);
      const signal = new Array(n).fill(null);
      const med = highs.map((h, i) => (h + lows[i]) / 2);
      let prevX = 0, prevF = 0;
      for (let i = period - 1; i < n; i++) {{
        let minV = Infinity, maxV = -Infinity;
        for (let j = i - period + 1; j <= i; j++) {{
          if (med[j] < minV) minV = med[j];
          if (med[j] > maxV) maxV = med[j];
        }}
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
      }}
      return {{ fisher, signal }};
    }}

    function calcATR(highs, lows, closes, period = 14) {{
      const n = closes.length;
      const out = new Array(n).fill(null);
      const alpha = 1 / period;
      let prevAtr = 0;
      for (let i = 0; i < n; i++) {{
        const tr = i === 0
          ? highs[i] - lows[i]
          : Math.max(
              highs[i] - lows[i],
              Math.abs(highs[i] - closes[i - 1]),
              Math.abs(lows[i] - closes[i - 1])
            );
        prevAtr = i === 0 ? tr : alpha * tr + (1 - alpha) * prevAtr;
        if (i >= period - 1) out[i] = prevAtr;
      }}
      return out;
    }}

    function calcVolatility(closes, period = 20, tf = "D") {{
      const annFactor = tf === "D" ? 252 : tf === "W" ? 52 : 12;
      const n = closes.length;
      const out = new Array(n).fill(null);
      const logRet = new Array(n).fill(null);
      for (let i = 1; i < n; i++) {{
        logRet[i] = Math.log(closes[i] / closes[i - 1]);
      }}
      for (let i = period; i < n; i++) {{
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += logRet[j];
        const mean = sum / period;
        let sq = 0;
        for (let j = i - period + 1; j <= i; j++) {{
          const d = logRet[j] - mean;
          sq += d * d;
        }}
        const std = Math.sqrt(sq / (period - 1));
        out[i] = std * Math.sqrt(annFactor) * 100;
      }}
      return out;
    }}

    function getSelectedIndicators() {{
      const set = new Set();
      document.querySelectorAll(".ind-check:checked").forEach(el => set.add(el.value));
      return set;
    }}

    function renderChart() {{
      const rawBars = RAW_DATA[currentTicker] || [];
      const bars = resampleBars(rawBars, currentTF);
      const dates = bars.map(b => b.d);
      const opens = bars.map(b => b.o);
      const highs = bars.map(b => b.h);
      const lows = bars.map(b => b.l);
      const closes = bars.map(b => b.c);
      const vols = bars.map(b => b.v);

      const isLog = document.getElementById("logToggle").checked;
      const inds = getSelectedIndicators();

      const subPanels = ["volume", "macd", "rsi", "fisher", "atr", "vol"].filter(k => inds.has(k));
      const numSub = subPanels.length;

      // Allocate vertical domain: main chart gets remaining height, each subpanel gets a share
      const subHeight = numSub === 0 ? 0 : Math.min(0.16, 0.52 / numSub);
      const gap = 0.025;
      const totalSub = numSub * (subHeight + gap);
      const mainBottom = totalSub;

      const traces = [];
      const layout = {{
        paper_bgcolor: "#0d1117",
        plot_bgcolor: "#0d1117",
        font: {{ color: "#e6edf3", size: 11 }},
        margin: {{ l: 60, r: 24, t: 36, b: 36 }},
        hovermode: "x unified",
        showlegend: true,
        legend: {{
          orientation: "h",
          x: 0,
          y: 1.03,
          bgcolor: "rgba(0,0,0,0)",
          font: {{ size: 11 }},
        }},
        xaxis: {{
          type: "category",
          rangeslider: {{ visible: false }},
          gridcolor: "#21262d",
          showspikes: true,
          spikemode: "across",
          spikecolor: "#8b949e",
          spikethickness: 1,
          nticks: 12,
        }},
        yaxis: {{
          domain: [mainBottom, 1.0],
          type: isLog ? "log" : "linear",
          gridcolor: "#21262d",
          title: isLog ? "价格 (LOG)" : "价格",
          autorange: true,
        }},
        shapes: [],
        annotations: [],
      }};

      // Current month vertical reference line
      if (inds.has("cur_month") && bars.length > 0) {{
        const nowYm = new Date().toISOString().slice(0, 7);
        const targetYm = bars.some(b => b.d.startsWith(nowYm))
          ? nowYm
          : bars[bars.length - 1].d.slice(0, 7);
        const firstIdx = bars.findIndex(b => b.d.startsWith(targetYm));
        if (firstIdx !== -1) {{
          const monthStartX = bars[firstIdx].d;
          layout.shapes.push({{
            type: "line",
            xref: "x",
            x0: monthStartX,
            x1: monthStartX,
            yref: "paper",
            y0: 0,
            y1: 1,
            line: {{ color: "#58a6ff", width: 1.3, dash: "dash" }},
          }});
          layout.annotations.push({{
            xref: "x",
            x: monthStartX,
            yref: "paper",
            y: 1.0,
            text: `本月 (${{targetYm}})`,
            showarrow: false,
            xanchor: "right",
            yanchor: "bottom",
            font: {{ color: "#58a6ff", size: 11 }},
          }});
        }}
      }}

      // Main Candlestick
      traces.push({{
        type: "candlestick",
        name: currentTicker,
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: {{ line: {{ color: "#26a69a" }}, fillcolor: "#26a69a" }},
        decreasing: {{ line: {{ color: "#ef5350" }}, fillcolor: "#ef5350" }},
        yaxis: "y",
      }});

      // Overlays: always use 30-week and 200-week SMA regardless of selected timeframe
      if (inds.has("sma")) {{
        const weeklyBars = resampleBars(rawBars, "W");
        const weeklyCloses = weeklyBars.map(b => b.c);
        const wSma30 = calcSMA(weeklyCloses, 30);
        const wSma200 = calcSMA(weeklyCloses, 200);

        const weekMap = new Map();
        weeklyBars.forEach((wb, i) => {{
          weekMap.set(getPeriodKey(wb.d, "W"), {{ s30: wSma30[i], s200: wSma200[i] }});
        }});

        const sma30Aligned = dates.map(d => {{
          const hit = weekMap.get(getPeriodKey(d, "W"));
          return hit ? hit.s30 : null;
        }});
        const sma200Aligned = dates.map(d => {{
          const hit = weekMap.get(getPeriodKey(d, "W"));
          return hit ? hit.s200 : null;
        }});

        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "30周均线",
          x: dates,
          y: sma30Aligned,
          line: {{ width: 1.4, color: "#f0b90b" }},
          yaxis: "y",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "200周均线",
          x: dates,
          y: sma200Aligned,
          line: {{ width: 1.4, color: "#d2a8ff" }},
          yaxis: "y",
        }});
      }}

      if (inds.has("ema")) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "EMA12",
          x: dates,
          y: calcEMA(closes, 12),
          line: {{ width: 1.3, color: "#ff9800", dash: "dot" }},
          yaxis: "y",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "EMA26",
          x: dates,
          y: calcEMA(closes, 26),
          line: {{ width: 1.3, color: "#00bcd4", dash: "dot" }},
          yaxis: "y",
        }});
      }}

      if (inds.has("bb")) {{
        const bb = calcBollinger(closes, 20, 2);
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "BB Upper",
          x: dates,
          y: bb.upper,
          line: {{ width: 1, color: "rgba(139, 148, 158, 0.75)" }},
          yaxis: "y",
        }});
        traces.push({{
          type: "scatter",
          mode: "lines",
          name: "BB Lower",
          x: dates,
          y: bb.lower,
          line: {{ width: 1, color: "rgba(139, 148, 158, 0.75)" }},
          fill: "tonexty",
          fillcolor: "rgba(88, 166, 255, 0.06)",
          yaxis: "y",
        }});
      }}

      // Subpanels (ordered from top subpanel to bottom)
      subPanels.forEach((panel, idx) => {{
        const axisNum = idx + 2;
        const yName = `yaxis${{axisNum}}`;
        const yRef = `y${{axisNum}}`;
        const top = mainBottom - gap - idx * (subHeight + gap);
        const bottom = Math.max(0, top - subHeight);

        const axisCfg = {{
          domain: [bottom, top],
          gridcolor: "#21262d",
          zerolinecolor: "#30363d",
          autorange: true,
        }};

        if (panel === "volume") {{
          axisCfg.title = "Vol";
          const colors = closes.map((c, i) => (c >= opens[i] ? "rgba(38,166,154,0.65)" : "rgba(239,83,80,0.65)"));
          traces.push({{
            type: "bar",
            name: "Volume",
            x: dates,
            y: vols,
            marker: {{ color: colors }},
            yaxis: yRef,
            showlegend: false,
          }});
        }} else if (panel === "macd") {{
          axisCfg.title = "MACD";
          const m = calcMACD(closes, 12, 26, 9);
          const histColors = m.hist.map(v => (v !== null && v >= 0 ? "rgba(38,166,154,0.7)" : "rgba(239,83,80,0.7)"));
          traces.push({{
            type: "bar",
            name: "MACD Hist",
            x: dates,
            y: m.hist,
            marker: {{ color: histColors }},
            yaxis: yRef,
          }});
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "MACD",
            x: dates,
            y: m.macdLine,
            line: {{ width: 1.3, color: "#58a6ff" }},
            yaxis: yRef,
          }});
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "MACD Signal",
            x: dates,
            y: m.sigLine,
            line: {{ width: 1.3, color: "#ff9800" }},
            yaxis: yRef,
          }});
        }} else if (panel === "rsi") {{
          axisCfg.title = "RSI(14)";
          axisCfg.range = [0, 100];
          axisCfg.autorange = false;
          const r = calcRSI(closes, 14);
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "RSI(14)",
            x: dates,
            y: r,
            line: {{ width: 1.4, color: "#c084fc" }},
            yaxis: yRef,
          }});
          layout.shapes.push(
            {{ type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 70, y1: 70, line: {{ color: "#ef5350", width: 1, dash: "dot" }} }},
            {{ type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 30, y1: 30, line: {{ color: "#26a69a", width: 1, dash: "dot" }} }}
          );
        }} else if (panel === "fisher") {{
          axisCfg.title = "Fisher(10)";
          const f = calcFisher(highs, lows, 10);
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "Fisher",
            x: dates,
            y: f.fisher,
            line: {{ width: 1.4, color: "#2dd4bf" }},
            yaxis: yRef,
          }});
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "Fisher Signal",
            x: dates,
            y: f.signal,
            line: {{ width: 1.2, color: "#f97316", dash: "dot" }},
            yaxis: yRef,
          }});
          layout.shapes.push(
            {{ type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 0, y1: 0, line: {{ color: "#484f58", width: 1, dash: "dash" }} }}
          );
        }} else if (panel === "atr") {{
          axisCfg.title = "ATR(14)";
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "ATR(14)",
            x: dates,
            y: calcATR(highs, lows, closes, 14),
            line: {{ width: 1.4, color: "#eab308" }},
            yaxis: yRef,
          }});
        }} else if (panel === "vol") {{
          axisCfg.title = "Vol(%)";
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: "Volatility(20)",
            x: dates,
            y: calcVolatility(closes, 20, currentTF),
            line: {{ width: 1.4, color: "#f43f5e" }},
            yaxis: yRef,
          }});
        }}

        layout[yName] = axisCfg;
      }});

      Plotly.newPlot("chart", traces, layout, {{ responsive: true, scrollZoom: true }});

      const tfName = currentTF === "D" ? "日线" : currentTF === "W" ? "周线" : "月线";
      const last = bars[bars.length - 1];
      if (last) {{
        document.getElementById("statusLeft").textContent =
          `${{currentTicker}} · ${{tfName}} · 共 ${{bars.length}} 根K线 (${{bars[0].d}} ~ ${{last.d}}) · 最新收盘: ${{last.c.toFixed(2)}}`;
      }}
    }}

    // --- Event Listeners ---
    document.getElementById("tickerSelect").addEventListener("change", e => {{
      currentTicker = e.target.value;
      renderChart();
    }});

    document.querySelectorAll("#tfGroup button").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll("#tfGroup button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentTF = btn.dataset.tf;
        renderChart();
      }});
    }});

    document.getElementById("logToggle").addEventListener("change", e => {{
      document.getElementById("logLabel").classList.toggle("checked", e.target.checked);
      renderChart();
    }});

    document.querySelectorAll(".ind-check").forEach(chk => {{
      chk.addEventListener("change", () => {{
        chk.parentElement.classList.toggle("checked", chk.checked);
        renderChart();
      }});
    }});

    renderChart();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone interactive HTML chart.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_HTML)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    raw_data_dir = Path(config.get("data_dir", "../data"))
    data_dir = raw_data_dir if raw_data_dir.is_absolute() else (config_path.parent / raw_data_dir).resolve()

    stock_data: dict[str, list[dict]] = {}
    for ticker in config.get("tickers", []):
        t = str(ticker).strip().upper()
        csv_path = data_dir / f"{t}.csv"
        if csv_path.exists():
            stock_data[t] = load_adjusted_ohlcv(csv_path)

    if not stock_data:
        raise RuntimeError(f"No CSV data found in {data_dir}")

    html = build_html(stock_data)
    args.output.write_text(html, encoding="utf-8")
    print(f"Generated {args.output} with {len(stock_data)} tickers: {', '.join(stock_data.keys())}")


if __name__ == "__main__":
    main()
