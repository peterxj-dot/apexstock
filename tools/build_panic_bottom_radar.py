"""Build standalone Panic Bottom State Machine page (panic_bottom_radar.html)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_HTML = ROOT / "panic_bottom_radar.html"


def _load_close(ticker: str, use_adj: bool = False) -> pd.Series:
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    col = "Adj Close" if (use_adj and "Adj Close" in df.columns) else "Close"
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if use_adj and s.empty:
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    return s


def compute_panic_bottom_data() -> dict:
    soxl = _load_close("SOXL", use_adj=True)
    smh = _load_close("SMH", use_adj=True)
    qqq = _load_close("QQQ", use_adj=True)
    vix = _load_close("^VIX", use_adj=False)
    vix3m = _load_close("^VIX3M", use_adj=False)
    vxn = _load_close("^VXN", use_adj=False)

    df = pd.DataFrame({
        "SOXL": soxl,
        "SMH": smh,
        "QQQ": qqq,
        "VIX": vix,
        "VIX3M": vix3m,
        "VXN": vxn,
    }).sort_index()
    df = df.loc[soxl.index].ffill()

    df["smh_sma200"] = df["SMH"].rolling(200).mean()
    df["smh_ema5"] = df["SMH"].ewm(span=5, adjust=False).mean()
    df["qqq_hv20"] = np.log(df["QQQ"] / df["QQQ"].shift(1)).rolling(20).std() * np.sqrt(252) * 100.0
    df["vix_ratio"] = df["VIX"] / df["VIX3M"]
    df["vr_5d_max"] = df["vix_ratio"].rolling(5).max()
    df["iv_hv_spread"] = df["VXN"] - df["qqq_hv20"]

    df["is_q1"] = (df["vix_ratio"] >= 1.00) & (df["iv_hv_spread"] >= 4.0)
    df["is_q2"] = (df["vix_ratio"] >= 1.00) & (df["iv_hv_spread"] < 0.0)
    df["is_deep_q2"] = (df["vix_ratio"] >= 1.00) & (df["iv_hv_spread"] <= -8.0)

    last_dt = df.index[-1]
    last = df.iloc[-1]
    vr = float(last["vix_ratio"])
    sp = float(last["iv_hv_spread"])
    smh_vs_200_pct = round(float((float(last["SMH"]) / float(last["smh_sma200"]) - 1.0) * 100.0), 1)

    recent20 = df.tail(20)
    q1_days_20d = int(recent20["is_q1"].sum())
    q2_days_20d = int(recent20["is_q2"].sum())
    deep_q2_hit_20d = bool(recent20["is_deep_q2"].any())
    min_sp_20d = round(float(recent20["iv_hv_spread"].min()), 1)
    min_sp_date = recent20["iv_hv_spread"].idxmin().strftime("%Y-%m-%d")
    max_vr_20d = round(float(recent20["vix_ratio"].max()), 3)
    max_vr_date = recent20["vix_ratio"].idxmax().strftime("%Y-%m-%d")

    vr_turned_down = bool(vr < float(last["vr_5d_max"]) * 0.96)
    vr_uninverted = bool(deep_q2_hit_20d and vr < 1.00)
    smh_above_ema5 = bool(float(last["SMH"]) > float(last["smh_ema5"]))
    soxl_dd_20d = round(float((float(last["SOXL"]) / float(recent20["SOXL"].max()) - 1.0) * 100.0), 1)
    smh_dd_20d = round(float((float(last["SMH"]) / float(recent20["SMH"].max()) - 1.0) * 100.0), 1)

    if not deep_q2_hit_20d and q1_days_20d > 0 and sp >= 4.0:
        b_stage = "STAGE1_DROP"
        b_label = "阶段1 · 主跌/保费飙升（严禁接飞刀）"
        b_tone = "sell"
        b_desc = "期权正溢价持续扩大，类似 2020-02/03-12 机构仍在加价买保险，尚未进入恐慌出清底。"
    elif deep_q2_hit_20d and (vr_uninverted or (vr_turned_down and smh_above_ema5)):
        b_stage = "STAGE4_CONFIRM"
        b_label = "阶段4 · 恐慌底部右侧确认（可分批/配合主策略进场）"
        b_tone = "buy"
        b_desc = "已打出深度负溢价(≤-8)且期权恐慌退潮/现货站上5日EMA，历史5次真恐慌底均在此区间确立。"
    elif deep_q2_hit_20d and (vr_turned_down or smh_above_ema5):
        b_stage = "STAGE3_FORMING"
        b_label = "阶段3 · 底部雏形初现（半左侧观察/轻仓试探）"
        b_tone = "hold"
        b_desc = "已满足深度负溢价(≤-8)并出现初步企稳信号，等待 VIX/3M 进一步回落或 SMH 站稳 5日EMA 确认。"
    elif deep_q2_hit_20d:
        b_stage = "STAGE2_CAPITULATION"
        b_label = "阶段2 · 深度恐慌砸盘中（勿割肉 · 等拐头）"
        b_tone = "hold"
        b_desc = f"近20日已打出深度负溢价 {min_sp_20d:+.1f}（跨过 -8 真恐慌门槛），但 VIX/3M 尚未明显拐头且 SMH 未站上 5日EMA，参考 2020-03 / 2025-04 需耐心等右侧信号。"
    else:
        b_stage = "NORMAL"
        b_label = "常态运行（无急跌恐慌底部形态）"
        b_tone = "neutral"
        b_desc = "近20日未出现 VIX倒挂 + 深度负溢价(≤-8) 急跌出清形态；注意慢熊底(如2022-10)需以主策略现货信号为准。"

    steps = [
        {"name": "1. 深度恐慌门槛 (SP ≤ -8)", "ok": deep_q2_hit_20d, "detail": f"近20日最低 {min_sp_20d:+.1f} ({min_sp_date})"},
        {"name": "2. 恐慌退潮 (VIX/3M 距5日峰回落≥4%)", "ok": vr_turned_down, "detail": f"当前 {vr:.3f} vs 5日峰 {float(last['vr_5d_max']):.3f}"},
        {"name": "3. 倒挂解除 (VIX/3M < 1.00)", "ok": vr_uninverted, "detail": f"当前 VIX/VIX3M = {vr:.3f}"},
        {"name": "4. 现货止跌 (SMH > 5日EMA)", "ok": smh_above_ema5, "detail": f"SMH ${float(last['SMH']):.2f} vs 5EMA ${float(last['smh_ema5']):.2f}"},
    ]

    plot_df = df.loc["2018-01-01":].copy()
    q1_pts = plot_df[plot_df["is_q1"]]
    q2d_pts = plot_df[plot_df["is_deep_q2"]]

    return {
        "live": {
            "last_date": last_dt.strftime("%Y-%m-%d"),
            "qqq_close": round(float(last["QQQ"]), 2),
            "smh_close": round(float(last["SMH"]), 2),
            "soxl_close": round(float(last["SOXL"]), 2),
            "smh_vs_200_pct": smh_vs_200_pct,
            "vix_ratio": round(vr, 3),
            "iv_hv_spread": round(sp, 1),
            "stage": b_stage,
            "stage_label": b_label,
            "stage_tone": b_tone,
            "stage_desc": b_desc,
            "q1_days_20d": q1_days_20d,
            "q2_days_20d": q2_days_20d,
            "deep_q2_hit_20d": deep_q2_hit_20d,
            "min_sp_20d": min_sp_20d,
            "min_sp_date": min_sp_date,
            "max_vr_20d": max_vr_20d,
            "max_vr_date": max_vr_date,
            "soxl_dd_20d": soxl_dd_20d,
            "smh_dd_20d": smh_dd_20d,
            "steps": steps,
        },
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in plot_df.index],
            "vix_ratio": [round(float(x), 3) for x in plot_df["vix_ratio"]],
            "iv_hv_spread": [round(float(x), 1) if pd.notna(x) else None for x in plot_df["iv_hv_spread"]],
            "smh": [round(float(x), 2) for x in plot_df["SMH"]],
            "smh_ema5": [round(float(x), 2) for x in plot_df["smh_ema5"]],
            "soxl": [round(float(x), 2) for x in plot_df["SOXL"]],
            "q1_dates": [d.strftime("%Y-%m-%d") for d in q1_pts.index],
            "q1_soxl": [round(float(x), 2) for x in q1_pts["SOXL"]],
            "q2d_dates": [d.strftime("%Y-%m-%d") for d in q2d_pts.index],
            "q2d_soxl": [round(float(x), 2) for x in q2d_pts["SOXL"]],
        },
    }


def render_html(payload: dict) -> str:
    live_json = json.dumps(payload["live"], separators=(",", ":"), ensure_ascii=False)
    series_json = json.dumps(payload["series"], separators=(",", ":"), ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ApexStock · 期权恐慌出清与周期底部状态机</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
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
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5;
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }}
    .banner {{
      background: linear-gradient(135deg, #0B1739 0%, #13224E 100%);
      border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px;
    }}
    .banner h1 {{ font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .pill {{
      display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px;
      border-radius: 999px; font-size: 12px; font-weight: 600;
    }}
    .pill-buy {{ background: rgba(20,202,116,0.16); color: var(--green); border: 1px solid rgba(20,202,116,0.35); }}
    .pill-hold {{ background: rgba(253,181,42,0.16); color: var(--amber); border: 1px solid rgba(253,181,42,0.35); }}
    .pill-sell {{ background: rgba(255,90,121,0.16); color: var(--red); border: 1px solid rgba(255,90,121,0.35); }}
    .pill-neutral {{ background: rgba(126,137,172,0.14); color: var(--text-sec); border: 1px solid rgba(126,137,172,0.3); }}

    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
    .card-title {{
      font-size: 14px; font-weight: 600; margin-bottom: 10px;
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
    }}
    .card-sub {{ font-size: 11px; color: var(--muted); font-weight: 400; }}

    .steps-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }}
    .step-box {{
      background: rgba(8,16,40,0.6); border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px;
    }}
    .step-box.ok {{ border-color: rgba(20,202,116,0.55); background: rgba(20,202,116,0.08); }}
    .step-top {{ font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px; }}
    .step-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

    .toolbar {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }}
    .btn-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .rng-btn {{
      background: rgba(8,16,40,0.7); color: var(--text-sec); border: 1px solid var(--border);
      padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer;
    }}
    .rng-btn.active {{
      background: rgba(0,194,255,0.16); color: var(--cyan); border-color: var(--cyan);
    }}

    .charts-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .span-2 {{ grid-column: 1 / -1; }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .guide-card {{
      background: rgba(8,16,40,0.55); border: 1px solid rgba(52,59,79,0.7); border-radius: 10px; padding: 12px;
      font-size: 12px; color: var(--text-sec);
    }}
    .guide-title {{ font-weight: 700; margin-bottom: 5px; font-size: 13px; }}

    @media (max-width: 1100px) {{
      .charts-col, .steps-grid, .guide-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .span-2 {{ grid-column: auto; }}
    }}
    @media (max-width: 680px) {{
      .charts-col, .steps-grid, .guide-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="banner">
      <div>
        <h1>
          <span>期权恐慌出清 &amp; 周期底部状态机</span>
          <span id="stagePill" class="pill"></span>
        </h1>
        <div class="sub" id="metaSub"></div>
      </div>
      <div style="text-align:right;font-size:12px;color:var(--text-sec);">
        <div>VIX/VIX3M: <b id="vrLive" style="color:var(--amber);"></b> · 保费溢价: <b id="spLive" style="color:var(--cyan);"></b></div>
        <div style="color:var(--muted);" id="cycleCtx"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>近期 20 日恐慌演化 &amp; 底部 4 步进度检查</span>
        <span class="card-sub" id="bottomSummary"></span>
      </div>
      <div style="font-size:12px;color:var(--text-sec);" id="stageDesc"></div>
      <div class="steps-grid" id="bottomSteps"></div>
    </div>

    <div class="card toolbar">
      <div>
        <span style="font-size:14px;font-weight:600;">阈值分界曲线追踪（取代文本表格）</span>
        <span class="card-sub" style="margin-left:8px;">含 VIX/3M 倒挂线 (1.00 / 0.95) 与 保费溢价阈值线 (+8 / +4 / 0 / -8)</span>
      </div>
      <div class="btn-group" id="rangeBtns">
        <button class="rng-btn" data-days="20">近 20 日</button>
        <button class="rng-btn active" data-days="60">近 60 日</button>
        <button class="rng-btn" data-days="252">近 1 年</button>
        <button class="rng-btn" data-days="0">2018+ 全历史</button>
      </div>
    </div>

    <div class="charts-col">
      <div class="card">
        <div class="card-title">
          <span>VIX / VIX3M 期限结构曲线</span>
          <span class="card-sub">红线 1.00 倒挂恐慌线 · 黄线 0.95 警戒线</span>
        </div>
        <div id="chartVixRatio" style="height:310px;"></div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>VXN - QQQ_HV20 保费溢价曲线</span>
          <span class="card-sub">红线 +8 / 黄线 +4 (保费抢买) · 紫线 -8 (深度恐慌出清底)</span>
        </div>
        <div id="chartSpread" style="height:310px;"></div>
      </div>

      <div class="card span-2">
        <div class="card-title">
          <span>SMH vs 5日EMA 现货止跌对照 &amp; SOXL 恐慌信号标注</span>
          <span class="card-sub">红三角 = Q1 主跌保费飙升 · 紫菱形 = Q2 深度恐慌出清 (SP ≤ -8)</span>
        </div>
        <div id="chartSpot" style="height:320px;"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>如何用这两条带分界线的曲线判断“阶段1→阶段4”？</span>
        <span class="card-sub">2015–2026 年 8 次 Q2 恐慌周期总结</span>
      </div>
      <div class="guide-grid">
        <div class="guide-card">
          <div class="guide-title" style="color:var(--red);">阶段 1 · 主跌保费飙升</div>
          <div><b>VIX/3M ≥ 1.00</b> 且 <b>VXN-HV20 ≥ +4</b>（红线之上 + 橙/红线之上）。说明机构仍在加价买保险，如 2020-03-09、03-12，<b>严禁接飞刀</b>。</div>
        </div>
        <div class="guide-card">
          <div class="guide-title" style="color:var(--amber);">阶段 2 · 深度恐慌砸盘</div>
          <div><b>VXN-HV20 跌破 -8</b> 紫色阈值线！说明现货已砸出暴跌而期权不再加价追空。史上 5 次跨过 -8 均命中真恐慌大底，此时<b>勿割肉、等拐头</b>。</div>
        </div>
        <div class="guide-card">
          <div class="guide-title" style="color:var(--cyan);">阶段 3 · 底部雏形初现</div>
          <div>在打出 ≤-8 后，<b>VIX/3M 较 5 日峰值回落 ≥ 4%</b>，或 <b>SMH 站上 5 日 EMA</b>。如 2020-03-20 (VIX/3M 从 1.22 降至 1.007)，适合分批左侧观察。</div>
        </div>
        <div class="guide-card">
          <div class="guide-title" style="color:var(--green);">阶段 4 · 右侧确认进场</div>
          <div>满足 <b>SP ≤ -8</b> 后，<b>VIX/3M 回落 + SMH 站上 5日EMA</b>（或 VIX/3M 跌破 1.00 解除倒挂），配合主策略执行抄底（历史 60 日 SMH 均值 +20.8%）。</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const PANIC_BOTTOM_LIVE = {live_json};
    const S = {series_json};

    function pillCls(tone) {{
      if (tone === "sell") return "pill-sell";
      if (tone === "hold") return "pill-hold";
      if (tone === "buy") return "pill-buy";
      return "pill-neutral";
    }}

    const spEl = document.getElementById("stagePill");
    spEl.textContent = PANIC_BOTTOM_LIVE.stage_label;
    spEl.className = "pill " + pillCls(PANIC_BOTTOM_LIVE.stage_tone);

    document.getElementById("metaSub").textContent =
      `数据截至 ${{PANIC_BOTTOM_LIVE.last_date}} · QQQ $${{PANIC_BOTTOM_LIVE.qqq_close}} · SMH $${{PANIC_BOTTOM_LIVE.smh_close}} · SOXL $${{PANIC_BOTTOM_LIVE.soxl_close}}`;
    document.getElementById("vrLive").textContent = PANIC_BOTTOM_LIVE.vix_ratio.toFixed(3);
    document.getElementById("spLive").textContent =
      (PANIC_BOTTOM_LIVE.iv_hv_spread >= 0 ? "+" : "") + PANIC_BOTTOM_LIVE.iv_hv_spread.toFixed(1);
    document.getElementById("cycleCtx").textContent =
      `SMH 相对 200日线: ${{PANIC_BOTTOM_LIVE.smh_vs_200_pct >= 0 ? "+" : ""}}${{PANIC_BOTTOM_LIVE.smh_vs_200_pct}}%`;

    document.getElementById("bottomSummary").textContent =
      `近20日回撤: SOXL ${{PANIC_BOTTOM_LIVE.soxl_dd_20d}}% / SMH ${{PANIC_BOTTOM_LIVE.smh_dd_20d}}% · Q1=${{PANIC_BOTTOM_LIVE.q1_days_20d}}天 · Q2=${{PANIC_BOTTOM_LIVE.q2_days_20d}}天 · VIX/3M峰=${{PANIC_BOTTOM_LIVE.max_vr_20d}} (${{PANIC_BOTTOM_LIVE.max_vr_date}})`;
    document.getElementById("stageDesc").textContent = PANIC_BOTTOM_LIVE.stage_desc;

    document.getElementById("bottomSteps").innerHTML = (PANIC_BOTTOM_LIVE.steps || []).map(st => `
      <div class="step-box ${{st.ok ? "ok" : ""}}">
        <div class="step-top" style="color:${{st.ok ? "var(--green)" : "var(--text-sec)"}};">
          <span>${{st.ok ? "✓" : "○"}}</span>
          <span>${{st.name}}</span>
        </div>
        <div class="step-sub">${{st.detail}}</div>
      </div>
    `).join("");

    function sliceSeries(days) {{
      const total = S.dates.length;
      const start = days > 0 ? Math.max(0, total - days) : 0;
      const d0 = S.dates[start];
      const q1x = [], q1y = [], q2x = [], q2y = [];
      for (let i = 0; i < S.q1_dates.length; i++) {{
        if (S.q1_dates[i] >= d0) {{ q1x.push(S.q1_dates[i]); q1y.push(S.q1_soxl[i]); }}
      }}
      for (let i = 0; i < S.q2d_dates.length; i++) {{
        if (S.q2d_dates[i] >= d0) {{ q2x.push(S.q2d_dates[i]); q2y.push(S.q2d_soxl[i]); }}
      }}
      return {{
        dates: S.dates.slice(start),
        vr: S.vix_ratio.slice(start),
        sp: S.iv_hv_spread.slice(start),
        smh: S.smh.slice(start),
        smh_ema5: S.smh_ema5.slice(start),
        soxl: S.soxl.slice(start),
        q1x, q1y, q2x, q2y,
      }};
    }}

    function hLine(y, color, dash) {{
      return {{
        type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: y, y1: y,
        line: {{ color, width: 1.4, dash }}
      }};
    }}
    function hLabel(y, text, color) {{
      return {{
        xref: "paper", x: 0.995, yref: "y", y: y, xanchor: "right", yanchor: "bottom",
        text, showarrow: false, font: {{ size: 10, color }}
      }};
    }}

    function renderCharts(days) {{
      const w = sliceSeries(days);
      const base = {{
        paper_bgcolor: "#0B1739", plot_bgcolor: "#0B1739", font: {{ color: "#D1DBF9", size: 11 }},
        margin: {{ l: 48, r: 16, t: 16, b: 28 }},
        xaxis: {{ gridcolor: "rgba(52,59,79,0.45)" }},
        legend: {{ orientation: "h", y: 1.12, x: 0 }},
      }};

      // 1. VIX / VIX3M with threshold zones
      const vrMax = Math.max(1.12, ...w.vr.filter(v => v != null)) * 1.03;
      const vrMin = Math.min(0.82, ...w.vr.filter(v => v != null)) * 0.97;
      Plotly.react("chartVixRatio", [
        {{ type: "scatter", mode: "lines", name: "VIX / VIX3M", x: w.dates, y: w.vr, line: {{ color: "#FDB52A", width: 2.2 }} }},
      ], {{
        ...base,
        yaxis: {{ gridcolor: "rgba(52,59,79,0.45)", range: [vrMin, vrMax] }},
        shapes: [
          {{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 1.0, y1: vrMax, fillcolor: "rgba(255,90,121,0.10)", line: {{ width: 0 }} }},
          hLine(1.0, "#FF5A79", "dash"),
          hLine(0.95, "#FDB52A", "dot"),
        ],
        annotations: [
          hLabel(1.0, "倒挂恐慌线 1.00", "#FF5A79"),
          hLabel(0.95, "警戒线 0.95", "#FDB52A"),
        ],
      }}, {{ responsive: true }});

      // 2. VXN - QQQ_HV20 Spread with threshold lines
      const spVals = w.sp.filter(v => v != null);
      const spMax = Math.max(11, ...spVals) + 2;
      const spMin = Math.min(-12, ...spVals) - 2;
      Plotly.react("chartSpread", [
        {{ type: "scatter", mode: "lines", name: "VXN - QQQ_HV20", x: w.dates, y: w.sp, line: {{ color: "#00C2FF", width: 2.2 }} }},
      ], {{
        ...base,
        yaxis: {{ gridcolor: "rgba(52,59,79,0.45)", range: [spMin, spMax] }},
        shapes: [
          {{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: spMin, y1: -8.0, fillcolor: "rgba(203,60,255,0.11)", line: {{ width: 0 }} }},
          {{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 4.0, y1: spMax, fillcolor: "rgba(255,90,121,0.08)", line: {{ width: 0 }} }},
          hLine(8.0, "#FF5A79", "dash"),
          hLine(4.0, "#FDB52A", "dot"),
          hLine(0.0, "#7E89AC", "solid"),
          hLine(-8.0, "#CB3CFF", "dash"),
        ],
        annotations: [
          hLabel(8.0, "保费暴涨 +8", "#FF5A79"),
          hLabel(4.0, "Q1 正溢价 +4", "#FDB52A"),
          hLabel(0.0, "零轴 0", "#7E89AC"),
          hLabel(-8.0, "深度恐慌出清 -8", "#CB3CFF"),
        ],
      }}, {{ responsive: true }});

      // 3. SMH vs 5EMA + SOXL + signals
      Plotly.react("chartSpot", [
        {{ type: "scatter", mode: "lines", name: "SMH", x: w.dates, y: w.smh, line: {{ color: "#00C2FF", width: 2.0 }}, yaxis: "y" }},
        {{ type: "scatter", mode: "lines", name: "SMH 5日EMA", x: w.dates, y: w.smh_ema5, line: {{ color: "#14CA74", width: 1.5, dash: "dash" }}, yaxis: "y" }},
        {{ type: "scatter", mode: "lines", name: "SOXL (右轴)", x: w.dates, y: w.soxl, line: {{ color: "rgba(203,60,255,0.55)", width: 1.5 }}, yaxis: "y2" }},
        {{ type: "scatter", mode: "markers", name: "Q1 主跌警报 (右轴)", x: w.q1x, y: w.q1y, marker: {{ color: "#FF5A79", size: 8, symbol: "triangle-down" }}, yaxis: "y2" }},
        {{ type: "scatter", mode: "markers", name: "Q2 深度恐慌 ≤-8 (右轴)", x: w.q2x, y: w.q2y, marker: {{ color: "#CB3CFF", size: 9, symbol: "diamond", line: {{ color: "#fff", width: 1 }} }}, yaxis: "y2" }},
      ], {{
        ...base,
        margin: {{ l: 52, r: 52, t: 16, b: 28 }},
        yaxis: {{ gridcolor: "rgba(52,59,79,0.45)", tickprefix: "$" }},
        yaxis2: {{ overlaying: "y", side: "right", showgrid: false, tickprefix: "$", type: days === 0 ? "log" : "linear" }},
      }}, {{ responsive: true }});
    }}

    document.querySelectorAll(".rng-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll(".rng-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderCharts(Number(btn.dataset.days));
      }});
    }});

    renderCharts(60);
  </script>
</body>
</html>
"""


def main() -> None:
    payload = compute_panic_bottom_data()
    html_out = render_html(payload)
    OUTPUT_HTML.write_text(html_out, encoding="utf-8")
    live = payload["live"]
    print(
        f"Generated {OUTPUT_HTML} ({live['last_date']} | {live['stage_label']} | "
        f"VIX/3M={live['vix_ratio']} | SP={live['iv_hv_spread']:+.1f})"
    )


if __name__ == "__main__":
    main()
