"""Build a unified Dashdark-X inspired ApexStock Command Center (dashboard.html) without modifying existing pages."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_HTML = ROOT / "dashboard.html"


def _extract_json_var(html_path: Path, var_name: str) -> dict | list | None:
    if not html_path.is_file():
        return None
    text = html_path.read_text(encoding="utf-8")
    m = re.search(rf"const {var_name}\s*=\s*(.*?);\n", text, re.S)
    if not m:
        return None
    return json.loads(m.group(1))


def collect_portal_data() -> dict:
    """Read live status and stats from existing generated HTMLs (read-only)."""
    cards = []

    def add_zone_page(key: str, title: str, file_name: str, strat_id: str, accent: str) -> None:
        p = ROOT / file_name
        lm = _extract_json_var(p, "LIVE_MARKET") or {}
        st_map = _extract_json_var(p, "STRATEGIES") or {}
        s_obj = st_map.get(strat_id) or (next(iter(st_map.values())) if st_map else {})
        ls = s_obj.get("live_status", {})
        stats = s_obj.get("stats", {})
        cards.append({
            "key": key,
            "title": title,
            "file": file_name,
            "accent": accent,
            "last_date": lm.get("last_date", "-"),
            "price": lm.get("soxl_close", 0.0),
            "action": ls.get("action", "WAIT"),
            "badge": ls.get("badge", "保持空仓"),
            "tone": ls.get("tone", "wait"),
            "position": ls.get("position_label", "空仓 (0%)"),
            "reason": ls.get("reason", ""),
            "final": stats.get("final", 0.0),
            "cagr": stats.get("cagr", 0.0),
            "max_dd": stats.get("max_dd", 0.0),
            "calmar": stats.get("calmar", 0.0),
            "win_rate": stats.get("win_rate", 0.0),
            "trades": stats.get("trades", 0),
            "curve": s_obj.get("equity_curve", []),
            "extra": lm.get("btc_live") or lm.get("mstr_live") or {},
        })

    add_zone_page("SOXL", "SOXL 半导体三倍", "soxl_monthly_zones.html", "C2_F1_or_MFishLin_TP3h", "#CB3CFF")
    add_zone_page("BTC", "BTC 减半周期", "btc_monthly_zones.html", "BTC_Halv_HV_Combo", "#00C2FF")
    add_zone_page("MSTR", "MSTR mNAV脉冲", "mstr_monthly_zones.html", "Halv_mNAV_Spike170", "#9A91FB")
    add_zone_page("TECL", "TECL (SOXL驱动)", "tecl_monthly_zones.html", "SOXL_H5_to_TECL", "#14CA74")
    add_zone_page("TQQQ", "TQQQ (SOXL驱动)", "tqqq_monthly_zones.html", "SOXL_H5_to_TQQQ", "#FDB52A")

    # VGT + 30% TECL Margin
    vgt_p = ROOT / "vgt_margin_comparison.html"
    vgt_data = _extract_json_var(vgt_p, "DATA") or {}
    vgt_live = _extract_json_var(vgt_p, "LIVE") or {}
    if vgt_data and vgt_live:
        h5_list = vgt_data.get("H5", [])
        tecl30 = next((x for x in h5_list if x.get("key") == "VGT_30TECL"), {})
        ls_h5 = vgt_live.get("engines", {}).get("H5", {})
        st30 = tecl30.get("stats", {})
        cards.append({
            "key": "VGT30",
            "title": "VGT + 30% TECL融资",
            "file": "vgt_margin_comparison.html",
            "accent": "#00C2FF",
            "last_date": vgt_live.get("last_date", "-"),
            "price": vgt_live.get("prices", {}).get("VGT", 0.0),
            "action": ls_h5.get("action", "WAIT"),
            "badge": ls_h5.get("badge", "保持空仓"),
            "tone": ls_h5.get("tone", "wait"),
            "position": ls_h5.get("position_label", "空仓 (0%)"),
            "reason": ls_h5.get("reason", ""),
            "final": st30.get("final", 0.0),
            "cagr": st30.get("cagr", 0.0),
            "max_dd": st30.get("max_dd", 0.0),
            "calmar": st30.get("calmar", 0.0),
            "win_rate": st30.get("win_rate", 0.0),
            "trades": st30.get("trades", 0),
            "curve": tecl30.get("curve", []),
            "extra": {"max_debt_ratio": st30.get("max_debt_ratio", 0.0)},
        })

    return {"cards": cards}


def render_dashboard(portal_data: dict) -> str:
    data_json = json.dumps(portal_data, separators=(",", ":"), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ApexStock X · 量化策略统一驾驶舱</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #081028;
      --sidebar: #081028;
      --surface: #0B1739;
      --surface-alt: #0A1330;
      --border: #343B4F;
      --text: #FFFFFF;
      --text-sec: #D1DBF9;
      --muted: #7E89AC;
      --purple: #CB3CFF;
      --violet: #7F25FB;
      --cyan: #00C2FF;
      --lavender: #9A91FB;
      --green: #14CA74;
      --green-bg: rgba(5, 193, 104, 0.18);
      --amber: #FDB52A;
      --amber-bg: rgba(253, 181, 42, 0.18);
      --red: #FF5A65;
      --red-bg: rgba(255, 90, 101, 0.18);
      --blue-bg: rgba(0, 194, 255, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0; height: 100%;
      background: var(--bg); color: var(--text);
      font-family: "Mona Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .app {{
      display: grid;
      grid-template-columns: 270px 1fr;
      height: 100vh;
      overflow: hidden;
    }}
    /* Sidebar matching Dashdark X */
    .sidebar {{
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      padding: 22px 16px;
      gap: 16px;
      overflow-y: auto;
    }}
    .brand {{
      display: flex; align-items: center; gap: 10px;
      padding: 4px 8px; font-size: 18px; font-weight: 700; letter-spacing: 0.2px;
    }}
    .brand-icon {{
      width: 28px; height: 28px; border-radius: 8px;
      background: linear-gradient(135deg, var(--purple), var(--cyan));
      display: grid; place-items: center; font-weight: 800; font-size: 14px;
      box-shadow: 0 0 16px rgba(203, 60, 255, 0.35);
    }}
    .search-input {{
      width: 100%; padding: 10px 12px; border-radius: 8px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--text); font-size: 13px; outline: none;
    }}
    .search-input::placeholder {{ color: var(--muted); }}
    .nav-label {{
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
      color: var(--muted); padding: 6px 10px 2px;
    }}
    .nav-list {{ display: flex; flex-direction: column; gap: 4px; }}
    .nav-item {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 12px; border-radius: 8px; cursor: pointer;
      color: #AEB9E1; font-size: 13.5px; font-weight: 500;
      border-left: 3px solid transparent; transition: all 0.15s ease;
    }}
    .nav-item:hover {{ background: rgba(11, 23, 57, 0.7); color: #fff; }}
    .nav-item.active {{
      background: var(--surface-alt);
      color: #FFFFFF;
      border-left-color: var(--purple);
      box-shadow: inset 0 0 0 1px rgba(203, 60, 255, 0.2);
    }}
    .nav-dot {{
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    }}
    .sidebar-footer {{
      margin-top: auto; padding-top: 14px; border-top: 1px solid var(--border);
      font-size: 12px; color: var(--muted); line-height: 1.5;
    }}
    .cta-btn {{
      display: block; width: 100%; margin-top: 10px; padding: 10px 12px;
      border: none; border-radius: 8px; cursor: pointer; text-align: center;
      font-weight: 600; font-size: 13px; color: #fff;
      background: linear-gradient(90deg, var(--purple), var(--violet));
      box-shadow: 0 4px 16px rgba(203, 60, 255, 0.28);
    }}
    /* Main Workspace */
    .main {{
      display: flex; flex-direction: column; height: 100vh; overflow: hidden;
      background: var(--bg);
    }}
    .topbar {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 26px; border-bottom: 1px solid var(--border);
      background: rgba(8, 16, 40, 0.92);
    }}
    .topbar h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
    .topbar-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 3px; }}
    .topbar-actions {{ display: flex; gap: 10px; align-items: center; }}
    .btn-sec {{
      background: var(--surface); color: var(--text-sec); border: 1px solid var(--border);
      padding: 8px 14px; border-radius: 8px; font-size: 12.5px; cursor: pointer;
      text-decoration: none;
    }}
    .btn-pri {{
      background: linear-gradient(90deg, var(--purple), var(--violet));
      color: #fff; border: none; padding: 8px 14px; border-radius: 8px;
      font-size: 12.5px; font-weight: 600; cursor: pointer; text-decoration: none;
    }}
    .content {{
      flex: 1; overflow-y: auto; padding: 20px 26px 32px;
    }}
    .content.iframe-mode {{ padding: 0; overflow: hidden; }}
    iframe.subframe {{ width: 100%; height: 100%; border: none; background: #0d1117; }}

    /* KPI Row */
    .kpi-grid {{
      display: grid; grid-template-columns: repeat(4, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 18px;
    }}
    @media (max-width: 1180px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px 18px;
    }}
    .kpi-card {{ cursor: pointer; transition: transform 0.12s ease, border-color 0.15s ease; }}
    .kpi-card:hover {{ transform: translateY(-2px); border-color: var(--purple); }}
    .kpi-head {{
      display: flex; justify-content: space-between; align-items: center;
      font-size: 12.5px; color: var(--muted); margin-bottom: 10px;
    }}
    .kpi-val-row {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
    .kpi-val {{ font-size: 23px; font-weight: 700; letter-spacing: -0.3px; }}
    .pill {{
      font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px;
      display: inline-flex; align-items: center; gap: 4px;
    }}
    .pill-buy {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(20, 202, 116, 0.35); }}
    .pill-sell {{ background: var(--red-bg); color: var(--red); border: 1px solid rgba(255, 90, 101, 0.35); }}
    .pill-hold {{ background: var(--blue-bg); color: var(--cyan); border: 1px solid rgba(0, 194, 255, 0.35); }}
    .pill-wait {{ background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(253, 181, 42, 0.35); }}
    .kpi-foot {{ margin-top: 8px; font-size: 12px; color: var(--muted); }}

    /* Hero 2-column grid */
    .hero-grid {{
      display: grid; grid-template-columns: 1.75fr 1fr; gap: 16px; margin-bottom: 18px;
    }}
    @media (max-width: 1100px) {{ .hero-grid {{ grid-template-columns: 1fr; }} }}
    .card-title {{
      display: flex; justify-content: space-between; align-items: center;
      font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 12px;
    }}
    .card-sub {{ font-size: 12px; color: var(--muted); font-weight: 400; }}
    .stack-col {{ display: flex; flex-direction: column; gap: 16px; }}

    /* Bottom Table */
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid rgba(52, 59, 79, 0.55); text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; background: var(--surface-alt); }}
    tr.row-click {{ cursor: pointer; }}
    tr.row-click:hover {{ background: rgba(203, 60, 255, 0.06); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-icon">A</div>
        <span>ApexStock X</span>
      </div>
      <input id="navSearch" class="search-input" type="text" placeholder="搜索标的或策略页面..." />

      <div class="nav-label">主控制台</div>
      <div class="nav-list" id="navMain">
        <div class="nav-item active" data-target="OVERVIEW">
          <span>📊 全市场驾驶舱</span>
          <span class="nav-dot" style="background: var(--purple);"></span>
        </div>
      </div>

      <div class="nav-label">策略图表页面 (独立原页)</div>
      <div class="nav-list" id="navPages"></div>

      <div class="sidebar-footer">
        <div>已联动数据源:</div>
        <div style="color:var(--text-sec);margin-top:2px;">yfinance · IBKR MCP · SEC EDGAR</div>
        <button class="cta-btn" onclick="openPage('vgt_margin_comparison.html')">查看 VGT+TECL 融资对比 →</button>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <h1 id="topTitle">Welcome back · 量化策略总览</h1>
          <div class="topbar-sub" id="topSub">集中监控 SOXL / TECL / TQQQ / VGT融资 / BTC / MSTR 实盘状态与下个交易日操作</div>
        </div>
        <div class="topbar-actions" id="topActions">
          <button class="btn-sec" onclick="selectNav('OVERVIEW')">返回总览</button>
          <a id="openRawBtn" class="btn-pri" href="#" target="_blank" style="display:none;">在新标签页打开 ↗</a>
        </div>
      </header>

      <section id="overviewSection" class="content">
        <div class="kpi-grid" id="kpiGrid"></div>

        <div class="hero-grid">
          <div class="card">
            <div class="card-title">
              <span>核心策略净值增长曲线 (LOG 对数坐标 · $10,000 起)</span>
              <span class="card-sub">SOXL · BTC · MSTR · VGT+30%TECL</span>
            </div>
            <div id="heroChart" style="height: 360px;"></div>
          </div>

          <div class="stack-col">
            <div class="card">
              <div class="card-title">
                <span>年化复利 (CAGR %) 对比</span>
                <span class="card-sub">全周期实测</span>
              </div>
              <div id="cagrChart" style="height: 155px;"></div>
            </div>
            <div class="card">
              <div class="card-title">
                <span>周期雷达 · BTC & MSTR 减半买点窗</span>
                <span class="card-sub" id="halvTag">实时监控</span>
              </div>
              <div id="radarMini" style="font-size:13px;line-height:1.7;color:var(--text-sec);"></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>全标的今日状态 & 下个交易日操作指令表（点击任意行直接进入该页面）</span>
            <span class="card-sub">严格因果 · 次日开盘执行</span>
          </div>
          <div style="overflow-x:auto;">
            <table>
              <thead>
                <tr>
                  <th>标的 / 策略</th>
                  <th>最新日期</th>
                  <th class="num">最新收盘</th>
                  <th>当前仓位</th>
                  <th>明日 / 下周操作</th>
                  <th class="num">回测终值</th>
                  <th class="num">CAGR</th>
                  <th class="num">MaxDD</th>
                  <th>信号依据</th>
                </tr>
              </thead>
              <tbody id="statusTableBody"></tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="iframeSection" class="content iframe-mode" style="display:none;">
        <iframe id="subFrame" class="subframe" src="about:blank"></iframe>
      </section>
    </main>
  </div>

  <script>
    const PORTAL = {data_json};
    const CARDS = PORTAL.cards || [];

    const PAGE_LIST = [
      {{ id: "soxl_monthly_zones.html", label: "SOXL 月线大周期", dot: "#CB3CFF" }},
      {{ id: "vgt_margin_comparison.html", label: "VGT + TECL/TQQQ 融资", dot: "#00C2FF" }},
      {{ id: "tecl_monthly_zones.html", label: "TECL (SOXL驱动)", dot: "#14CA74" }},
      {{ id: "tqqq_monthly_zones.html", label: "TQQQ (SOXL驱动)", dot: "#FDB52A" }},
      {{ id: "btc_monthly_zones.html", label: "BTC 减半周期", dot: "#00C2FF" }},
      {{ id: "mstr_monthly_zones.html", label: "MSTR mNAV脉冲", dot: "#9A91FB" }},
      {{ id: "chart.html", label: "基础K线对比 (chart)", dot: "#7E89AC" }},
    ];

    function pillClass(tone) {{
      if (tone === "buy") return "pill-buy";
      if (tone === "sell") return "pill-sell";
      if (tone === "hold") return "pill-hold";
      return "pill-wait";
    }}

    function renderSidebar(filterText = "") {{
      const q = filterText.trim().toLowerCase();
      const box = document.getElementById("navPages");
      box.innerHTML = PAGE_LIST
        .filter(p => !q || p.label.toLowerCase().includes(q) || p.id.toLowerCase().includes(q))
        .map(p => `
          <div class="nav-item" data-target="${{p.id}}" onclick="openPage('${{p.id}}')">
            <span>${{p.label}}</span>
            <span class="nav-dot" style="background:${{p.dot}};"></span>
          </div>
        `).join("");
    }}

    function selectNav(target) {{
      document.querySelectorAll(".nav-item").forEach(el => {{
        el.classList.toggle("active", el.getAttribute("data-target") === target);
      }});
      const ov = document.getElementById("overviewSection");
      const ifrSec = document.getElementById("iframeSection");
      const rawBtn = document.getElementById("openRawBtn");
      if (target === "OVERVIEW") {{
        ov.style.display = "block";
        ifrSec.style.display = "none";
        rawBtn.style.display = "none";
        document.getElementById("topTitle").textContent = "Welcome back · 量化策略总览";
        document.getElementById("topSub").textContent = "集中监控 SOXL / TECL / TQQQ / VGT融资 / BTC / MSTR 实盘状态与下个交易日操作";
      }} else {{
        ov.style.display = "none";
        ifrSec.style.display = "block";
        document.getElementById("subFrame").src = target;
        rawBtn.href = target;
        rawBtn.style.display = "inline-block";
        const meta = PAGE_LIST.find(p => p.id === target);
        document.getElementById("topTitle").textContent = meta ? meta.label : target;
        document.getElementById("topSub").textContent = "原页面独立嵌入式视图（未修改原文件，可点击右上角新标签页打开）";
      }}
    }}

    function openPage(file) {{
      selectNav(file);
    }}

    function renderOverview() {{
      const top4 = ["SOXL", "BTC", "MSTR", "VGT30"].map(k => CARDS.find(c => c.key === k)).filter(Boolean);
      document.getElementById("kpiGrid").innerHTML = top4.map(c => `
        <div class="card kpi-card" onclick="openPage('${{c.file}}')">
          <div class="kpi-head">
            <span>${{c.title}}</span>
            <span>${{c.last_date}}</span>
          </div>
          <div class="kpi-val-row">
            <span class="kpi-val">$${{Number(c.price).toLocaleString("en-US", {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</span>
            <span class="pill ${{pillClass(c.tone)}}">${{c.badge}}</span>
          </div>
          <div class="kpi-foot">CAGR <b>${{c.cagr}}%</b> · MaxDD <b>${{c.max_dd}}%</b> · ${{c.position}}</div>
        </div>
      `).join("");

      // Hero equity curves (2015+)
      const heroKeys = ["SOXL", "BTC", "MSTR", "VGT30"];
      const traces = heroKeys.map(k => CARDS.find(c => c.key === k)).filter(c => c && c.curve && c.curve.length).map(c => ({{
        type: "scatter", mode: "lines", name: c.title,
        x: c.curve.map(p => p.d), y: c.curve.map(p => p.v),
        line: {{ color: c.accent, width: 2.3 }},
      }}));
      Plotly.newPlot("heroChart", traces, {{
        paper_bgcolor: "#0B1739", plot_bgcolor: "#0B1739", font: {{ color: "#D1DBF9", size: 11 }},
        margin: {{ l: 55, r: 16, t: 10, b: 30 }},
        yaxis: {{ type: "log", gridcolor: "rgba(52,59,79,0.5)", tickprefix: "$" }},
        xaxis: {{ gridcolor: "rgba(52,59,79,0.5)" }},
        legend: {{ orientation: "h", y: 1.09 }},
      }}, {{ responsive: true }});

      // CAGR Bar chart
      Plotly.newPlot("cagrChart", [{{
        type: "bar",
        x: CARDS.map(c => c.key),
        y: CARDS.map(c => c.cagr),
        marker: {{ color: CARDS.map(c => c.accent) }},
        text: CARDS.map(c => c.cagr.toFixed(0) + "%"),
        textposition: "outside",
      }}], {{
        paper_bgcolor: "#0B1739", plot_bgcolor: "#0B1739", font: {{ color: "#D1DBF9", size: 11 }},
        margin: {{ l: 32, r: 10, t: 18, b: 26 }},
        yaxis: {{ gridcolor: "rgba(52,59,79,0.45)", ticksuffix: "%" }},
        xaxis: {{ gridcolor: "transparent" }},
      }}, {{ responsive: true }});

      // Halving Radar Mini
      const btcCard = CARDS.find(c => c.key === "BTC");
      const mstrCard = CARDS.find(c => c.key === "MSTR");
      if (btcCard && btcCard.extra) {{
        const ex = btcCard.extra;
        document.getElementById("halvTag").textContent = ex.in_buy_zone ? "处于买点区" : "周期监控中";
        document.getElementById("radarMini").innerHTML = `
          <div>• 距离下次减半: <b style="color:var(--cyan);">-${{ex.du}} 天</b> (上次 +${{ex.ds}} 天)</div>
          <div>• BTC 波动率比: <b>${{ex.hv_ratio}}</b> (触发冷却需 &lt; 1.0，或 H-490d 兜底)</div>
          <div>• MSTR mNAV脉冲: <b>${{mstrCard && mstrCard.extra ? mstrCard.extra.mstr_btc_r60 : "-"}}x</b> (逃顶阈值 ≥ 1.70x)</div>
          <div>• 通知链路: <span style="color:var(--green);">Telegram + Pushover 强提醒已就绪</span></div>
        `;
      }}

      // Status Table
      document.getElementById("statusTableBody").innerHTML = CARDS.map(c => `
        <tr class="row-click" onclick="openPage('${{c.file}}')">
          <td><span class="nav-dot" style="display:inline-block;background:${{c.accent}};margin-right:8px;"></span><b>${{c.title}}</b></td>
          <td>${{c.last_date}}</td>
          <td class="num">$${{Number(c.price).toLocaleString("en-US", {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</td>
          <td>${{c.position}}</td>
          <td><span class="pill ${{pillClass(c.tone)}}">${{c.badge}}</span></td>
          <td class="num" style="color:var(--green);font-weight:600;">$${{Number(c.final).toLocaleString("en-US", {{maximumFractionDigits:0}})}}</td>
          <td class="num">${{c.cagr.toFixed(1)}}%</td>
          <td class="num" style="color:var(--red);">${{c.max_dd.toFixed(1)}}%</td>
          <td style="color:var(--muted);max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${{c.reason}}">${{c.reason}}</td>
        </tr>
      `).join("");
    }}

    document.getElementById("navMain").querySelector(".nav-item").addEventListener("click", () => selectNav("OVERVIEW"));
    document.getElementById("navSearch").addEventListener("input", (e) => renderSidebar(e.target.value));
    renderSidebar();
    renderOverview();
  </script>
</body>
</html>
"""


def main() -> None:
    data = collect_portal_data()
    html = render_dashboard(data)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(data['cards'])} strategy cards)")


if __name__ == "__main__":
    main()
