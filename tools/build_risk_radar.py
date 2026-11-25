"""Build the Risk Radar page (risk_radar.html) combining options/credit signals and live news RSS."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_HTML = ROOT / "risk_radar.html"

RSS_FEEDS = [
    ("CNBC Top", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC Finance", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]

RISK_RULES = [
    ("关税/出口", 3, [
        "tariff", "tariffs", "trade war", "export ban", "export control", "export curb",
        "sanction", "sanctions", "customs duty", "retaliat", "trade restriction",
    ]),
    ("战争/地缘", 3, [
        "war", "missile", "airstrike", "strike", "invasion", "military", "troops",
        "iran", "israel", "ukraine", "russia", "taiwan", "strait", "nato", "nuclear", "conflict",
    ]),
    ("半导体/AI", 2, [
        "nvidia", "nvda", "tsmc", "semiconductor", "semiconductors", "chip", "chips",
        "soxl", "amd", "broadcom", "asml", "hbm", "ai spending", "data center",
    ]),
    ("央行/宏观", 2, [
        "fed", "federal reserve", "powell", "rate hike", "interest rate", "inflation",
        "cpi", "pce", "treasury yield", "bond yield", "recession", "stagflation", "fomc",
    ]),
    ("财报/信用", 2, [
        "earnings", "guidance", "downgrade", "default", "bankruptcy", "credit spread",
        "layoff", "layoffs", "warning", "miss", "death cross", "selloff", "sell-off",
    ]),
]


def load_close(ticker: str) -> pd.Series:
    p = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    if "Adj Close" in df.columns and ticker in {"QQQ", "SMH", "SOXL", "HYG", "LQD"}:
        s = pd.to_numeric(df["Adj Close"], errors="coerce")
        if s.notna().sum() > 0:
            return s
    return pd.to_numeric(df["Close"], errors="coerce")


def is_earnings_season(dt: pd.Timestamp) -> bool:
    md = dt.month * 100 + dt.day
    return (
        (115 <= md <= 229)
        or (415 <= md <= 531)
        or (715 <= md <= 831)
        or (1015 <= md <= 1130)
    )


def fetch_news_radar() -> dict:
    items: list[dict] = []
    seen_titles: set[str] = set()
    ok_sources: list[str] = []
    err_sources: list[str] = []

    for source, url in RSS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            ok_sources.append(source)
            for node in root.findall(".//item")[:25]:
                title = (node.findtext("title") or "").strip()
                link = (node.findtext("link") or "").strip()
                desc = (node.findtext("description") or "").strip()
                pub_raw = (node.findtext("pubDate") or "").strip()
                if not title:
                    continue
                norm_key = re.sub(r"\W+", "", title.lower())[:64]
                if norm_key in seen_titles:
                    continue
                seen_titles.add(norm_key)

                text_blob = f"{title} {desc}".lower()
                cats: list[str] = []
                score = 0
                for cat_name, weight, kws in RISK_RULES:
                    if any(re.search(rf"\b{re.escape(kw)}", text_blob) for kw in kws):
                        cats.append(cat_name)
                        score += weight

                pub_ts = 0.0
                pub_str = "-"
                if pub_raw:
                    try:
                        dt = parsedate_to_datetime(pub_raw)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        pub_ts = dt.timestamp()
                        pub_str = dt.astimezone().strftime("%m-%d %H:%M")
                    except Exception:
                        pub_str = pub_raw[:16]

                items.append({
                    "source": source,
                    "title": html_lib.unescape(title),
                    "link": link,
                    "pub_str": pub_str,
                    "pub_ts": pub_ts,
                    "cats": cats if cats else ["常规财经"],
                    "score": score,
                })
        except Exception as exc:
            err_sources.append(f"{source}: {exc}")

    # Sort high-risk first, then newest
    items.sort(key=lambda x: (x["score"], x["pub_ts"]), reverse=True)
    risk_items = [x for x in items if x["score"] > 0]
    total_risk_points = sum(x["score"] for x in risk_items[:20])
    news_heat = min(100, int(round(total_risk_points * 1.8)))
    geo_tariff_hits = sum(
        1 for x in risk_items if ("关税/出口" in x["cats"] or "战争/地缘" in x["cats"])
    )

    return {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ok_sources": ok_sources,
        "err_sources": err_sources,
        "total_items": len(items),
        "risk_items_count": len(risk_items),
        "geo_tariff_hits": geo_tariff_hits,
        "news_heat": news_heat,
        "items": (risk_items + [x for x in items if x["score"] == 0])[:35],
    }


def compute_risk_data() -> dict:
    soxl = load_close("SOXL")
    smh = load_close("SMH")
    qqq = load_close("QQQ")
    vix = load_close("^VIX")
    vix3m = load_close("^VIX3M")
    vxn = load_close("^VXN")
    move = load_close("^MOVE")
    hyg = load_close("HYG")
    lqd = load_close("LQD")

    df = pd.DataFrame({
        "SOXL": soxl,
        "SMH": smh,
        "QQQ": qqq,
        "VIX": vix,
        "VIX3M": vix3m,
        "VXN": vxn,
        "MOVE": move,
        "HYG": hyg,
        "LQD": lqd,
    }).sort_index()
    df = df.loc[soxl.index].ffill()

    df["smh_sma200"] = df["SMH"].rolling(200).mean()
    df["qqq_hv20"] = np.log(df["QQQ"] / df["QQQ"].shift(1)).rolling(20).std() * np.sqrt(252) * 100.0
    df["soxl_hv20"] = np.log(df["SOXL"] / df["SOXL"].shift(1)).rolling(20).std() * np.sqrt(252) * 100.0
    df["vix_ratio"] = df["VIX"] / df["VIX3M"]
    df["vxn_sma60"] = df["VXN"].rolling(60).mean()
    df["iv_hv_spread"] = df["VXN"] - df["qqq_hv20"]
    df["move_sma60"] = df["MOVE"].rolling(60).mean()
    df["hyg_lqd"] = df["HYG"] / df["LQD"]
    df["hyg_lqd_sma50"] = df["hyg_lqd"].rolling(50).mean()

    df["sig_vix_inv"] = df["vix_ratio"] >= 1.00
    df["sig_vix_95"] = df["vix_ratio"] >= 0.95
    df["sig_vxn_spike"] = (df["VXN"] > df["vxn_sma60"] * 1.20) & (df["VXN"] >= 24.0)
    df["sig_iv_hv"] = df["iv_hv_spread"] >= 8.0
    df["sig_move_alert"] = (df["MOVE"] >= 95.0) & (df["MOVE"] > df["move_sma60"] * 1.15)
    df["sig_credit_weak"] = df["hyg_lqd"] < df["hyg_lqd_sma50"] * 0.99

    df["warn_score"] = (
        df["sig_vix_inv"].astype(int)
        + df["sig_vxn_spike"].astype(int)
        + df["sig_move_alert"].astype(int)
        + df["sig_credit_weak"].astype(int)
    )
    df["high_alt"] = df["SMH"] > df["smh_sma200"] * 1.05
    df["high_alt_warn"] = df["high_alt"] & ((df["vix_ratio"] >= 1.00) | (df["sig_vxn_spike"] & df["sig_credit_weak"]))

    last_dt = df.index[-1]
    last = df.iloc[-1]
    earn_season = is_earnings_season(last_dt)
    smh_above_200_pct = (last["SMH"] / last["smh_sma200"] - 1.0) * 100.0

    # Traffic light logic
    ws = int(last["warn_score"])
    vr = float(last["vix_ratio"])
    if ws >= 3 or (ws >= 2 and bool(last["high_alt"])):
        light = "RED"
        light_label = "红灯高危"
        light_tone = "sell"
        action_hint = "机构期权与债市同步恐慌且处于高位区，建议立刻降杠杆（SOXL→SMH/现金）或买短期 Put 对冲"
    elif ws >= 1 or vr >= 0.95 or bool(last["sig_iv_hv"]):
        light = "YELLOW"
        light_label = "黄灯警戒"
        light_tone = "hold"
        action_hint = "出现早期对冲/避险苗头，暂勿追高加杠杆；若处于财报季或高位区建议减仓 30%–50% 观察"
    else:
        light = "GREEN"
        light_label = "绿灯正常"
        light_tone = "buy"
        action_hint = "期权期限结构与信用利差平稳，按主策略大周期规则正常执行即可"

    news = fetch_news_radar()
    heat = news["news_heat"]
    geo_hits = news["geo_tariff_hits"]

    # News x Options Cross-Validation
    if (heat >= 40 or geo_hits >= 3) and light == "GREEN":
        cross_title = "新闻热闹但期权未动（口水战/暂未定价恐慌）"
        cross_desc = f"当前地缘/关税/宏观新闻热度 {heat}/100（涉关税或地缘 {geo_hits} 条），但 VIX/VIX3M={vr:.3f} 且信用利差稳定，华尔街真金白银尚未大举避险，暂按绿灯观望。"
    elif (heat >= 40 or geo_hits >= 3) and light in {"YELLOW", "RED"}:
        cross_title = "新闻升级 × 期权/债市共振（高度警惕）"
        cross_desc = f"政策/地缘新闻热度 {heat}/100 与期权/信用指标同步亮灯（warn_score={ws}/4, VIX/VIX3M={vr:.3f}），属于真金白银避险确认，建议严格控杠杆或对冲。"
    elif heat < 40 and light in {"YELLOW", "RED"}:
        cross_title = "新闻平静但期权/债市先动（隐蔽流动性异动）"
        cross_desc = f"新闻面尚温和（热度 {heat}/100），但期权/信用层面已出现异动（warn_score={ws}/4, VIX/VIX3M={vr:.3f}），注意防慢发酵宏观风险。"
    else:
        cross_title = "新闻与期权双静默（常态运行）"
        cross_desc = f"新闻风险热度 {heat}/100，VIX/VIX3M={vr:.3f}，市场无系统性抢买保险迹象。"

    # Chart series from 2018-01-01 onward
    plot_df = df.loc["2018-01-01":].copy()
    dates = [d.strftime("%Y-%m-%d") for d in plot_df.index]

    hi_pts = plot_df[plot_df["high_alt_warn"]]
    inv_pts = plot_df[plot_df["sig_vix_inv"]]

    return {
        "live": {
            "last_date": last_dt.strftime("%Y-%m-%d"),
            "soxl_close": round(float(last["SOXL"]), 2),
            "smh_close": round(float(last["SMH"]), 2),
            "smh_vs_200_pct": round(float(smh_above_200_pct), 1),
            "vix": round(float(last["VIX"]), 2),
            "vix3m": round(float(last["VIX3M"]), 2),
            "vix_ratio": round(vr, 3),
            "vxn": round(float(last["VXN"]), 2),
            "qqq_hv20": round(float(last["qqq_hv20"]), 2),
            "iv_hv_spread": round(float(last["iv_hv_spread"]), 2),
            "move": round(float(last["MOVE"]), 2),
            "hyg_lqd": round(float(last["hyg_lqd"]), 4),
            "hyg_lqd_dev_pct": round(float((last["hyg_lqd"] / last["hyg_lqd_sma50"] - 1.0) * 100.0), 2),
            "warn_score": ws,
            "is_earnings_season": earn_season,
            "light": light,
            "light_label": light_label,
            "light_tone": light_tone,
            "action_hint": action_hint,
            "news_heat": heat,
            "geo_tariff_hits": geo_hits,
            "cross_title": cross_title,
            "cross_desc": cross_desc,
            "signals": {
                "vix_inv": bool(last["sig_vix_inv"]),
                "vxn_spike": bool(last["sig_vxn_spike"]),
                "move_alert": bool(last["sig_move_alert"]),
                "credit_weak": bool(last["sig_credit_weak"]),
                "iv_hv_high": bool(last["sig_iv_hv"]),
            },
        },
        "news": news,
        "series": {
            "dates": dates,
            "soxl": [round(float(x), 2) for x in plot_df["SOXL"]],
            "vix_ratio": [round(float(x), 3) for x in plot_df["vix_ratio"]],
            "vxn": [round(float(x), 2) for x in plot_df["VXN"]],
            "qqq_hv20": [round(float(x), 2) for x in plot_df["qqq_hv20"]],
            "move": [round(float(x), 2) for x in plot_df["MOVE"]],
            "hyg_lqd": [round(float(x), 4) for x in plot_df["hyg_lqd"]],
            "hyg_lqd_sma50": [round(float(x), 4) if pd.notna(x) else None for x in plot_df["hyg_lqd_sma50"]],
            "hi_dates": [d.strftime("%Y-%m-%d") for d in hi_pts.index],
            "hi_soxl": [round(float(x), 2) for x in hi_pts["SOXL"]],
            "inv_dates": [d.strftime("%Y-%m-%d") for d in inv_pts.index],
            "inv_soxl": [round(float(x), 2) for x in inv_pts["SOXL"]],
        },
    }


def render_html(payload: dict) -> str:
    live_json = json.dumps(payload["live"], separators=(",", ":"), ensure_ascii=False)
    news_json = json.dumps(payload["news"], separators=(",", ":"), ensure_ascii=False)
    series_json = json.dumps(payload["series"], separators=(",", ":"), ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ApexStock · 宏观期权与新闻风险雷达</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #081028;
      --surface: #0B1739;
      --surface-2: #0F1E47;
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
    .banner h1 {{ font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
    .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .pill {{
      display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
      border-radius: 999px; font-size: 12px; font-weight: 600;
    }}
    .pill-buy {{ background: rgba(20,202,116,0.16); color: var(--green); border: 1px solid rgba(20,202,116,0.35); }}
    .pill-hold {{ background: rgba(253,181,42,0.16); color: var(--amber); border: 1px solid rgba(253,181,42,0.35); }}
    .pill-sell {{ background: rgba(255,90,121,0.16); color: var(--red); border: 1px solid rgba(255,90,121,0.35); }}
    .pill-neutral {{ background: rgba(126,137,172,0.14); color: var(--text-sec); border: 1px solid rgba(126,137,172,0.3); }}

    .cross-box {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px;
      display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; align-items: center;
    }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .kpi {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
      display: flex; flex-direction: column; gap: 8px;
    }}
    .kpi-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
    .kpi-title {{ font-size: 13px; font-weight: 700; color: var(--text); }}
    .kpi-badge {{ font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; }}
    .gauge-wrap {{ display: flex; flex-direction: column; align-items: center; margin: 2px 0; }}
    .gauge-val {{ font-size: 22px; font-weight: 700; margin-top: -10px; font-variant-numeric: tabular-nums; }}
    .gauge-sub {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
    .kpi-guide {{
      background: rgba(8,16,40,0.55); border: 1px solid rgba(52,59,79,0.65); border-radius: 8px;
      padding: 8px 10px; font-size: 11px; line-height: 1.55; color: var(--text-sec);
    }}
    .kpi-guide b {{ color: var(--cyan); }}

    .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    }}
    .card-title {{
      font-size: 14px; font-weight: 600; margin-bottom: 10px;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .card-sub {{ font-size: 11px; color: var(--muted); font-weight: 400; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{
      text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
      padding: 9px 10px; border-bottom: 1px solid var(--border);
    }}
    td {{ padding: 9px 10px; border-bottom: 1px solid rgba(52,59,79,0.45); color: var(--text-sec); vertical-align: top; }}
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    a {{ color: var(--text); text-decoration: none; }}
    a:hover {{ color: var(--cyan); text-decoration: underline; }}
    .tag {{
      display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 11px; margin-right: 4px;
      background: rgba(0,194,255,0.14); color: var(--cyan); border: 1px solid rgba(0,194,255,0.28);
    }}
    .tag-hot {{ background: rgba(255,90,121,0.16); color: var(--red); border-color: rgba(255,90,121,0.35); }}

    @media (max-width: 1200px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .charts-grid, .cross-box {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 680px) {{
      .kpi-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="banner">
      <div>
        <h1>
          <span>宏观期权 & 新闻风险雷达</span>
          <span id="lightPill" class="pill"></span>
          <span id="earnPill" class="pill pill-neutral"></span>
        </h1>
        <div class="sub" id="metaSub"></div>
      </div>
      <div style="text-align:right;font-size:12px;color:var(--text-sec);">
        <div>4 项量化共振: <b id="warnScore" style="color:var(--cyan);"></b></div>
        <div style="color:var(--muted);" id="cycleCtx"></div>
      </div>
    </div>

    <div class="cross-box">
      <div>
        <div style="font-size:13px;font-weight:700;color:var(--cyan);margin-bottom:4px;" id="crossTitle"></div>
        <div style="font-size:12px;color:var(--text-sec);" id="crossDesc"></div>
      </div>
      <div style="border-left:1px solid var(--border);padding-left:16px;">
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">操作建议（仅作风控雷达，不直接全自动清仓）</div>
        <div style="font-size:12px;color:var(--text);" id="actionHint"></div>
      </div>
    </div>

    <div class="kpi-grid" id="kpiGrid"></div>

    <div class="charts-grid">
      <div class="card">
        <div class="card-title">
          <span>SOXL 对数走势 × 历史期权高位预警点 (2018+)</span>
          <span class="card-sub">紫色菱形 = 高位共振预警 · 红色小点 = VIX倒挂</span>
        </div>
        <div id="chartSoxl" style="height:310px;"></div>
      </div>
      <div class="card">
        <div class="card-title">
          <span>VIX / VIX3M 期限结构比值（≥ 1.00 为机构抢买近月保险）</span>
          <span class="card-sub">红线 1.00 倒挂阈值</span>
        </div>
        <div id="chartVixRatio" style="height:310px;"></div>
      </div>
      <div class="card">
        <div class="card-title">
          <span>^VXN 纳指隐含波动率 vs QQQ 20日已实现波动 (HV20)</span>
          <span class="card-sub">差值扩大 = 股价未跌保费先涨</span>
        </div>
        <div id="chartVxn" style="height:280px;"></div>
      </div>
      <div class="card">
        <div class="card-title">
          <span>^MOVE 美债波动率 & HYG/LQD 信用比值</span>
          <span class="card-sub">2022 加息 & 流动性抽水先知</span>
        </div>
        <div id="chartMacro" style="height:280px;"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>历史慢发酵黑天鹅实证对比（2018–2026）</span>
        <span class="card-sub">对比原策略刹车（SMH&lt;200SMA / SOXL HV20&gt;115%）</span>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>历史事件</th>
              <th>SOXL 见顶 → 见底</th>
              <th>期权/信用最早预警日</th>
              <th>对比见顶提前/滞后</th>
              <th>原策略刹车触发日</th>
              <th>实战启示</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>2020 疫情崩盘</b></td>
              <td>2020-02-19 ($21.02) → 03-20 ($4.12, -80.4%)</td>
              <td>HYG/LQD: 2020-01-24 ($19.74)<br>VIX倒挂: 2020-01-27 ($17.45)</td>
              <td style="color:var(--green);">提前 23–26 天</td>
              <td>HV115: 2020-02-24 ($15.67)<br>200SMA: 2020-03-09 ($9.56)</td>
              <td>疫情仍在早期发酵，债市与VIX期限已提前近1个月报警</td>
            </tr>
            <tr>
              <td><b>2022 加息熊市</b></td>
              <td>2021-12-27 ($70.79) → 06-16 ($13.16, -81.4%)</td>
              <td>VXN+信用: 2021-11-26 ($59.50)<br>VIX倒挂: 2021-12-01 ($61.24)</td>
              <td style="color:var(--green);">提前 26–31 天</td>
              <td>200SMA: 2022-01-25 ($38.98)<br>HV115: 2022-01-28 ($37.25)</td>
              <td>高估值杀跌前1个月，纳指期权与信用利差已同步亮灯</td>
            </tr>
            <tr>
              <td><b>2025 关税冲击</b></td>
              <td>2025-02-20 ($31.47) → 04-08 ($8.22, -73.9%)</td>
              <td>VIX/3M≥0.95: 2025-02-03 ($25.43)<br>VIX倒挂+MOVE: 2025-03-03 ($20.36)</td>
              <td style="color:var(--amber);">升级前 35 天</td>
              <td>200SMA & HV115: 2025-02-03 ($25.43)</td>
              <td>政策升级主跌段前，VIX倒挂与MOVE再次确认了机构避险</td>
            </tr>
            <tr>
              <td><b>2026-10 回撤</b></td>
              <td>2026-10-16 ($93.67) 触发高位预警</td>
              <td>高位共振: 2026-10-16 ($93.67)</td>
              <td style="color:var(--green);">20日回撤 -37.7%</td>
              <td>财报季窗口亮灯</td>
              <td>财报季+高位区亮灯后反弹明显偏弱，含金量最高</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>实时财经 & 地缘政策新闻雷达（RSS 抓取 + 关键词分级）</span>
        <span class="card-sub" id="newsMeta"></span>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th style="width:95px;">时间</th>
              <th style="width:110px;">来源</th>
              <th style="width:150px;">风险标签</th>
              <th style="width:60px;">权重</th>
              <th>新闻标题（点击看原文）</th>
            </tr>
          </thead>
          <tbody id="newsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const RISK_LIVE = {live_json};
    const NEWS = {news_json};
    const S = {series_json};

    // Header
    const lp = document.getElementById("lightPill");
    lp.textContent = RISK_LIVE.light_label;
    lp.className = "pill " + (RISK_LIVE.light_tone === "sell" ? "pill-sell" : RISK_LIVE.light_tone === "hold" ? "pill-hold" : "pill-buy");
    const ep = document.getElementById("earnPill");
    ep.textContent = RISK_LIVE.is_earnings_season ? "当前处于美股财报季 (警报权重↑)" : "非财报季 (注意区分情绪洗盘)";
    document.getElementById("metaSub").textContent = `市场数据截至 ${{RISK_LIVE.last_date}} · 新闻抓取 ${{NEWS.fetched_at}} · SOXL $${{RISK_LIVE.soxl_close}} · SMH $${{RISK_LIVE.smh_close}}`;
    document.getElementById("warnScore").textContent = `${{RISK_LIVE.warn_score}} / 4`;
    document.getElementById("cycleCtx").textContent = `SMH 相对 200日线: ${{RISK_LIVE.smh_vs_200_pct >= 0 ? "+" : ""}}${{RISK_LIVE.smh_vs_200_pct}}%`;
    document.getElementById("crossTitle").textContent = RISK_LIVE.cross_title;
    document.getElementById("crossDesc").textContent = RISK_LIVE.cross_desc;
    document.getElementById("actionHint").textContent = RISK_LIVE.action_hint;

    // Speedometer Gauge helper (SVG semi-circle with green/amber/red zones + needle)
    function polarPt(cx, cy, r, frac) {{
      const th = Math.PI * (1 - Math.max(0, Math.min(1, frac)));
      return [cx + r * Math.cos(th), cy - r * Math.sin(th)];
    }}
    function arcPath(cx, cy, r, f0, f1) {{
      const [x0, y0] = polarPt(cx, cy, r, f0);
      const [x1, y1] = polarPt(cx, cy, r, f1);
      return `M ${{x0.toFixed(1)}} ${{y0.toFixed(1)}} A ${{r}} ${{r}} 0 0 1 ${{x1.toFixed(1)}} ${{y1.toFixed(1)}}`;
    }}
    function buildGaugeSvg(val, min, max, zones, minLabel, maxLabel) {{
      const cx = 110, cy = 96, r = 74;
      const toFrac = (v) => Math.max(0, Math.min(1, (v - min) / (max - min)));
      const arcs = zones.map(z => {{
        const p = arcPath(cx, cy, r, toFrac(z.from), toFrac(z.to));
        return `<path d="${{p}}" fill="none" stroke="${{z.color}}" stroke-width="14" stroke-linecap="butt" opacity="0.88" />`;
      }}).join("");
      const fVal = toFrac(val);
      const [nx, ny] = polarPt(cx, cy, r - 10, fVal);
      const [tx, ty] = polarPt(cx, cy, r + 15, fVal);
      return `
        <svg viewBox="0 0 220 112" style="width:100%;max-width:240px;height:auto;overflow:visible;">
          <path d="${{arcPath(cx, cy, r, 0, 1)}}" fill="none" stroke="#1B284F" stroke-width="14" />
          ${{arcs}}
          <line x1="${{cx}}" y1="${{cy}}" x2="${{nx.toFixed(1)}}" y2="${{ny.toFixed(1)}}" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" />
          <circle cx="${{cx}}" cy="${{cy}}" r="5.5" fill="#00C2FF" stroke="#FFFFFF" stroke-width="1.5" />
          <text x="20" y="110" fill="#7E89AC" font-size="10" text-anchor="middle">${{minLabel}}</text>
          <text x="200" y="110" fill="#7E89AC" font-size="10" text-anchor="middle">${{maxLabel}}</text>
        </svg>
      `;
    }}

    const G = "#14CA74", Y = "#FDB52A", R = "#FF5A79";
    const gauges = [
      {{
        title: "VIX / VIX3M 期限结构",
        val: RISK_LIVE.vix_ratio,
        valText: RISK_LIVE.vix_ratio.toFixed(3),
        min: 0.75, max: 1.20, minLabel: "0.75 安全", maxLabel: "1.20 极度恐慌",
        zones: [{{ from: 0.75, to: 0.95, color: G }}, {{ from: 0.95, to: 1.00, color: Y }}, {{ from: 1.00, to: 1.20, color: R }}],
        status: RISK_LIVE.signals.vix_inv ? "红灯倒挂" : RISK_LIVE.vix_ratio >= 0.95 ? "黄灯警戒" : "绿灯正常",
        tone: RISK_LIVE.signals.vix_inv ? "sell" : RISK_LIVE.vix_ratio >= 0.95 ? "hold" : "buy",
        sub: `VIX ${{RISK_LIVE.vix}} / VIX3M ${{RISK_LIVE.vix3m}} (≥1.00 倒挂)`,
        role: "对比 1 个月与 3 个月期权价格，衡量机构是否抢买近期保护。",
        usage: "≥1.00 说明大资金不计成本避险；若叠加高位或财报季，果断降杠杆/买 Put。",
      }},
      {{
        title: "^VXN 纳指恐慌指数",
        val: RISK_LIVE.vxn,
        valText: RISK_LIVE.vxn.toFixed(2),
        min: 12, max: 45, minLabel: "12 平静", maxLabel: "45 恐慌",
        zones: [{{ from: 12, to: 24, color: G }}, {{ from: 24, to: 30, color: Y }}, {{ from: 30, to: 45, color: R }}],
        status: RISK_LIVE.signals.vxn_spike ? "脉冲警报" : RISK_LIVE.vxn >= 24 ? "偏高警戒" : "绿灯平静",
        tone: RISK_LIVE.signals.vxn_spike ? "sell" : RISK_LIVE.vxn >= 24 ? "hold" : "buy",
        sub: RISK_LIVE.signals.vxn_spike ? "突破60日均值 +20% 且 ≥24" : "正常区间 (<24 安全 / ≥30 高危)",
        role: "反映华尔街对纳指科技股未来 30 天波动的恐慌定价。",
        usage: "科技股杀估值前常先飙升；突破 60 日均线 +20% 且 ≥24 时停止加杠杆。",
      }},
      {{
        title: "VXN - QQQ_HV20 保费溢价",
        val: RISK_LIVE.iv_hv_spread,
        valText: (RISK_LIVE.iv_hv_spread >= 0 ? "+" : "") + RISK_LIVE.iv_hv_spread.toFixed(2),
        min: -5, max: 15, minLabel: "-5 便宜", maxLabel: "+15 暴涨",
        zones: [{{ from: -5, to: 4, color: G }}, {{ from: 4, to: 8, color: Y }}, {{ from: 8, to: 15, color: R }}],
        status: RISK_LIVE.signals.iv_hv_high ? "保费偏贵" : RISK_LIVE.iv_hv_spread >= 4 ? "略偏高" : "绿灯正常",
        tone: RISK_LIVE.signals.iv_hv_high ? "hold" : "buy",
        sub: `QQQ 实际波动 HV20=${{RISK_LIVE.qqq_hv20}}% (≥+8 预警)`,
        role: "对比“期权暗含恐慌”与“现货实际波动”，抓股价没跌保费先涨。",
        usage: "差值 ≥+8 说明机构暗中买保险；若此时现货还在高位横盘，警惕突发下跌。",
      }},
      {{
        title: "^MOVE 美债波动率",
        val: RISK_LIVE.move,
        valText: RISK_LIVE.move.toFixed(1),
        min: 50, max: 150, minLabel: "50 稳健", maxLabel: "150 动荡",
        zones: [{{ from: 50, to: 95, color: G }}, {{ from: 95, to: 115, color: Y }}, {{ from: 115, to: 150, color: R }}],
        status: RISK_LIVE.signals.move_alert ? "债市警报" : RISK_LIVE.move >= 95 ? "债市升温" : "绿灯稳健",
        tone: RISK_LIVE.signals.move_alert ? "sell" : RISK_LIVE.move >= 95 ? "hold" : "buy",
        sub: "美债利率波动 (≥95 且高于60日均线+15%报警)",
        role: "监测美联储加息/利率预期重定价，2022 熊市与财报季的核心先知。",
        usage: "债市抽水比股市先动；亮红灯时高估值科技与三倍杠杆 ETF 极易承压。",
      }},
      {{
        title: "HYG / LQD 信用偏离度",
        val: RISK_LIVE.hyg_lqd_dev_pct,
        valText: (RISK_LIVE.hyg_lqd_dev_pct >= 0 ? "+" : "") + RISK_LIVE.hyg_lqd_dev_pct.toFixed(2) + "%",
        min: -3.0, max: 2.0, minLabel: "-3% 恶化", maxLabel: "+2% 稳健",
        zones: [{{ from: -3.0, to: -1.0, color: R }}, {{ from: -1.0, to: -0.3, color: Y }}, {{ from: -0.3, to: 2.0, color: G }}],
        status: RISK_LIVE.signals.credit_weak ? "信用走弱" : RISK_LIVE.hyg_lqd_dev_pct < -0.3 ? "略偏弱" : "绿灯稳健",
        tone: RISK_LIVE.signals.credit_weak ? "hold" : "buy",
        sub: `当前比值 ${{RISK_LIVE.hyg_lqd.toFixed(4)}} (相对50日均线偏离)`,
        role: "垃圾债相对投资级债强弱，反映企业信用与系统性抽水风险。",
        usage: "跌破 50 日均线 -1% 说明大资金撤离风险资产（2020 疫情前曾提前 26 天报警）。",
      }},
      {{
        title: "新闻风险热度 (地缘/关税)",
        val: RISK_LIVE.news_heat,
        valText: `${{RISK_LIVE.news_heat}} / 100`,
        min: 0, max: 100, minLabel: "0 平静", maxLabel: "100 高危",
        zones: [{{ from: 0, to: 25, color: G }}, {{ from: 25, to: 50, color: Y }}, {{ from: 50, to: 100, color: R }}],
        status: RISK_LIVE.news_heat >= 50 ? "高危密集" : RISK_LIVE.news_heat >= 25 ? "升温关注" : "绿灯平静",
        tone: RISK_LIVE.news_heat >= 50 ? "sell" : RISK_LIVE.news_heat >= 25 ? "hold" : "buy",
        sub: `关税/战争命中 ${{RISK_LIVE.geo_tariff_hits}} 条 · 共 ${{NEWS.risk_items_count}} 条风险新闻`,
        role: "实时扫描主流外媒中关税、战争、芯片管制与央行冲击密度。",
        usage: "配合期权看：新闻热+期权绿灯=口水战；新闻热+期权红灯=真避险共振。",
      }},
    ];

    function badgeStyle(tone) {{
      if (tone === "sell") return "background:rgba(255,90,121,0.16);color:var(--red);border:1px solid rgba(255,90,121,0.35);";
      if (tone === "hold") return "background:rgba(253,181,42,0.16);color:var(--amber);border:1px solid rgba(253,181,42,0.35);";
      return "background:rgba(20,202,116,0.16);color:var(--green);border:1px solid rgba(20,202,116,0.35);";
    }}

    document.getElementById("kpiGrid").innerHTML = gauges.map(g => `
      <div class="kpi">
        <div class="kpi-head">
          <span class="kpi-title">${{g.title}}</span>
          <span class="kpi-badge" style="${{badgeStyle(g.tone)}}">${{g.status}}</span>
        </div>
        <div class="gauge-wrap">
          ${{buildGaugeSvg(g.val, g.min, g.max, g.zones, g.minLabel, g.maxLabel)}}
          <div class="gauge-val">${{g.valText}}</div>
          <div class="gauge-sub">${{g.sub}}</div>
        </div>
        <div class="kpi-guide">
          <div><b>作用：</b>${{g.role}}</div>
          <div><b>用法：</b>${{g.usage}}</div>
        </div>
      </div>
    `).join("");

    const baseLayout = {{
      paper_bgcolor: "#0B1739", plot_bgcolor: "#0B1739", font: {{ color: "#D1DBF9", size: 11 }},
      margin: {{ l: 50, r: 18, t: 10, b: 28 }},
      xaxis: {{ gridcolor: "rgba(52,59,79,0.45)" }},
      legend: {{ orientation: "h", y: 1.12, x: 0 }},
    }};

    Plotly.newPlot("chartSoxl", [
      {{ type: "scatter", mode: "lines", name: "SOXL", x: S.dates, y: S.soxl, line: {{ color: "#00C2FF", width: 1.8 }} }},
      {{ type: "scatter", mode: "markers", name: "VIX倒挂 (≥1.0)", x: S.inv_dates, y: S.inv_soxl, marker: {{ color: "#FF5A79", size: 4, opacity: 0.7 }} }},
      {{ type: "scatter", mode: "markers", name: "高位预警", x: S.hi_dates, y: S.hi_soxl, marker: {{ color: "#CB3CFF", size: 8, symbol: "diamond", line: {{ color: "#fff", width: 1 }} }} }},
    ], {{
      ...baseLayout,
      yaxis: {{ type: "log", gridcolor: "rgba(52,59,79,0.45)", tickprefix: "$" }},
    }}, {{ responsive: true }});

    Plotly.newPlot("chartVixRatio", [
      {{ type: "scatter", mode: "lines", name: "VIX / VIX3M", x: S.dates, y: S.vix_ratio, line: {{ color: "#FDB52A", width: 1.8 }} }},
      {{ type: "scatter", mode: "lines", name: "倒挂线 1.00", x: [S.dates[0], S.dates[S.dates.length - 1]], y: [1.0, 1.0], line: {{ color: "#FF5A79", width: 1.4, dash: "dash" }} }},
    ], {{
      ...baseLayout,
      yaxis: {{ gridcolor: "rgba(52,59,79,0.45)" }},
    }}, {{ responsive: true }});

    Plotly.newPlot("chartVxn", [
      {{ type: "scatter", mode: "lines", name: "^VXN (纳指IV)", x: S.dates, y: S.vxn, line: {{ color: "#CB3CFF", width: 1.8 }} }},
      {{ type: "scatter", mode: "lines", name: "QQQ HV20", x: S.dates, y: S.qqq_hv20, line: {{ color: "#14CA74", width: 1.5 }} }},
    ], {{
      ...baseLayout,
      yaxis: {{ gridcolor: "rgba(52,59,79,0.45)", ticksuffix: "%" }},
    }}, {{ responsive: true }});

    Plotly.newPlot("chartMacro", [
      {{ type: "scatter", mode: "lines", name: "HYG/LQD", x: S.dates, y: S.hyg_lqd, line: {{ color: "#00C2FF", width: 1.8 }}, yaxis: "y" }},
      {{ type: "scatter", mode: "lines", name: "HYG/LQD SMA50", x: S.dates, y: S.hyg_lqd_sma50, line: {{ color: "#7E89AC", width: 1.2, dash: "dot" }}, yaxis: "y" }},
      {{ type: "scatter", mode: "lines", name: "^MOVE (右轴)", x: S.dates, y: S.move, line: {{ color: "#FF5A79", width: 1.4 }}, yaxis: "y2" }},
    ], {{
      ...baseLayout,
      margin: {{ l: 50, r: 45, t: 10, b: 28 }},
      yaxis: {{ gridcolor: "rgba(52,59,79,0.45)" }},
      yaxis2: {{ overlaying: "y", side: "right", showgrid: false }},
    }}, {{ responsive: true }});

    // News Table
    const okCount = (NEWS.ok_sources || []).length;
    document.getElementById("newsMeta").textContent =
      okCount > 0 ? `成功源 ${{okCount}} 个 (${{NEWS.ok_sources.join(", ")}}) · 共 ${{NEWS.total_items}} 条` : "新闻源离线或超时（仅显示量化雷达）";

    const newsRows = (NEWS.items || []).map(it => `
      <tr>
        <td style="white-space:nowrap;">${{it.pub_str}}</td>
        <td>${{it.source}}</td>
        <td>${{(it.cats || []).map(c => `<span class="tag ${{it.score >= 3 ? "tag-hot" : ""}}">${{c}}</span>`).join("")}}</td>
        <td><b>${{it.score}}</b></td>
        <td><a href="${{it.link}}" target="_blank" rel="noopener">${{it.title}}</a></td>
      </tr>
    `).join("");
    document.getElementById("newsBody").innerHTML = newsRows || `<tr><td colspan="5" style="color:var(--muted);">暂无抓取到新闻条目</td></tr>`;
  </script>
</body>
</html>
"""


def _maybe_notify_tg(live: dict) -> None:
    if live["light"] == "GREEN":
        print(f"[RiskRadar TG] Light is GREEN ({live['last_date']}); skipping Telegram alert.")
        return
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[RiskRadar TG] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID; skipped.")
        return
    msg = (
        f"⚠️ ApexStock 风险雷达 [{live['light_label']}] ({live['last_date']})\n"
        f"• 4项共振: {live['warn_score']}/4 | VIX/VIX3M={live['vix_ratio']:.3f} | VXN={live['vxn']} | MOVE={live['move']}\n"
        f"• 交叉印证: {live['cross_title']}\n"
        f"• 建议: {live['action_hint']}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": msg}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"[RiskRadar TG] sent={res.get('ok')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notify-tg", action="store_true", help="Send Telegram alert if light is YELLOW/RED")
    args = parser.parse_args()

    payload = compute_risk_data()
    html_out = render_html(payload)
    OUTPUT_HTML.write_text(html_out, encoding="utf-8")
    live = payload["live"]
    news = payload["news"]
    print(
        f"Generated {OUTPUT_HTML} ({live['last_date']} | {live['light_label']} | "
        f"warn={live['warn_score']}/4 | VIX/3M={live['vix_ratio']} | news_heat={news['news_heat']}/100, "
        f"sources={len(news['ok_sources'])})"
    )
    if args.notify_tg:
        _maybe_notify_tg(live)


if __name__ == "__main__":
    main()
