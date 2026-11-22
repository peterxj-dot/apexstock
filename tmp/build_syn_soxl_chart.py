"""Generate standalone interactive HTML chart (/Users/jessemacree/apexstock/syn_soxl_zones.html) for simulated SYN_SOXL + SYN_SMH (seed=118)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tmp"))

from build_soxl_zones import (
    BT_END,
    BT_START,
    build_causal_features,
    build_monthly_and_weekly,
    detect_c7_vol_entries,
    load_daily_adjusted,
    render_html,
    simulate_d1_x1a_wtsl35,
    simulate_h5_adapt_hv,
    simulate_tp3h,
)
from simulate_synthetic_soxl import run_strategy1_from_dfs

SYN_SOXL_CSV = ROOT / "tmp" / "SYN_SOXL.csv"
SYN_SMH_CSV = ROOT / "tmp" / "SYN_SMH.csv"
OUTPUT_HTML = ROOT / "syn_soxl_zones.html"


def _fmt_money_cn(val: float) -> str:
    if val >= 100000.0:
        return f"${val / 10000.0:.1f}万"
    return f"${val:,.0f}"


def build_s1_overlay(df_soxl, df_smh) -> dict:
    """Run Strategy 1 on SYN_SOXL/SYN_SMH and adapt trade_logs to HTML_TEMPLATE schema."""
    d_soxl = df_soxl.set_index("Date")
    d_smh = df_smh.set_index("Date")
    s1 = run_strategy1_from_dfs(d_soxl, d_smh, BT_START, BT_END)

    trades = []
    for t in s1["trade_logs"]:
        is_open = t["exit_reason"] == "OPEN"
        ex_date = t["exit_date"].replace(" (Open)", "*") if is_open else t["exit_date"]
        ex_reason = "OPEN_END" if is_open else t["exit_reason"]
        trades.append(
            {
                "mode": t["mode"],
                "entry_date": t["entry_date"],
                "entry_m": t["entry_date"][:7],
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(float(t["entry_px"]), 2),
                "avg_entry_px": round(float(t["entry_px"]), 2),
                "exit_date": ex_date,
                "exit_m": ex_date[:7],
                "exit_px": round(float(t["exit_px"]), 2),
                "ret_pct": round(float(t["ret_pct"]), 1),
                "exit_reason": ex_reason,
                "is_open": is_open,
            }
        )

    return {
        "id": "S1_Paired_SMA200_hv95",
        "name": "策略 1 趋势+刹车经典版: Paired_SMA200_hv95",
        "short_label": "策略1 (SMA200_hv95)",
        "rule_desc": "SMH>200日SMA×1.01 & HV20<=105.8%开TREND | 跌破0.99×200SMA或HV>115%离场(刹车后等<=95%) | 周线W_Fish<-1.8恐慌ALT(+25%/-20%/40d)",
        "stats": {
            "final": round(float(s1["Final"]), 2),
            "cagr": round(float(s1["CAGR"]), 2),
            "max_dd": round(float(s1["MaxDD"]), 2),
            "calmar": round(float(s1["Calmar"]), 3),
            "trades": int(s1["Trades"]),
            "win_rate": round(float(s1["WinRate"]), 1),
        },
        "trades": trades,
    }


def main() -> None:
    df_soxl = load_daily_adjusted(SYN_SOXL_CSV)
    df_smh = load_daily_adjusted(SYN_SMH_CSV)

    monthly, weekly_sma = build_monthly_and_weekly(df_soxl)
    c7_entries = detect_c7_vol_entries(df_soxl)
    feats = build_causal_features(df_soxl, df_smh)

    h5 = simulate_h5_adapt_hv(feats)
    s1 = build_s1_overlay(df_soxl, df_smh)
    f1_tp3h = simulate_tp3h(feats, use_f1_filter=True)
    f0_tp3h = simulate_tp3h(feats, use_f1_filter=False)
    d1_full = simulate_d1_x1a_wtsl35(feats, use_split_50_50=False)
    d1_split = simulate_d1_x1a_wtsl35(feats, use_split_50_50=True)

    # Verify numerical parity with simulate_synthetic_soxl.py v2 (seed=118)
    assert abs(h5["stats"]["final"] - 1969201.52) < 1.0, h5["stats"]
    assert abs(s1["stats"]["final"] - 1437053.87) < 1.0, s1["stats"]
    assert abs(f1_tp3h["stats"]["final"] - 38789.46) < 1.0, f1_tp3h["stats"]

    strategies = {
        "H5_E2_Promote_S1_AdaptHV": h5,
        "S1_Paired_SMA200_hv95": s1,
        "F1_TP3h": f1_tp3h,
        "F0_TP3h": f0_tp3h,
        "D1_X1a_WTSL35": d1_full,
        "D1_X1a_WTSL35_Split50": d1_split,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | "
            f"Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    html = render_html(monthly, weekly_sma, c7_entries, strategies)

    # Customize visible labels cleanly for the synthetic stock
    html = html.replace(
        "<title>SOXL 月线 LOG — 顶部与底部区域及最优策略标注</title>",
        "<title>SYN_SOXL 虚拟行情 (seed=118) — 月线顶底与策略标注对比</title>",
    )
    html = html.replace(
        '<span class="title-badge">SOXL 月线顶底区域</span>',
        '<span class="title-badge">SYN_SOXL 虚拟行情 (seed=118)</span>',
    )
    html = html.replace('name: "SOXL 月线"', 'name: "SYN_SOXL 虚拟月线"')
    html = html.replace(
        'title: isLog ? "SOXL 价格 (LOG)" : "SOXL 价格"',
        'title: isLog ? "SYN_SOXL 虚拟价格 (LOG)" : "SYN_SOXL 虚拟价格"',
    )

    def _opt(key: str, label: str, selected: bool = False) -> str:
        st = strategies[key]["stats"]
        sel = " selected" if selected else ""
        return (
            f'        <option value="{key}"{sel}>{label} '
            f'({st["trades"]}笔, 胜率{st["win_rate"]:.1f}%, 12年{_fmt_money_cn(st["final"])}, 回撤{st["max_dd"]:.1f}%)</option>'
        )

    new_select = "\n".join(
        [
            '      <select id="strategySelect" class="strategy-select">',
            _opt("H5_E2_Promote_S1_AdaptHV", "策略4 双引擎旗舰: H5_E2_Promote_S1_AdaptHV", selected=True),
            _opt("S1_Paired_SMA200_hv95", "策略1 经典趋势+刹车: S1_Paired_SMA200_hv95"),
            _opt("F1_TP3h", "F1+TP3h (纯大底+主动止盈)"),
            _opt("F0_TP3h", "F0+TP3h (通用大底+主动止盈)"),
            _opt("D1_X1a_WTSL35", "策略3 极简大周期: D1_X1a_WTSL35"),
            _opt("D1_X1a_WTSL35_Split50", "策略3 分批建仓版: D1_X1a_WTSL35_Split50"),
            f'        <option value="C7_Vol">仅大底入口: C7_Vol ({len(c7_entries)}次)</option>',
            '        <option value="NONE">不显示策略标注</option>',
            "      </select>",
        ]
    )
    html, n_sub = re.subn(
        r'      <select id="strategySelect" class="strategy-select">.*?</select>',
        new_select,
        html,
        count=1,
        flags=re.DOTALL,
    )
    assert n_sub == 1, "Failed to replace strategySelect block"

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars, {len(c7_entries)} C7_Vol entries)")


if __name__ == "__main__":
    main()
