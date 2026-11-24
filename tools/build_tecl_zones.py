"""Build a dedicated TECL monthly LOG chart with SOXL-driven cross-asset strategies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import (
    DATA_DIR,
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
from tools.build_tqqq_zones import simulate_tqqq_from_soxl_signal

TECL_PATH = DATA_DIR / "TECL.csv"
SOXL_PATH = DATA_DIR / "SOXL.csv"
SMH_PATH = DATA_DIR / "SMH.csv"
OUTPUT_HTML = ROOT / "tecl_monthly_zones.html"


def main() -> None:
    df_tecl = load_daily_adjusted(TECL_PATH)
    df_soxl = load_daily_adjusted(SOXL_PATH)
    df_smh = load_daily_adjusted(SMH_PATH)

    latest_date = str(min(df_tecl["Date"].max(), df_soxl["Date"].max()).date())
    monthly, weekly_sma = build_monthly_and_weekly(df_tecl)
    c7_entries = detect_c7_vol_entries(df_tecl)
    feats_tecl = build_causal_features(df_tecl, df_smh)
    feats_soxl = build_causal_features(df_soxl, df_smh)

    soxl_h5 = simulate_h5_adapt_hv(feats_soxl, end_date=latest_date)
    soxl_c2 = simulate_tp3h(
        feats_soxl,
        use_f1_filter=True,
        end_date=latest_date,
        entry_col_override="C2_Deep85HL",
        mode_tag_override="BOT_C2",
        strategy_id="C2_F1_or_MFishLin_TP3h",
        tp_split_lottery_pct=0.10,
    )
    soxl_f0 = simulate_tp3h(feats_soxl, use_f1_filter=False, end_date=latest_date)

    s_h5_tecl = simulate_tqqq_from_soxl_signal(
        df_tecl,
        soxl_h5,
        strategy_id="SOXL_H5_to_TECL",
        name="跨标的旗舰: SOXL双引擎(H5)信号 → 买卖TECL (12年$1,090万, CAGR 80.3%, MaxDD -43.8%)",
        short_label="SOXL(H5)→TECL",
        rule_desc="由 SOXL H5 双引擎(E2大底晋升+SMH 200SMA+自适应HV+ALT恐慌)发出买卖信号，次日开盘全仓买卖 TECL",
        end_date=latest_date,
    )
    s_c2_tecl = simulate_tqqq_from_soxl_signal(
        df_tecl,
        soxl_c2,
        strategy_id="SOXL_C2_to_TECL",
        name="跨标的大周期: SOXL(C2+Deep85_HL+TP3h)信号 → 买卖TECL (5战全胜, 12年$784万, CAGR 75.3%)",
        short_label="SOXL(C2)→TECL",
        rule_desc="由 SOXL C2+Deep85_HL+TP3h 发出大底与逃顶信号，次日开盘全仓买卖 TECL（5战全胜，Calmar 1.62）",
        end_date=latest_date,
    )
    s_f0_tecl = simulate_tqqq_from_soxl_signal(
        df_tecl,
        soxl_f0,
        strategy_id="SOXL_F0_to_TECL",
        name="跨标的通用大周期: SOXL(F0+TP3h)信号 → 买卖TECL (6笔, 12年$309万)",
        short_label="SOXL(F0)→TECL",
        rule_desc="由 SOXL F0+TP3h 发出信号，次日开盘全仓买卖 TECL",
        end_date=latest_date,
    )
    s_d1_tecl = simulate_d1_x1a_wtsl35(feats_tecl, use_split_50_50=False, end_date=latest_date)

    strategies = {
        "SOXL_H5_to_TECL": s_h5_tecl,
        "SOXL_C2_to_TECL": s_c2_tecl,
        "SOXL_F0_to_TECL": s_f0_tecl,
        "D1_X1a_WTSL35": s_d1_tecl,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    live_market = build_live_market_snapshot(feats_tecl)
    html = render_html(monthly, weekly_sma, c7_entries, strategies, live_market=live_market)

    tecl_options = """<select id="strategySelect" class="strategy-select">
        <option value="SOXL_H5_to_TECL" selected>跨标的旗舰: SOXL双引擎(H5)信号 → 买TECL (12年$1,090万, MaxDD -43.8%, Calmar 1.83)</option>
        <option value="SOXL_C2_to_TECL">跨标的大周期: SOXL(C2+Deep85_HL+TP3h) → 买TECL (5战全胜, 12年$784万, CAGR 75.3%)</option>
        <option value="SOXL_F0_to_TECL">跨标的通用大周期: SOXL(F0+TP3h) → 买TECL (6笔, 12年$309万)</option>
        <option value="D1_X1a_WTSL35">TECL 原生极简大周期: D1_X1a_WTSL35</option>
        <option value="NONE">不显示策略标注</option>
      </select>"""

    html = re.sub(
        r'<select id="strategySelect" class="strategy-select">.*?</select>',
        tecl_options,
        html,
        count=1,
        flags=re.S,
    )
    html = html.replace("SOXL 月线顶底区域", "TECL 月线顶底区域 (SOXL 信号驱动)")
    html = html.replace("SOXL 月线", "TECL 月线")
    html = html.replace("SOXL 价格", "TECL 价格")
    html = html.replace("<span>SOXL <b>", "<span>TECL <b>")

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")


if __name__ == "__main__":
    main()
