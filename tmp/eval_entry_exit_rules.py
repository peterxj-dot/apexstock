"""Evaluate individual entry and exit rules on SOXL (2015-2025)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import fisher_transform, sma
from backtest_soxl import load_soxl_daily


def main() -> None:
    daily = load_soxl_daily()
    start_date, end_date = "2015-01-01", "2025-12-31"

    w = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Sig"] = wf["fisher_signal"]
    w["d_W_Fisher"] = w["W_Fisher"].diff()
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    w["W_SMA30"] = sma(w["Close"], 30)

    m = daily.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    mf = fisher_transform(m["High"], m["Low"], period=10)
    m["M_Fisher"] = mf["fisher"]
    m["M_Vol_R"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean()

    w_dates = w.index
    m_dates = m.index
    bt_daily = daily.loc[start_date:end_date]
    dates = bt_daily.index

    # --- 1. Causal Strategy B Attribution ---
    in_pos = False
    pending_action = None
    pending_rule = None
    entry_price = 0.0
    entry_date = None
    entry_rule = ""
    trades = []
    last_w_idx = -1
    fee = 0.0005

    # Also record standalone weekly triggers in 2015-2025
    standalone_buys = []
    standalone_sells = []

    for i, dt in enumerate(dates):
        o = float(bt_daily["Open"].iloc[i])
        c = float(bt_daily["Close"].iloc[i])

        if pending_action == "BUY" and not in_pos:
            in_pos = True
            entry_price = o
            entry_date = dt
            entry_rule = pending_rule
        elif pending_action == "SELL" and in_pos:
            ret_pct = (o * (1.0 - fee) / (entry_price * (1.0 + fee)) - 1.0) * 100.0
            # Forward return 20d (~4w) and 60d (~12w) after exit
            fwd4w = (float(bt_daily["Close"].iloc[min(i + 20, len(dates) - 1)]) / o - 1.0) * 100.0
            fwd12w = (float(bt_daily["Close"].iloc[min(i + 60, len(dates) - 1)]) / o - 1.0) * 100.0
            trades.append(
                {
                    "entry_rule": entry_rule,
                    "exit_rule": pending_rule,
                    "ret_pct": ret_pct,
                    "exit_fwd4w%": fwd4w,
                    "exit_fwd12w%": fwd12w,
                }
            )
            in_pos = False
        pending_action = None
        pending_rule = None

        w_pos = int(w_dates.searchsorted(dt, side="right")) - 1
        if w_pos > last_w_idx and w_pos >= 1:
            last_w_idx = w_pos
            rw = w.iloc[w_pos]
            m_pos = int(m_dates.searchsorted(w_dates[w_pos], side="right")) - 1
            mf_val = float(m["M_Fisher"].iloc[m_pos]) if m_pos >= 0 and pd.notna(m["M_Fisher"].iloc[m_pos]) else 0.0
            mvr_val = float(m["M_Vol_R"].iloc[m_pos]) if m_pos >= 0 and pd.notna(m["M_Vol_R"].iloc[m_pos]) else 1.0

            wf_v = rw["W_Fisher"]
            dwf_v = rw["d_W_Fisher"]
            wvr_v = rw["W_Vol_R"]
            s30 = rw["W_SMA30"]

            if pd.notna(wf_v) and pd.notna(dwf_v) and pd.notna(s30):
                b1 = wf_v < -1.5 and dwf_v > 0
                b2 = wf_v < -1.5 and wvr_v > 1.3
                b3 = wf_v < -1.2 and mvr_val > 1.4
                b4 = rw["Close"] > s30 and wf_v > rw["W_Fisher_Sig"]

                s1 = mf_val > 2.0 and dwf_v < 0 and rw["Close"] < s30 * 1.05
                s2 = mf_val > 2.2 and dwf_v < -0.3
                s3 = rw["Close"] < s30 and wf_v < rw["W_Fisher_Sig"] and wf_v < 0.5

                # Standalone forward returns from next trading day's Open (i+1)
                if i + 1 < len(dates):
                    next_o = float(bt_daily["Open"].iloc[i + 1])
                    r4 = (float(bt_daily["Close"].iloc[min(i + 21, len(dates) - 1)]) / next_o - 1.0) * 100.0
                    r12 = (float(bt_daily["Close"].iloc[min(i + 61, len(dates) - 1)]) / next_o - 1.0) * 100.0
                    r26 = (float(bt_daily["Close"].iloc[min(i + 126, len(dates) - 1)]) / next_o - 1.0) * 100.0
                    for code, flag in [
                        ("B1: W_Fisher<-1.5 & dW>0", b1),
                        ("B2: W_Fisher<-1.5 & W_Vol>1.3", b2),
                        ("B3: W_Fisher<-1.2 & M_Vol>1.4", b3),
                        ("B4: Close>30W & W_Fisher金叉", b4),
                    ]:
                        if flag:
                            standalone_buys.append({"Rule": code, "r4w": r4, "r12w": r12, "r26w": r26})
                    for code, flag in [
                        ("S1: M_Fisher>2.0 & dW<0 & Close<1.05*30W", s1),
                        ("S2: M_Fisher>2.2 & dW<-0.3", s2),
                        ("S3: Close<30W & W_Fisher死叉(<0.5)", s3),
                    ]:
                        if flag:
                            standalone_sells.append({"Rule": code, "r4w": r4, "r12w": r12})

                buy_hits = [k for k, f in [("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4)] if f]
                sell_hits = [k for k, f in [("S1", s1), ("S2", s2), ("S3", s3)] if f]

                if not in_pos and buy_hits:
                    pending_action = "BUY"
                    pending_rule = "+".join(buy_hits)
                elif in_pos and sell_hits:
                    pending_action = "SELL"
                    pending_rule = "+".join(sell_hits)

    if in_pos:
        last_c = float(bt_daily["Close"].iloc[-1])
        ret_pct = (last_c / (entry_price * (1.0 + fee)) - 1.0) * 100.0
        trades.append(
            {
                "entry_rule": entry_rule,
                "exit_rule": "OPEN",
                "ret_pct": ret_pct,
                "exit_fwd4w%": np.nan,
                "exit_fwd12w%": np.nan,
            }
        )

    tdf = pd.DataFrame(trades)
    print("=== 1. Strategy B 实盘序列中：各买入入口组合表现 ===")
    e_stats = (
        tdf.groupby("entry_rule")["ret_pct"]
        .agg(
            Trades="count",
            WinRate_pct=lambda s: round((s > 0).mean() * 100, 1),
            AvgRet_pct=lambda s: round(s.mean(), 1),
            MedRet_pct=lambda s: round(s.median(), 1),
            MinRet_pct=lambda s: round(s.min(), 1),
            MaxRet_pct=lambda s: round(s.max(), 1),
        )
        .reset_index()
    )
    print(e_stats.to_string(index=False))

    print("\n=== 2. Strategy B 实盘序列中：各卖出出口组合表现 ===")
    x_stats = (
        tdf[tdf["exit_rule"] != "OPEN"]
        .groupby("exit_rule")
        .agg(
            Exits=("ret_pct", "count"),
            TradeWinRate_pct=("ret_pct", lambda s: round((s > 0).mean() * 100, 1)),
            AvgTradeRet_pct=("ret_pct", lambda s: round(s.mean(), 1)),
            PostExit4w_Avg_pct=("exit_fwd4w%", lambda s: round(s.mean(), 1)),
            PostExit12w_Avg_pct=("exit_fwd12w%", lambda s: round(s.mean(), 1)),
            AvoidDrop4w_pct=("exit_fwd4w%", lambda s: round((s < 0).mean() * 100, 1)),
        )
        .reset_index()
    )
    print(x_stats.to_string(index=False))

    print("\n=== 3. 各买入规则独立触发统计 (每次触发后 4周 / 12周 / 26周表现) ===")
    sb = pd.DataFrame(standalone_buys)
    sb_stats = (
        sb.groupby("Rule")
        .agg(
            Signals=("r4w", "count"),
            Win4w_pct=("r4w", lambda s: round((s > 0).mean() * 100, 1)),
            Avg4w_pct=("r4w", lambda s: round(s.mean(), 1)),
            Win12w_pct=("r12w", lambda s: round((s > 0).mean() * 100, 1)),
            Avg12w_pct=("r12w", lambda s: round(s.mean(), 1)),
            Win26w_pct=("r26w", lambda s: round((s > 0).mean() * 100, 1)),
            Avg26w_pct=("r26w", lambda s: round(s.mean(), 1)),
        )
        .reset_index()
    )
    print(sb_stats.to_string(index=False))

    print("\n=== 4. 各卖出规则独立触发统计 (每次触发后 4周 / 12周避险成功率) ===")
    ss = pd.DataFrame(standalone_sells)
    ss_stats = (
        ss.groupby("Rule")
        .agg(
            Signals=("r4w", "count"),
            Drop4w_pct=("r4w", lambda s: round((s < 0).mean() * 100, 1)),
            AvgFwd4w_pct=("r4w", lambda s: round(s.mean(), 1)),
            Drop12w_pct=("r12w", lambda s: round((s < 0).mean() * 100, 1)),
            AvgFwd12w_pct=("r12w", lambda s: round(s.mean(), 1)),
        )
        .reset_index()
    )
    print(ss_stats.to_string(index=False))


if __name__ == "__main__":
    main()
