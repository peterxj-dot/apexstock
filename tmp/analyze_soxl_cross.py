"""Test cross-timeframe and delta/slope indicator rules on SOXL Top/Bottom zones."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import fisher_transform, historical_volatility, macd, rsi, sma
from analyze_soxl_zones import load_soxl, zigzag_zones


def eval_rule(name: str, cond: pd.Series, target: pd.Series) -> dict:
    valid = cond.notna() & target.notna()
    c = cond[valid].astype(bool)
    t = target[valid].astype(bool)
    triggers = int(c.sum())
    hits = int((c & t).sum())
    total_pos = int(t.sum())
    base_rate = total_pos / len(t) if len(t) else 0.0
    precision = hits / triggers if triggers else 0.0
    recall = hits / total_pos if total_pos else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0
    return {
        "Rule": name,
        "Triggers": triggers,
        "Hits": hits,
        "Precision%": round(precision * 100, 1),
        "Recall%": round(recall * 100, 1),
        "Lift": round(lift, 2),
    }


def main() -> None:
    df = load_soxl()
    m = df.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    mf = fisher_transform(m["High"], m["Low"], period=10)
    m["M_Fisher"] = mf["fisher"]
    m["M_Fisher_Sig"] = mf["fisher_signal"]
    m["M_Fisher_Diff"] = mf["fisher"] - mf["fisher_signal"]
    m["d_M_Fisher"] = m["M_Fisher"].diff()
    m["M_RSI"] = rsi(m["Close"], 14)
    m["d_M_RSI"] = m["M_RSI"].diff()
    m["M_Vol_R"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean()
    m["d_M_Vol_R"] = m["M_Vol_R"].diff()

    w = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Diff"] = wf["fisher"] - wf["fisher_signal"]
    w["d_W_Fisher"] = w["W_Fisher"].diff()
    w["W_RSI"] = rsi(w["Close"], 14)
    w["d_W_RSI"] = w["W_RSI"].diff()
    w["W_HV20"] = historical_volatility(w["Close"], period=20, annualization_factor=52)
    w["d_W_HV20"] = w["W_HV20"].diff()
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    w["W_SMA30"] = sma(w["Close"], 30)
    w["W_SMA200"] = sma(w["Close"], 200)
    w["Dist30"] = (w["Close"] / w["W_SMA30"] - 1.0) * 100.0
    w["Dist200"] = (w["Close"] / w["W_SMA200"] - 1.0) * 100.0

    wm = w.resample("ME").last()
    joined = m.join(
        wm[
            [
                "W_Fisher", "W_Fisher_Diff", "d_W_Fisher", "W_RSI", "d_W_RSI",
                "W_HV20", "d_W_HV20", "W_Vol_R", "Dist30", "Dist200",
            ]
        ],
        how="left",
    )

    _, labels = zigzag_zones(m)
    joined["Zone"] = labels
    is_top = joined["Zone"] == "Top"
    is_bot = joined["Zone"] == "Bottom"

    # Cross-timeframe & delta rules for TOP
    top_rules = [
        ("M_Fisher > 2.0", joined["M_Fisher"] > 2.0),
        ("M_Fisher > 2.0 & d_M_Fisher < 0 (月线高位拐头)", (joined["M_Fisher"] > 2.0) & (joined["d_M_Fisher"] < 0)),
        ("M_Fisher > 2.0 & M_Fisher_Diff < 0 (月线高位死叉)", (joined["M_Fisher"] > 2.0) & (joined["M_Fisher_Diff"] < 0)),
        ("M_Fisher > 2.0 & W_Fisher < M_Fisher (周线先于月线转弱)", (joined["M_Fisher"] > 2.0) & (joined["W_Fisher"] < joined["M_Fisher"])),
        ("M_Fisher > 2.0 & d_W_Fisher < 0 (月高位+周Fisher向下)", (joined["M_Fisher"] > 2.0) & (joined["d_W_Fisher"] < 0)),
        ("M_Fisher > 2.0 & Dist200 > 100% (月高位+远高于200W)", (joined["M_Fisher"] > 2.0) & (joined["Dist200"] > 100)),
        ("M_Fisher > 2.0 & W_Fisher < M_Fisher & d_M_RSI < 0", (joined["M_Fisher"] > 2.0) & (joined["W_Fisher"] < joined["M_Fisher"]) & (joined["d_M_RSI"] < 0)),
    ]

    # Cross-timeframe & delta rules for BOTTOM
    bot_rules = [
        ("W_Fisher < -1.5", joined["W_Fisher"] < -1.5),
        ("W_Fisher < -1.5 & M_Vol_R > 1.3 (周Fisher极低+放量)", (joined["W_Fisher"] < -1.5) & (joined["M_Vol_R"] > 1.3)),
        ("W_Fisher < -1.5 & d_W_Fisher > 0 (周Fisher极低后拐头)", (joined["W_Fisher"] < -1.5) & (joined["d_W_Fisher"] > 0)),
        ("W_Fisher < -1.5 & W_Fisher > M_Fisher (周线领先月线反弹)", (joined["W_Fisher"] < -1.5) & (joined["W_Fisher"] > joined["M_Fisher"])),
        ("W_HV20 > 90 & d_W_HV20 < 0 (高波动率见顶回落)", (joined["W_HV20"] > 90) & (joined["d_W_HV20"] < 0)),
        ("Dist200 < 20% & W_Fisher < -1.2 & M_Vol_R > 1.3", (joined["Dist200"] < 20) & (joined["W_Fisher"] < -1.2) & (joined["M_Vol_R"] > 1.3)),
        ("M_Vol_R > 1.4 & d_M_Vol_R < 0 & W_Fisher < 0 (放巨量后缩量企稳)", (joined["M_Vol_R"] > 1.4) & (joined["d_M_Vol_R"] < 0) & (joined["W_Fisher"] < 0)),
    ]

    print(f"Base rates -> Top: {is_top.mean()*100:.1f}%, Bottom: {is_bot.mean()*100:.1f}%\n")
    print("=== TOP Rules (Cross-Timeframe & Delta) ===")
    print(pd.DataFrame([eval_rule(n, c, is_top) for n, c in top_rules]).to_string(index=False))
    print("\n=== BOTTOM Rules (Cross-Timeframe & Delta) ===")
    print(pd.DataFrame([eval_rule(n, c, is_bot) for n, c in bot_rules]).to_string(index=False))


if __name__ == "__main__":
    main()
