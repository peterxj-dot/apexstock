"""Analyze monthly & weekly indicator signatures in SOXL Top vs Bottom zones."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import atr, fisher_transform, historical_volatility, macd, rsi, sma


def load_soxl() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "SOXL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date")
    ratio = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * ratio
    return df.set_index("Date")


def zigzag_zones(m: pd.DataFrame, up_th: float = 1.2, down_th: float = 0.55, max_off: int = 2, tol: float = 0.20):
    pivots = []
    trend = None
    lh_i, lh = 0, m["High"].iloc[0]
    ll_i, ll = 0, m["Low"].iloc[0]

    for i in range(1, len(m)):
        h, l = m["High"].iloc[i], m["Low"].iloc[i]
        if trend is None:
            if h >= ll * (1 + up_th):
                pivots.append((ll_i, "L", ll))
                trend = "UP"
                lh_i, lh = i, h
            elif l <= lh * (1 - down_th):
                pivots.append((lh_i, "H", lh))
                trend = "DOWN"
                ll_i, ll = i, l
            else:
                if h > lh: lh_i, lh = i, h
                if l < ll: ll_i, ll = i, l
        elif trend == "UP":
            if h >= lh:
                lh_i, lh = i, h
            elif l <= lh * (1 - down_th):
                pivots.append((lh_i, "H", lh))
                trend = "DOWN"
                ll_i, ll = i, l
        elif trend == "DOWN":
            if l <= ll:
                ll_i, ll = i, l
            elif h >= ll * (1 + up_th):
                pivots.append((ll_i, "L", ll))
                trend = "UP"
                lh_i, lh = i, h

    labels = np.array(["Neutral"] * len(m), dtype=object)
    for idx, tp, pr in pivots:
        s = e = idx
        if tp == "H":
            bound = pr * (1 - tol)
            for i in range(idx - 1, max(-1, idx - max_off - 1), -1):
                if m["High"].iloc[i] >= bound or m["Close"].iloc[i] >= bound: s = i
                else: break
            for i in range(idx + 1, min(len(m), idx + max_off + 1)):
                if m["High"].iloc[i] >= bound or m["Close"].iloc[i] >= bound: e = i
                else: break
            labels[s : e + 1] = "Top"
        else:
            bound = pr * (1 + tol)
            for i in range(idx - 1, max(-1, idx - max_off - 1), -1):
                if m["Low"].iloc[i] <= bound or m["Close"].iloc[i] <= bound: s = i
                else: break
            for i in range(idx + 1, min(len(m), idx + max_off + 1)):
                if m["Low"].iloc[i] <= bound or m["Close"].iloc[i] <= bound: e = i
                else: break
            labels[s : e + 1] = "Bottom"

    return pivots, labels


def main() -> None:
    df = load_soxl()

    # Monthly bars & indicators
    m = df.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()

    mf = fisher_transform(m["High"], m["Low"], period=10)
    m["M_Fisher"] = mf["fisher"]
    m["M_Fisher_Diff"] = mf["fisher"] - mf["fisher_signal"]
    m["M_RSI14"] = rsi(m["Close"], 14)
    mm = macd(m["Close"], 12, 26, 9)
    m["M_MACD_Hist"] = mm["macd_hist"]
    m["M_Vol_Ratio12"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean()
    m["M_HV12"] = historical_volatility(m["Close"], period=12, annualization_factor=12)

    # Weekly bars & indicators (aligned to month end)
    w = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Diff"] = wf["fisher"] - wf["fisher_signal"]
    w["W_RSI14"] = rsi(w["Close"], 14)
    wm = macd(w["Close"], 12, 26, 9)
    w["W_MACD_Hist"] = wm["macd_hist"]
    w["W_Vol_Ratio20"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    w["W_HV20"] = historical_volatility(w["Close"], period=20, annualization_factor=52)
    w["W_SMA30"] = sma(w["Close"], 30)
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_Dist_SMA30%"] = (w["Close"] / w["W_SMA30"] - 1.0) * 100.0
    w["W_Dist_SMA200%"] = (w["Close"] / w["W_SMA200"] - 1.0) * 100.0

    w_monthly = w.resample("ME").last()
    cols_w = [
        "W_Fisher", "W_Fisher_Diff", "W_RSI14", "W_MACD_Hist",
        "W_Vol_Ratio20", "W_HV20", "W_Dist_SMA30%", "W_Dist_SMA200%",
    ]
    joined = m.join(w_monthly[cols_w], how="left")

    pivots, labels = zigzag_zones(m)
    joined["Zone"] = labels

    features = [
        "M_Fisher", "M_Fisher_Diff", "M_RSI14", "M_Vol_Ratio12", "M_HV12",
        "W_Fisher", "W_Fisher_Diff", "W_RSI14", "W_Vol_Ratio20", "W_HV20",
        "W_Dist_SMA30%", "W_Dist_SMA200%",
    ]

    print("Zone counts:", joined["Zone"].value_counts().to_dict())
    print("\n=== Median Indicator Values by Zone ===")
    med = joined.groupby("Zone")[features].median().T[["Bottom", "Neutral", "Top"]]
    print(med.round(2).to_string())

    print("\n=== Mean Indicator Values by Zone ===")
    mean = joined.groupby("Zone")[features].mean().T[["Bottom", "Neutral", "Top"]]
    print(mean.round(2).to_string())

    print("\n=== Exact Monthly Pivot Snapshots ===")
    snap_cols = ["Close", "M_Fisher", "M_Vol_Ratio12", "M_RSI14", "W_Fisher", "W_RSI14", "W_Vol_Ratio20", "W_Dist_SMA30%", "W_Dist_SMA200%"]
    rows = []
    for idx, tp, pr in pivots:
        dt = joined.index[idx]
        r = joined.iloc[idx][snap_cols].to_dict()
        r["Month"] = dt.strftime("%Y-%m")
        r["Type"] = "TOP" if tp == "H" else "BOT"
        rows.append(r)
    snap_df = pd.DataFrame(rows)[["Month", "Type"] + snap_cols]
    print(snap_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
