"""Audit BTC 4H Gaussian Channel strategies across parameter sets, Kraken fee tiers, and sub-periods."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("/Users/jessemacree/.gemini/antigravity/brain/1da4a5ea-7033-49da-8c48-4d3afd61e2a3/scratch/BTC_4h.csv")


def pine_gaussian_filter(src: pd.Series, period: int, poles: int) -> pd.Series:
    beta = 0.5 * (1.0 - np.cos(2.0 * np.pi / period))
    g = src.copy()
    for _ in range(poles):
        g = g.ewm(alpha=beta, adjust=False).mean()
    return g


def run_gac_backtest(
    df: pd.DataFrame,
    g_per: int,
    g_poles: int,
    g_mult: float,
    fee_rate: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    # Compute causal indicators on full history, then slice evaluation window cleanly
    mid = pine_gaussian_filter(df["close"], g_per, g_poles)
    diff = (df["close"] - mid).abs()
    dev = pine_gaussian_filter(diff, g_per, g_poles)
    upper = mid + g_mult * dev

    cross_up = ((df["close"] > upper) & (df["close"].shift(1) <= upper.shift(1))).to_numpy(dtype=bool)
    cross_dn = ((df["close"] < mid) & (df["close"].shift(1) >= mid.shift(1))).to_numpy(dtype=bool)

    ts = df["timestamp"]
    mask = np.ones(len(df), dtype=bool)
    if start_date is not None:
        mask &= (ts >= pd.Timestamp(start_date)).to_numpy()
    if end_date is not None:
        mask &= (ts <= pd.Timestamp(end_date)).to_numpy()

    pos_idx = np.where(mask)[0]
    i0, i1 = int(pos_idx[0]), int(pos_idx[-1])

    opens = df["open"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    ts_sub = ts.iloc[i0 : i1 + 1].reset_index(drop=True)

    cash = 10000.0
    shares = 0.0
    in_pos = False
    entry_px = 0.0
    trades = 0
    wins = 0
    equity = np.empty(i1 - i0 + 1, dtype=np.float64)

    for step, i in enumerate(range(i0, i1 + 1)):
        if step == 0:
            equity[step] = cash
            continue
        if (not in_pos) and cross_up[i - 1]:
            exec_px = opens[i]
            shares = (cash * (1.0 - fee_rate)) / exec_px
            cash = 0.0
            in_pos = True
            entry_px = exec_px
        elif in_pos and cross_dn[i - 1]:
            exec_px = opens[i]
            cash = shares * exec_px * (1.0 - fee_rate)
            shares = 0.0
            in_pos = False
            trades += 1
            if exec_px * (1.0 - fee_rate) > entry_px * (1.0 + fee_rate):
                wins += 1
        equity[step] = cash + shares * closes[i]

    eq_s = pd.Series(equity, index=ts_sub)
    days = (ts_sub.iloc[-1] - ts_sub.iloc[0]).total_seconds() / 86400.0
    years = max(days / 365.25, 1e-6)
    final_val = float(equity[-1])
    cagr = ((final_val / 10000.0) ** (1.0 / years) - 1.0) * 100.0
    mdd = float(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100.0)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    # Daily resampled Sharpe (annualized 365d for crypto)
    d_eq = eq_s.resample("1D").last().dropna()
    d_ret = d_eq.pct_change().dropna()
    sharpe = float(d_ret.mean() / d_ret.std() * np.sqrt(365.0)) if len(d_ret) > 10 and d_ret.std() > 0 else 0.0

    # Yearly returns
    yr_rets = {}
    for y, grp in eq_s.groupby(eq_s.index.year):
        if len(grp) > 5:
            yr_rets[int(y)] = round((float(grp.iloc[-1]) / float(grp.iloc[0]) - 1.0) * 100.0, 1)

    return {
        "Final": round(final_val, 2),
        "Mult": round(final_val / 10000.0, 2),
        "CAGR%": round(cagr, 2),
        "MaxDD%": round(mdd, 2),
        "Calmar": round(calmar, 3),
        "Sharpe": round(sharpe, 2),
        "Trades": trades,
        "WinRate%": round(wins / trades * 100.0, 1) if trades > 0 else 0.0,
        "yearly": yr_rets,
    }


def run_bh(df: pd.DataFrame, start_date: str | None = None, end_date: str | None = None) -> dict:
    sub = df.copy()
    if start_date is not None:
        sub = sub[sub["timestamp"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        sub = sub[sub["timestamp"] <= pd.Timestamp(end_date)]
    eq = 10000.0 * (sub["close"] / sub["open"].iloc[0])
    eq.index = sub["timestamp"]
    days = (sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0]).total_seconds() / 86400.0
    years = max(days / 365.25, 1e-6)
    final_val = float(eq.iloc[-1])
    cagr = ((final_val / 10000.0) ** (1.0 / years) - 1.0) * 100.0
    mdd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100.0)
    d_ret = eq.resample("1D").last().dropna().pct_change().dropna()
    sharpe = float(d_ret.mean() / d_ret.std() * np.sqrt(365.0)) if d_ret.std() > 0 else 0.0
    return {
        "Final": round(final_val, 2),
        "Mult": round(final_val / 10000.0, 2),
        "CAGR%": round(cagr, 2),
        "MaxDD%": round(mdd, 2),
        "Calmar": round(cagr / abs(mdd), 3) if mdd < 0 else 0.0,
        "Sharpe": round(sharpe, 2),
        "Trades": 1,
        "WinRate%": 100.0 if final_val > 10000.0 else 0.0,
    }


def main() -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded BTC 4H: {len(df)} bars ({df['timestamp'].min()} ~ {df['timestamp'].max()})\n")

    configs = [
        ("LiveBot (15,1,1.8)", 15, 1, 1.8),
        ("Backtest_62X (30,2,1.5)", 30, 2, 1.5),
        ("Backtest_73X (40,1,2.0)", 40, 1, 2.0),
    ]
    fees = [
        ("0.04% (BinanceVIP)", 0.0004),
        ("0.25% (KrakenMaker)", 0.0025),
        ("0.40% (KrakenTaker)", 0.0040),
    ]
    windows = [
        ("Full (2017-08 ~ 2026-11)", None, None),
        ("InSample (2017-08 ~ 2021-12)", None, "2021-12-31 23:59:59"),
        ("Recent_OOS (2022-01 ~ 2026-11)", "2022-01-01", None),
        ("2026_YTD (2026-01 ~ 2026-11)", "2026-01-01", None),
        ("LiveWindow (2026-03-30 ~ 2026-11)", "2026-03-30", None),
    ]

    for wname, s, e in windows:
        rows = [{"Strategy": "Buy&Hold BTC", "Fee": "-", **run_bh(df, s, e)}]
        for cname, per, pol, mul in configs:
            for fname, fr in fees:
                res = run_gac_backtest(df, per, pol, mul, fr, s, e)
                rows.append(
                    {
                        "Strategy": cname,
                        "Fee": fname,
                        "Final": res["Final"],
                        "Mult": res["Mult"],
                        "CAGR%": res["CAGR%"],
                        "MaxDD%": res["MaxDD%"],
                        "Calmar": res["Calmar"],
                        "Sharpe": res["Sharpe"],
                        "Trades": res["Trades"],
                        "WinRate%": res["WinRate%"],
                    }
                )
        print(f"=== {wname} ===")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    print("=== YEAR-BY-YEAR RETURN (%) AT REALISTIC KRAKEN TAKER FEE (0.40%) vs 0.04% ===")
    yr_rows = []
    for cname, per, pol, mul in configs:
        for fname, fr in [("0.04%", 0.0004), ("0.40%", 0.0040)]:
            res = run_gac_backtest(df, per, pol, mul, fr)
            row = {"Strategy": cname, "Fee": fname}
            for y in range(2018, 2027):
                row[str(y)] = res["yearly"].get(y, 0.0)
            yr_rows.append(row)
    print(pd.DataFrame(yr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
