"""Backtest SOXL cross-timeframe Fisher + Volume + 30W SMA strategies (2015-2025)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import fisher_transform, sma


def load_soxl_daily() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "SOXL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date")
    ratio = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * ratio
    return df.set_index("Date")


def compute_metrics(equity: pd.Series, trades: list[dict], initial: float = 10000.0) -> dict:
    final_val = float(equity.iloc[-1])
    total_ret = (final_val / initial - 1.0) * 100.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((final_val / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0) * 100.0
    max_dd = float(dd.min())

    daily_ret = equity.pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0
        else 0.0
    )

    wins = sum(1 for t in trades if t["ret_pct"] > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0

    return {
        "Final ($)": round(final_val, 2),
        "Total Return%": round(total_ret, 1),
        "CAGR%": round(cagr, 1),
        "MaxDD%": round(max_dd, 1),
        "Sharpe": round(sharpe, 2),
        "Trades": len(trades),
        "WinRate%": round(win_rate, 1),
    }


def run_weekly_signal_backtest(
    daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.0005,
    use_trend_reentry: bool = True,
) -> tuple[pd.Series, list[dict]]:
    """Run causal weekly-evaluated backtest executed at next trading day's Open."""
    # Weekly bars (Friday close)
    w = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Sig"] = wf["fisher_signal"]
    w["d_W_Fisher"] = w["W_Fisher"].diff()
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    w["W_SMA30"] = sma(w["Close"], 30)

    # Monthly bars (completed months only, shifted forward by 1 day so no look-ahead)
    m = daily.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    mf = fisher_transform(m["High"], m["Low"], period=10)
    m["M_Fisher"] = mf["fisher"]
    m["M_Vol_R"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean()

    # Map each trading day to the latest completed weekly signal
    w_dates = w.index
    m_dates = m.index

    bt_daily = daily.loc[start_date:end_date].copy()
    dates = bt_daily.index

    cash = initial_capital
    shares = 0.0
    in_pos = False
    pending_action: str | None = None
    entry_price = 0.0
    entry_date = None
    trades: list[dict] = []
    equity_curve = np.zeros(len(dates), dtype=float)

    # Track which weekly bar was last processed
    last_w_idx = -1

    for i, dt in enumerate(dates):
        o = float(bt_daily["Open"].iloc[i])
        c = float(bt_daily["Close"].iloc[i])

        # Execute pending order at today's Open
        if pending_action == "BUY" and not in_pos:
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            in_pos = True
            entry_price = o
            entry_date = dt
        elif pending_action == "SELL" and in_pos:
            proceeds = shares * o * (1.0 - fee_rate)
            ret_pct = (o * (1.0 - fee_rate) / (entry_price * (1.0 + fee_rate)) - 1.0) * 100.0
            trades.append(
                {
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": dt.strftime("%Y-%m-%d"),
                    "entry_px": round(entry_price, 2),
                    "exit_px": round(o, 2),
                    "ret_pct": round(ret_pct, 2),
                }
            )
            cash = proceeds
            shares = 0.0
            in_pos = False
        pending_action = None

        equity_curve[i] = shares * c if in_pos else cash

        # Check if a weekly bar closed on or before today (and hasn't been evaluated yet)
        w_pos = int(w_dates.searchsorted(dt, side="right")) - 1
        if w_pos > last_w_idx and w_pos >= 1:
            last_w_idx = w_pos
            row_w = w.iloc[w_pos]
            m_pos = int(m_dates.searchsorted(w_dates[w_pos], side="right")) - 1
            m_fisher = float(m["M_Fisher"].iloc[m_pos]) if m_pos >= 0 and pd.notna(m["M_Fisher"].iloc[m_pos]) else 0.0
            m_vol_r = float(m["M_Vol_R"].iloc[m_pos]) if m_pos >= 0 and pd.notna(m["M_Vol_R"].iloc[m_pos]) else 1.0

            wf_val = row_w["W_Fisher"]
            dwf_val = row_w["d_W_Fisher"]
            w_vol_r = row_w["W_Vol_R"]
            sma30 = row_w["W_SMA30"]

            if pd.notna(wf_val) and pd.notna(dwf_val) and pd.notna(sma30):
                # Bottom entry: weekly Fisher deeply oversold + turning up OR volume surge
                bottom_buy = (wf_val < -1.5 and dwf_val > 0) or (wf_val < -1.5 and w_vol_r > 1.3) or (wf_val < -1.2 and m_vol_r > 1.4)
                # Trend re-entry: reclaimed 30W SMA with positive Fisher momentum
                trend_buy = use_trend_reentry and (row_w["Close"] > sma30) and (wf_val > row_w["W_Fisher_Sig"])

                # Top exit: monthly Fisher overheated (> 2.0) AND weekly Fisher turning down
                top_sell = (m_fisher > 2.0 and dwf_val < 0 and row_w["Close"] < sma30 * 1.05) or (
                    m_fisher > 2.2 and dwf_val < -0.3
                )
                # Protective trend stop: broke below 30W SMA while weekly Fisher turned negative
                trend_sell = use_trend_reentry and (row_w["Close"] < sma30) and (wf_val < row_w["W_Fisher_Sig"]) and (wf_val < 0.5)

                if not in_pos and (bottom_buy or trend_buy):
                    pending_action = "BUY"
                elif in_pos and (top_sell or trend_sell):
                    pending_action = "SELL"

    # Close open position in trade log at end_date for stats (without forcing cash exit on curve)
    if in_pos:
        last_c = float(bt_daily["Close"].iloc[-1])
        ret_pct = (last_c / entry_price - 1.0) * 100.0
        trades.append(
            {
                "entry": entry_date.strftime("%Y-%m-%d"),
                "exit": dates[-1].strftime("%Y-%m-%d") + " (Open)",
                "entry_px": round(entry_price, 2),
                "exit_px": round(last_c, 2),
                "ret_pct": round(ret_pct, 2),
            }
        )

    return pd.Series(equity_curve, index=dates), trades


def main() -> None:
    daily = load_soxl_daily()
    start_date, end_date = "2015-01-01", "2025-12-31"
    bt_slice = daily.loc[start_date:end_date]

    # Buy & Hold benchmark
    bh_equity = 10000.0 * (bt_slice["Close"] / bt_slice["Close"].iloc[0])
    bh_trades = [
        {
            "entry": bt_slice.index[0].strftime("%Y-%m-%d"),
            "exit": bt_slice.index[-1].strftime("%Y-%m-%d"),
            "ret_pct": round((bh_equity.iloc[-1] / 10000.0 - 1.0) * 100.0, 2),
        }
    ]

    eq_pure, tr_pure = run_weekly_signal_backtest(daily, start_date, end_date, use_trend_reentry=False)
    eq_trend, tr_trend = run_weekly_signal_backtest(daily, start_date, end_date, use_trend_reentry=True)

    summary = pd.DataFrame(
        [
            {"Strategy": "Buy & Hold (SOXL)", **compute_metrics(bh_equity, bh_trades)},
            {"Strategy": "A: 纯顶底拐点 (Fisher+量)", **compute_metrics(eq_pure, tr_pure)},
            {"Strategy": "B: 顶底拐点 + 30周均线趋势", **compute_metrics(eq_trend, tr_trend)},
        ]
    )
    print(f"=== SOXL Backtest ({start_date} ~ {end_date}, Initial=$10,000, Fee=0.05%) ===")
    print(summary.to_string(index=False))

    print("\n=== Strategy B Trades ===")
    print(pd.DataFrame(tr_trend).to_string(index=False))


if __name__ == "__main__":
    main()
