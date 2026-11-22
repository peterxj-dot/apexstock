"""Compare original vs refined SOXL strategies (2015-2025)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import fisher_transform, sma
from backtest_soxl import compute_metrics, load_soxl_daily, run_weekly_signal_backtest


def run_refined_backtest(
    daily: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: str = "2025-12-31",
    initial_capital: float = 10000.0,
    fee_rate: float = 0.0005,
    s3_sma_buf: float = 0.95,
    s3_fisher_max: float = -0.3,
) -> tuple[pd.Series, list[dict]]:
    """Refined strategy: drop B3, tighten S3 to reduce whipsaws around 30W SMA."""
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

    w_dates = w.index
    m_dates = m.index
    bt_daily = daily.loc[start_date:end_date]
    dates = bt_daily.index

    cash = initial_capital
    shares = 0.0
    in_pos = False
    pending_action: str | None = None
    entry_price = 0.0
    entry_date = None
    trades: list[dict] = []
    equity_curve = np.zeros(len(dates), dtype=float)
    last_w_idx = -1

    for i, dt in enumerate(dates):
        o = float(bt_daily["Open"].iloc[i])
        c = float(bt_daily["Close"].iloc[i])

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

        w_pos = int(w_dates.searchsorted(dt, side="right")) - 1
        if w_pos > last_w_idx and w_pos >= 1:
            last_w_idx = w_pos
            rw = w.iloc[w_pos]
            m_pos = int(m_dates.searchsorted(w_dates[w_pos], side="right")) - 1
            mf_val = float(m["M_Fisher"].iloc[m_pos]) if m_pos >= 0 and pd.notna(m["M_Fisher"].iloc[m_pos]) else 0.0

            wf_v = rw["W_Fisher"]
            dwf_v = rw["d_W_Fisher"]
            wvr_v = rw["W_Vol_R"]
            s30 = rw["W_SMA30"]

            if pd.notna(wf_v) and pd.notna(dwf_v) and pd.notna(s30):
                b1 = wf_v < -1.5 and dwf_v > 0
                b2 = wf_v < -1.5 and wvr_v > 1.3
                b4 = rw["Close"] > s30 and wf_v > rw["W_Fisher_Sig"]

                s1 = mf_val > 2.0 and dwf_v < 0 and rw["Close"] < s30 * 1.05
                s2 = mf_val > 2.2 and dwf_v < -0.3
                s3 = rw["Close"] < s30 * s3_sma_buf and wf_v < rw["W_Fisher_Sig"] and wf_v < s3_fisher_max

                if not in_pos and (b1 or b2 or b4):
                    pending_action = "BUY"
                elif in_pos and (s1 or s2 or s3):
                    pending_action = "SELL"

    if in_pos:
        last_c = float(bt_daily["Close"].iloc[-1])
        ret_pct = (last_c / (entry_price * (1.0 + fee_rate)) - 1.0) * 100.0
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

    bh_equity = 10000.0 * (bt_slice["Close"] / bt_slice["Close"].iloc[0])
    bh_trades = [{"ret_pct": round((bh_equity.iloc[-1] / 10000.0 - 1.0) * 100.0, 2)}]

    eq_a, tr_a = run_weekly_signal_backtest(daily, start_date, end_date, use_trend_reentry=False)
    eq_b, tr_b = run_weekly_signal_backtest(daily, start_date, end_date, use_trend_reentry=True)
    eq_c1, tr_c1 = run_refined_backtest(daily, start_date, end_date, s3_sma_buf=0.95, s3_fisher_max=-0.3)
    eq_c2, tr_c2 = run_refined_backtest(daily, start_date, end_date, s3_sma_buf=1.00, s3_fisher_max=-0.5)

    summary = pd.DataFrame(
        [
            {"Strategy": "Buy & Hold (SOXL)", **compute_metrics(bh_equity, bh_trades)},
            {"Strategy": "A: 纯顶底拐点 (B1~B3 / S1~S2)", **compute_metrics(eq_a, tr_a)},
            {"Strategy": "B: 原版全规则 (B1~B4 / S1~S3)", **compute_metrics(eq_b, tr_b)},
            {"Strategy": "C1: 删B3 + 收紧S3(<0.95*30W & F<-0.3)", **compute_metrics(eq_c1, tr_c1)},
            {"Strategy": "C2: 删B3 + 收紧S3(<30W & F<-0.5)", **compute_metrics(eq_c2, tr_c2)},
        ]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
