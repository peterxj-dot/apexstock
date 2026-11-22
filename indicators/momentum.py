"""Momentum and oscillator technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 and avg_gain > 0, RSI is 100; when both are 0, RSI is 50.
    result = result.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return result


def fisher_transform(
    high: pd.Series,
    low: pd.Series,
    period: int = 10,
) -> pd.DataFrame:
    """Ehlers Fisher Transform indicator.

    Normalizes median price `(High + Low) / 2` over `period` bars and applies
    the Fisher transform `0.5 * ln((1 + x) / (1 - x))`.

    Returns a DataFrame with columns: `fisher`, `fisher_signal`.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    median_price = (high + low) / 2.0
    rolling_min = median_price.rolling(window=period, min_periods=period).min()
    rolling_max = median_price.rolling(window=period, min_periods=period).max()

    n = len(median_price)
    fisher_vals = np.full(n, np.nan, dtype=float)
    signal_vals = np.full(n, np.nan, dtype=float)

    prev_x = 0.0
    prev_fisher = 0.0

    prices = median_price.to_numpy(dtype=float)
    mins = rolling_min.to_numpy(dtype=float)
    maxs = rolling_max.to_numpy(dtype=float)

    for i in range(n):
        if np.isnan(prices[i]) or np.isnan(mins[i]) or np.isnan(maxs[i]):
            continue

        span = maxs[i] - mins[i]
        norm = 0.0 if span == 0.0 else (prices[i] - mins[i]) / span - 0.5
        x = 0.33 * 2.0 * norm + 0.67 * prev_x
        x = float(np.clip(x, -0.999, 0.999))

        fisher_val = 0.5 * float(np.log((1.0 + x) / (1.0 - x))) + 0.5 * prev_fisher
        fisher_vals[i] = fisher_val
        signal_vals[i] = prev_fisher

        prev_x = x
        prev_fisher = fisher_val

    return pd.DataFrame(
        {
            "fisher": fisher_vals,
            "fisher_signal": signal_vals,
        },
        index=median_price.index,
    )
