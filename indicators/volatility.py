"""Volatility technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def historical_volatility(
    close: pd.Series,
    period: int = 20,
    annualization_factor: int = 252,
) -> pd.Series:
    """Annualized rolling log-return historical volatility (in %)."""
    if period <= 1:
        raise ValueError("period must be greater than 1")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=period, min_periods=period).std(ddof=1)
    return rolling_std * np.sqrt(annualization_factor) * 100.0
