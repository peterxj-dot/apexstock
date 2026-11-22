"""Trend and overlay technical indicators."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Returns a DataFrame with columns: `macd`, `macd_signal`, `macd_hist`.
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, and signal periods must be positive")
    if fast >= slow:
        raise ValueError("fast period must be smaller than slow period")

    fast_ema = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        },
        index=series.index,
    )


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns a DataFrame with columns: `bb_mid`, `bb_upper`, `bb_lower`.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if num_std <= 0:
        raise ValueError("num_std must be positive")

    mid = sma(series, period=period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std

    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
        },
        index=series.index,
    )
