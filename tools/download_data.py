"""Incremental daily stock data downloader using yfinance."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "tickers.json"


def load_config(config_path: Path) -> dict:
    """Load ticker configuration from JSON."""
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not config.get("tickers"):
        raise ValueError(f"No tickers configured in {config_path}")
    return config


def resolve_data_dir(config_path: Path, data_dir_setting: str | None) -> Path:
    """Resolve data directory relative to the config file if not absolute."""
    raw = Path(data_dir_setting or "../data")
    if raw.is_absolute():
        return raw
    return (config_path.resolve().parent / raw).resolve()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present and normalize Date index/column."""
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def download_ticker_incremental(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    full_refresh: bool = False,
) -> None:
    """Download daily OHLCV data for a single ticker incrementally."""
    csv_path = output_dir / f"{ticker}.csv"
    existing_df = pd.DataFrame()
    fetch_start = start_date

    if not full_refresh and csv_path.exists() and csv_path.stat().st_size > 0:
        existing_df = pd.read_csv(csv_path)
        if not existing_df.empty and "Date" in existing_df.columns:
            existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.strftime("%Y-%m-%d")
            existing_df = existing_df.dropna(subset=["Open", "High", "Low", "Close"])
            last_date_str = existing_df["Date"].max()
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            # Re-fetch from the last recorded date so any partial intraday bar is refreshed.
            fetch_start = last_date.strftime("%Y-%m-%d")
            if last_date > date.today():
                print(f"[{ticker}] Already up to date (latest: {last_date_str}).")
                return
            print(f"[{ticker}] Existing data up to {last_date_str}; fetching from {fetch_start}...")
        else:
            print(f"[{ticker}] Existing file empty or missing Date; fetching from {fetch_start}...")
    else:
        mode = "Full refresh" if full_refresh else "No local file found"
        print(f"[{ticker}] {mode}; fetching from {fetch_start}...")

    # yfinance end parameter is exclusive, so add 1 day to include end_date.
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() + timedelta(days=1)
    new_df = yf.download(
        ticker,
        start=fetch_start,
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if new_df is None or new_df.empty:
        if not existing_df.empty:
            print(f"[{ticker}] No new bars available (latest: {existing_df['Date'].max()}).")
        else:
            print(f"[{ticker}] Warning: no data returned.")
        return

    new_df = normalize_df(new_df)

    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)
        added_rows = len(combined) - len(existing_df)
    else:
        combined = new_df
        added_rows = len(combined)

    combined.to_csv(csv_path, index=False)
    print(
        f"[{ticker}] Saved {len(combined)} total rows "
        f"(+{added_rows} new) -> {csv_path} "
        f"[{combined['Date'].iloc[0]} ~ {combined['Date'].iloc[-1]}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental daily stock data downloader.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to tickers JSON config file.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="Inclusive end date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-download full history from start_date instead of incremental update.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional comma-separated tickers to update (default: all tickers in config).",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    start_date = config.get("start_date", "2010-01-01")
    output_dir = resolve_data_dir(config_path, config.get("data_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    target_tickers = config["tickers"]
    if args.tickers.strip():
        target_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    for ticker in target_tickers:
        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean:
            continue
        try:
            download_ticker_incremental(
                ticker=ticker_clean,
                start_date=start_date,
                end_date=args.end_date,
                output_dir=output_dir,
                full_refresh=args.full,
            )
        except Exception as exc:
            print(f"[{ticker_clean}] Error: {exc}")


if __name__ == "__main__":
    main()
