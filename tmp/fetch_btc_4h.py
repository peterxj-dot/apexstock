"""Fetch public BTCUSDT 4H OHLCV history (2017-08 ~ present) to scratch/BTC_4h.csv."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT_CSV = Path("/Users/jessemacree/.gemini/antigravity/brain/1da4a5ea-7033-49da-8c48-4d3afd61e2a3/scratch/BTC_4h.csv")
ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    start_ts = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
    all_rows: list = []

    base_url = None
    for url in ENDPOINTS:
        try:
            r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "4h", "limit": 5}, timeout=8)
            if r.status_code == 200 and isinstance(r.json(), list):
                base_url = url
                break
        except Exception:
            continue
    if base_url is None:
        raise RuntimeError("Could not reach Binance public klines API")

    cur_ts = start_ts
    while True:
        r = requests.get(
            base_url,
            params={"symbol": "BTCUSDT", "interval": "4h", "startTime": cur_ts, "limit": 1000},
            timeout=10,
        )
        data = r.json()
        if not isinstance(data, list) or not data:
            break
        all_rows.extend(data)
        cur_ts = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.08)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} 4H bars ({df['timestamp'].min()} ~ {df['timestamp'].max()}) to {OUT_CSV}")


if __name__ == "__main__":
    main()
