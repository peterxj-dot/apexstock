"""Build a dedicated SOXL monthly LOG chart with top/bottom zones and verified strategy overlays."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import ema, fisher_transform, historical_volatility, sma

DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "SOXL.csv"
SMH_PATH = DATA_DIR / "SMH.csv"
OUTPUT_HTML = ROOT / "soxl_monthly_zones.html"

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005
BT_START = "2015-01-01"
BT_END = "2026-11-19"


def load_daily_adjusted(csv_path: Path) -> pd.DataFrame:
    """Load daily CSV, drop incomplete rows, and split/dividend-adjust OHLC."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    if "Adj Close" in df.columns:
        ratio = (df["Adj Close"] / df["Close"]).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * ratio
    return df


def load_soxl_daily_adjusted(csv_path: Path) -> pd.DataFrame:
    """Backward-compatible wrapper for loading adjusted SOXL daily data."""
    return load_daily_adjusted(csv_path)


def build_monthly_and_weekly(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Aggregate daily data to monthly and weekly OHLCV records."""
    indexed = df.set_index("Date")

    monthly = (
        indexed.resample("ME")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    monthly_records = [
        {
            "m": idx.strftime("%Y-%m"),
            "o": round(float(r.Open), 4),
            "h": round(float(r.High), 4),
            "l": round(float(r.Low), 4),
            "c": round(float(r.Close), 4),
            "v": int(r.Volume),
        }
        for idx, r in zip(monthly.index, monthly.itertuples(index=False))
    ]

    weekly = (
        indexed.resample("W-FRI")
        .agg({"Close": "last"})
        .dropna(subset=["Close"])
    )
    weekly["sma30w"] = weekly["Close"].rolling(30, min_periods=30).mean()
    weekly["sma200w"] = weekly["Close"].rolling(200, min_periods=200).mean()
    weekly_by_month = weekly.resample("ME").last()
    weekly_sma_records = [
        {
            "m": idx.strftime("%Y-%m"),
            "s30": round(float(r.sma30w), 4) if pd.notna(r.sma30w) else None,
            "s200": round(float(r.sma200w), 4) if pd.notna(r.sma200w) else None,
        }
        for idx, r in zip(weekly_by_month.index, weekly_by_month.itertuples(index=False))
    ]

    return monthly_records, weekly_sma_records


def _agg_periodic(daily_idx_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    with_dt = daily_idx_df.assign(_dt=daily_idx_df.index)
    grouped = (
        with_dt.groupby(pd.Grouper(freq=freq))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    grouped.index = pd.DatetimeIndex(grouped["_dt"])
    return grouped.drop(columns=["_dt"])


def build_causal_features(df_soxl: pd.DataFrame, df_smh: pd.DataFrame) -> dict:
    """Build strictly causal daily, weekly, and completed-month features for SOXL driven by SMH."""
    daily = df_soxl.set_index("Date").copy()
    bench = df_smh.set_index("Date").reindex(daily.index, method="ffill")

    daily["HV20"] = historical_volatility(daily["Close"], period=20, annualization_factor=252)
    daily["Bench_Close"] = bench["Close"]
    daily["Bench_SMA200"] = sma(bench["Close"], 200)

    w = _agg_periodic(daily, "W-FRI")
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_Fisher_Sig"] = wf["fisher_signal"]
    w["dW_Fish"] = w["W_Fisher"].diff()
    w["W_SMA30"] = sma(w["Close"], 30)
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_EMA10"] = ema(w["Close"], 10)
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()

    w["Low_vs_200W%"] = (w["Low"] / w["W_SMA200"] - 1.0) * 100.0
    w["Close_vs_30W%"] = (w["Close"] / w["W_SMA30"] - 1.0) * 100.0

    hi_arr = w["High"].to_numpy(dtype=np.float64)
    lo_arr = w["Low"].to_numpy(dtype=np.float64)
    wks_52h = np.empty(len(w), dtype=np.float64)
    dd52_8wl = np.empty(len(w), dtype=np.float64)
    for k in range(len(w)):
        s52 = max(0, k - 51)
        p52 = s52 + int(np.argmax(hi_arr[s52 : k + 1]))
        wks_52h[k] = float(k - p52)
        min8l = float(np.min(lo_arr[max(0, k - 7) : k + 1]))
        dd52_8wl[k] = (min8l / hi_arr[p52] - 1.0) * 100.0
    w["wks_52H"] = wks_52h
    w["dd52_8wL%"] = dd52_8wl

    m = _agg_periodic(daily, "ME")
    mf = fisher_transform(m["High"], m["Low"], period=10)
    m_fish = mf["fisher"]
    m_sig = mf["fisher_signal"]
    m_cross_up = (m_fish > m_sig) & (m_fish.shift(1) <= m_sig.shift(1)) & (m_fish.shift(1) < 0.8)

    w["M_Fish_prev"] = m_fish.reindex(w.index, method="ffill")
    daily["M_Fish_prev_ffill"] = m_fish.reindex(daily.index, method="ffill")

    w["MF_Lin_Buy"] = False
    for mdt in m.index[m_cross_up.fillna(False)]:
        w_pos = int(w.index.searchsorted(mdt, side="left"))
        if w_pos < len(w):
            w.iloc[w_pos, w.columns.get_loc("MF_Lin_Buy")] = True

    min_low_200_8w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    max4w_vol_r = w["W_Vol_R"].rolling(4, min_periods=1).max()

    w["pre_c7_vol"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (max4w_vol_r >= 1.6)
    w["E1_C7Vol_EMA10"] = w["pre_c7_vol"] & (w["Close"] > w["W_EMA10"])
    w["E2_PrevHigh"] = w["pre_c7_vol"] & (w["Close"] > w["High"].shift(1))
    w["F1_E2_PrevHigh"] = (
        w["pre_c7_vol"]
        & ((w["wks_52H"] >= 30) | (w["dd52_8wL%"] <= -80.0))
        & (w["Close"] > w["High"].shift(1))
    )
    w["C2_F1_or_MFishLin"] = w["F1_E2_PrevHigh"] | w["MF_Lin_Buy"]
    w["ALT_Buy_S1"] = (w["W_Fisher"] < -1.8) & ((w["dW_Fish"] > 0) | (w["W_Vol_R"] > 1.3))

    daily_pos_map = {dt: pos for pos, dt in enumerate(daily.index)}
    week_end_daily_pos = {daily_pos_map[w_dt]: w_pos for w_pos, w_dt in enumerate(w.index)}

    return {
        "daily": daily,
        "weekly": w,
        "week_end_daily_pos": week_end_daily_pos,
    }


def _compute_stats(
    equity: np.ndarray,
    trades: list[dict],
    days_invested: int,
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    n = len(equity)
    final_val = float(equity[-1])
    years_span = n / 252.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0

    cum_max = np.maximum.accumulate(equity)
    max_dd = float(((equity / cum_max) - 1.0).min() * 100.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    daily_ret = np.diff(equity) / equity[:-1]
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252.0)) if std > 1e-12 else 0.0

    trades_cnt = len(trades)
    win_rate = (sum(1 for t in trades if t["ret_pct"] > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = days_invested / n * 100.0

    yrs = dates.year.to_numpy()
    yearly_rets: dict[int, float] = {}
    prev_eq = initial_capital
    for yr in sorted(set(yrs)):
        last_pos = int(np.where(yrs == yr)[0][-1])
        yr_eq = float(equity[last_pos])
        yearly_rets[int(yr)] = (yr_eq / prev_eq - 1.0) * 100.0
        prev_eq = yr_eq
    worst_year = min(yearly_rets.values()) if yearly_rets else 0.0

    return {
        "final": round(final_val, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "sharpe": round(sharpe, 2),
        "trades": trades_cnt,
        "win_rate": round(win_rate, 1),
        "exposure": round(exposure, 1),
        "worst_year": round(worst_year, 1),
    }


def simulate_d1_x1a_wtsl35(
    feats: dict,
    use_split_50_50: bool = False,
    start_date: str = BT_START,
    end_date: str = BT_END,
) -> dict:
    """Simulate Strategy 3: D1_X1a_WTSL35 (full or 50/50 split entry)."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    tranches = 0
    first_entry_px = 0.0
    avg_entry_px = 0.0
    entry_date_str = ""
    entry_m_str = ""
    tranche2_date_str = "-"
    tranche2_m_str: str | None = None
    tranche2_px: float | None = None
    entry_fill_pos = -1
    trade_cash_start = INITIAL_CAPITAL
    trade_mode = ""

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_entry_mode = ""
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        just_exited = False
        if pending_action == "BUY_FULL" and tranches == 0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            tranches = 2
            first_entry_px = float(o)
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            tranche2_date_str = "-"
            tranche2_m_str = None
            tranche2_px = None
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_HALF" and tranches == 0:
            trade_cash_start = cash
            cost = 0.5 * cash
            shares = (cost * (1.0 - FEE_RATE)) / o
            cash -= cost
            tranches = 1
            first_entry_px = float(o)
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            tranche2_date_str = "-"
            tranche2_m_str = None
            tranche2_px = None
            entry_fill_pos = pos
            trade_mode = pending_entry_mode
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "BUY_TOPUP" and tranches == 1 and shares > 0:
            new_shares = (cash * (1.0 - FEE_RATE)) / o
            tot = shares + new_shares
            avg_entry_px = (shares * avg_entry_px + new_shares * o) / tot
            shares = tot
            cash = 0.0
            tranches = 2
            tranche2_date_str = dt_str
            tranche2_m_str = m_str
            tranche2_px = round(float(o), 2)
            if c > peak_d_close:
                peak_d_close = c
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            proceeds = shares * o * (1.0 - FEE_RATE)
            cash += proceeds
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append(
                {
                    "mode": trade_mode,
                    "entry_date": entry_date_str,
                    "entry_m": entry_m_str,
                    "tranche2_date": tranche2_date_str,
                    "tranche2_m": tranche2_m_str,
                    "tranche2_px": tranche2_px,
                    "entry_px": round(first_entry_px, 2),
                    "avg_entry_px": round(avg_entry_px, 2),
                    "exit_date": dt_str,
                    "exit_m": m_str,
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            if pending_reason.startswith("SL_"):
                cooldown_until_pos = pos + 10
            shares = 0.0
            tranches = 0
            avg_entry_px = 0.0
            just_exited = True
        pending_action = ""
        pending_reason = ""
        pending_entry_mode = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end or pos not in week_end_daily_pos:
            continue

        w_row = w_df.iloc[week_end_daily_pos[pos]]

        if shares > 0:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            mf = w_row["M_Fish_prev"]
            unmatured = (pd.isna(mf) or float(mf) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
            elif pos > entry_fill_pos and pd.notna(mf) and float(mf) >= 2.0:
                vs30 = w_row["Close_vs_30W%"]
                if pd.notna(vs30) and float(vs30) < 0.0:
                    pending_action = "SELL"
                    pending_reason = "M>=2&<30W0"
                elif peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
                    pending_action = "SELL"
                    pending_reason = "M>=2&WTSL35%"

            if pending_action == "" and use_split_50_50 and tranches == 1 and bool(w_row["E1_C7Vol_EMA10"]):
                pending_action = "BUY_TOPUP"

        elif not just_exited and pos >= cooldown_until_pos:
            if use_split_50_50:
                if bool(w_row["E1_C7Vol_EMA10"]):
                    pending_action = "BUY_FULL"
                    pending_entry_mode = "BOT_E1"
                elif bool(w_row["E2_PrevHigh"]):
                    pending_action = "BUY_HALF"
                    pending_entry_mode = "BOT_SPLIT"
            elif bool(w_row["E2_PrevHigh"]):
                pending_action = "BUY_FULL"
                pending_entry_mode = "BOT_E2"

    if shares > 0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trades.append(
            {
                "mode": trade_mode,
                "entry_date": entry_date_str,
                "entry_m": entry_m_str,
                "tranche2_date": tranche2_date_str,
                "tranche2_m": tranche2_m_str,
                "tranche2_px": tranche2_px,
                "entry_px": round(first_entry_px, 2),
                "avg_entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    if use_split_50_50:
        return {
            "id": "D1_X1a_WTSL35_Split50",
            "name": "策略 3 分批版: D1_X1a_WTSL35_Split50",
            "short_label": "策略3分批 (D1_X1a_WTSL35_Split50)",
            "rule_desc": "pre_c7_vol + PrevHigh先买50%/EMA10补满 | 假底SL_W25% | M_Fish≥2.0后 破30W或周回撤35%逃顶",
            "stats": stats,
            "trades": trades,
        }
    return {
        "id": "D1_X1a_WTSL35",
        "name": "策略 3 极简大周期: D1_X1a_WTSL35",
        "short_label": "策略3 (D1_X1a_WTSL35)",
        "rule_desc": "pre_c7_vol + 周收盘>上周高点买入 | 假底SL_W25% | M_Fish≥2.0后 破30W或周回撤35%逃顶",
        "stats": stats,
        "trades": trades,
    }


def simulate_h5_adapt_hv(
    feats: dict,
    start_date: str = BT_START,
    end_date: str = BT_END,
) -> dict:
    """Simulate Strategy 4: H5_E2_Promote_S1_AdaptHV."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)
    hv20 = daily_df["HV20"].to_numpy(dtype=np.float64)
    b_close = daily_df["Bench_Close"].to_numpy(dtype=np.float64)
    b_sma200 = daily_df["Bench_SMA200"].to_numpy(dtype=np.float64)
    m_fish_ffill = daily_df["M_Fish_prev_ffill"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    pos_mode = 0  # 0=FLAT, 1=TREND, 2=ALT, 3=BOT
    initial_mode_str = ""
    promoted_in_trade = False
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    last_exit_reason = "INIT"
    peak_d_close = 0.0

    pending = 0
    pending_reason = ""

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, i in enumerate(range(i_start, i_end + 1)):
        o = opens[i]
        c = closes[i]

        just_exited = False
        if pending in (1, 2, 3) and pos_mode == 0:
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            pos_mode = pending
            initial_mode_str = {1: "TREND", 2: "ALT", 3: "BOT"}[pending]
            promoted_in_trade = False
            entry_px = float(o)
            entry_idx = i
            peak_d_close = c
        elif pending == -1 and pos_mode != 0:
            if i <= entry_idx:
                raise AssertionError(f"Same-day buy-sell on {idx[i]}")
            proceeds = shares * o * (1.0 - FEE_RATE)
            ret_pct = ((o * (1.0 - FEE_RATE)) / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
            mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
            trades.append(
                {
                    "mode": mode_label,
                    "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                    "entry_m": idx[entry_idx].strftime("%Y-%m"),
                    "tranche2_date": "-",
                    "tranche2_m": None,
                    "tranche2_px": None,
                    "entry_px": round(entry_px, 2),
                    "avg_entry_px": round(entry_px, 2),
                    "exit_date": idx[i].strftime("%Y-%m-%d"),
                    "exit_m": idx[i].strftime("%Y-%m"),
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            cash = proceeds
            shares = 0.0
            pos_mode = 0
            exit_idx = i
            last_exit_reason = pending_reason
            just_exited = True
        pending = 0
        pending_reason = ""

        equity[step] = cash + shares * c
        if pos_mode != 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if i >= i_end:
            continue

        is_w_end = i in week_end_daily_pos
        w_row = w_df.iloc[week_end_daily_pos[i]] if is_w_end else None

        tb_raw = (not np.isnan(b_sma200[i])) and (b_close[i] > b_sma200[i] * 1.01)
        ma_break = (not np.isnan(b_sma200[i])) and (b_close[i] < b_sma200[i] * 0.99)
        hv = hv20[i]
        mf = m_fish_ffill[i]

        high_mf = (not np.isnan(mf)) and (mf >= 1.5)
        vol_exit = (not np.isnan(hv)) and ((high_mf and hv > 115.0) or (hv > 140.0))
        tb_def = tb_raw and (not np.isnan(hv)) and (hv <= 115.0 * 0.92)
        tb_after_vol = tb_raw and (not np.isnan(hv)) and (hv <= 95.0) and (not vol_exit)

        if pos_mode == 3:
            hit_sl = False
            if is_w_end and w_row is not None:
                m_w = w_row["M_Fish_prev"]
                unmatured = (pd.isna(m_w) or float(m_w) < 1.5) and (peak_d_close < entry_px * 1.50)
                if unmatured and float(w_row["Close"]) < entry_px * 0.75:
                    hit_sl = True
            if hit_sl:
                pending = -1
                pending_reason = "SL_W25"
            if pending == 0 and tb_raw and (not vol_exit) and (not ma_break):
                pos_mode = 1
                promoted_in_trade = True

        elif pos_mode == 1:
            if vol_exit:
                pending = -1
                pending_reason = "VOL_SPIKE"
            elif ma_break:
                pending = -1
                pending_reason = "TREND_BREAK"

        elif pos_mode == 2:
            if c <= entry_px * 0.80:
                pending = -1
                pending_reason = "ALT_FSL"
            elif c >= entry_px * 1.25 or (i - entry_idx) >= 40:
                pending = -1
                pending_reason = "ALT_TP"

        elif not just_exited:
            elapsed = i - exit_idx
            bot_sig = bool(is_w_end and w_row is not None and bool(w_row["E2_PrevHigh"]))
            alt_sig = bool(is_w_end and w_row is not None and bool(w_row["ALT_Buy_S1"]))

            if last_exit_reason == "SL_W25":
                if elapsed >= 10 and bot_sig:
                    pending = 3
            else:
                bot_allowed = (elapsed >= 10) if last_exit_reason in ("ALT_FSL", "ALT_TP") else True
                if last_exit_reason in ("INIT", "TREND_BREAK"):
                    if tb_def:
                        pending = 1
                    elif bot_allowed and bot_sig:
                        pending = 3
                    elif alt_sig:
                        pending = 2
                elif last_exit_reason == "VOL_SPIKE":
                    if tb_after_vol:
                        pending = 1
                    elif bot_allowed and bot_sig:
                        pending = 3
                    elif alt_sig:
                        pending = 2
                elif last_exit_reason in ("ALT_FSL", "ALT_TP"):
                    if elapsed >= 10:
                        if tb_def:
                            pending = 1
                        elif bot_sig:
                            pending = 3
                        elif alt_sig:
                            pending = 2

    if pos_mode != 0:
        ret_pct = (closes[i_end] / (entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        mode_label = initial_mode_str + ("->TREND" if promoted_in_trade else "")
        trades.append(
            {
                "mode": mode_label,
                "entry_date": idx[entry_idx].strftime("%Y-%m-%d"),
                "entry_m": idx[entry_idx].strftime("%Y-%m"),
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(entry_px, 2),
                "avg_entry_px": round(entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    return {
        "id": "H5_E2_Promote_S1_AdaptHV",
        "name": "策略 4 双引擎旗舰: H5_E2_Promote_S1_AdaptHV",
        "short_label": "策略4 (H5_E2_Promote_S1_AdaptHV)",
        "rule_desc": "E2大底早买(SL_W25)->无缝晋升SMH200D趋势仓 | 自适应HV刹车(M_Fish<1.5容忍140%, ≥1.5限115%) | 熊市ALT快撤",
        "stats": stats,
        "trades": trades,
    }


def simulate_tp3h(
    feats: dict,
    use_f1_filter: bool = True,
    start_date: str = BT_START,
    end_date: str = BT_END,
    entry_col_override: str | None = None,
    mode_tag_override: str | None = None,
    strategy_id: str | None = None,
) -> dict:
    """Simulate upgraded Strategy 3 with proactive high-zone TP3h (with or without F1 SOXL anti-fake-bottom filter)."""
    daily_df = feats["daily"]
    w_df = feats["weekly"]
    week_end_daily_pos = feats["week_end_daily_pos"]
    idx = daily_df.index

    bt_positions = np.where((idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date)))[0]
    i_start, i_end = int(bt_positions[0]), int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    entry_col = entry_col_override or ("F1_E2_PrevHigh" if use_f1_filter else "E2_PrevHigh")
    mode_tag = mode_tag_override or ("BOT_F1" if use_f1_filter else "BOT_E2")
    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    avg_entry_px = 0.0
    entry_date_str = ""
    entry_m_str = ""
    entry_fill_pos = -1
    entry_sig_w_pos = -1
    high_zone_armed = False

    peak_d_close = 0.0
    peak_w_close = 0.0
    pending_action = ""
    pending_reason = ""
    pending_sig_w_pos = -1
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trades: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")
        m_str = idx[pos].strftime("%Y-%m")

        just_exited = False
        if pending_action == "BUY" and shares == 0:
            shares = (cash * (1.0 - FEE_RATE)) / o
            cash = 0.0
            avg_entry_px = float(o)
            entry_date_str = dt_str
            entry_m_str = m_str
            entry_fill_pos = pos
            entry_sig_w_pos = pending_sig_w_pos
            high_zone_armed = False
            peak_d_close = c
            peak_w_close = 0.0
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day buy-sell at {dt_str}")
            proceeds = shares * o * (1.0 - FEE_RATE)
            ret_pct = ((o * (1.0 - FEE_RATE)) / (avg_entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
            trades.append(
                {
                    "mode": mode_tag,
                    "entry_date": entry_date_str,
                    "entry_m": entry_m_str,
                    "tranche2_date": "-",
                    "tranche2_m": None,
                    "tranche2_px": None,
                    "entry_px": round(avg_entry_px, 2),
                    "avg_entry_px": round(avg_entry_px, 2),
                    "exit_date": dt_str,
                    "exit_m": m_str,
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "exit_reason": pending_reason,
                    "is_open": False,
                }
            )
            if pending_reason.startswith("SL_"):
                cooldown_until_pos = pos + 10
            cash = proceeds
            shares = 0.0
            avg_entry_px = 0.0
            just_exited = True
        pending_action = ""
        pending_reason = ""

        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        if pos >= i_end or pos not in week_end_daily_pos:
            continue

        w_pos = week_end_daily_pos[pos]
        w_row = w_df.iloc[w_pos]

        if shares > 0:
            wc = float(w_row["Close"])
            if wc > peak_w_close:
                peak_w_close = wc

            mf = w_row["M_Fish_prev"]
            vs30 = w_row["Close_vs_30W%"]
            unmatured = (pd.isna(mf) or float(mf) < 1.5) and (peak_d_close < avg_entry_px * 1.50)
            if unmatured and wc < avg_entry_px * 0.75:
                pending_action = "SELL"
                pending_reason = "SL_W25%"
            elif pos > entry_fill_pos:
                hold_weeks = w_pos - entry_sig_w_pos
                if hold_weeks >= 45 and pd.notna(mf) and float(mf) >= 2.0 and pd.notna(vs30) and float(vs30) >= 40.0:
                    high_zone_armed = True
                if (
                    high_zone_armed
                    and pd.notna(w_row["W_Fisher"])
                    and pd.notna(w_row["W_Fisher_Sig"])
                    and float(w_row["W_Fisher"]) < float(w_row["W_Fisher_Sig"])
                ):
                    pending_action = "SELL"
                    pending_reason = "HighStr40&WFish<Sig"
                elif pd.notna(mf) and float(mf) >= 2.0:
                    if pd.notna(vs30) and float(vs30) < 0.0:
                        pending_action = "SELL"
                        pending_reason = "M>=2&<30W0"
                    elif peak_w_close > 0.0 and wc <= peak_w_close * 0.65:
                        pending_action = "SELL"
                        pending_reason = "M>=2&WTSL35%"

        elif not just_exited and pos >= cooldown_until_pos:
            if bool(w_row[entry_col]):
                pending_action = "BUY"
                pending_sig_w_pos = w_pos

    if shares > 0:
        ret_pct = (closes[i_end] / (avg_entry_px * (1.0 + FEE_RATE)) - 1.0) * 100.0
        trades.append(
            {
                "mode": mode_tag,
                "entry_date": entry_date_str,
                "entry_m": entry_m_str,
                "tranche2_date": "-",
                "tranche2_m": None,
                "tranche2_px": None,
                "entry_px": round(avg_entry_px, 2),
                "avg_entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_m": idx[i_end].strftime("%Y-%m"),
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "exit_reason": "OPEN_END",
                "is_open": True,
            }
        )

    stats = _compute_stats(equity, trades, days_invested, idx[i_start : i_end + 1])
    if strategy_id == "C2_F1_or_MFishLin_TP3h":
        return {
            "id": "C2_F1_or_MFishLin_TP3h",
            "name": "SOXL 防踏空双保险旗舰: C2 (F1+月Fisher金叉兜底)+TP3h (15.5年$1.66亿 / 12年$9049万)",
            "short_label": "SOXL防踏空旗舰 (C2+TP3h)",
            "rule_desc": "主入口F1(200W+放量1.6x+30w|-80%+PrevHigh) 或 月线M_Fish<0.8金叉兜底 | 45w&M_Fish≥2&30W≥40%周Fisher死叉止盈 (X1a兜底)",
            "stats": stats,
            "trades": trades,
        }
    if use_f1_filter:
        return {
            "id": "F1_TP3h",
            "name": "SOXL 极致大周期止盈: F1+TP3h (5战全胜)",
            "short_label": "SOXL极致版 (F1+TP3h)",
            "rule_desc": "pre_c7_vol & (wks_52H≥30 或 8w低跌≤-80%) & PrevHigh买 | 持满45w&M_Fish≥2&乖离30W≥40%后周Fisher死叉止盈 (X1a兜底)",
            "stats": stats,
            "trades": trades,
        }
    return {
        "id": "F0_TP3h",
        "name": "通用大周期主动止盈: F0+TP3h (跨标的稳健)",
        "short_label": "通用止盈版 (F0+TP3h)",
        "rule_desc": "pre_c7_vol & PrevHigh买 + SL_W25% | 持满45w&M_Fish≥2&乖离30W≥40%后周Fisher死叉止盈 (X1a兜底)",
        "stats": stats,
        "trades": trades,
    }


def detect_c7_vol_entries(
    df_daily: pd.DataFrame,
    start_date: str = "2014-01-01",
    cluster_weeks: int = 8,
) -> list[dict]:
    """Detect causal C7_Vol major-bottom entries on weekly bars, executed at next day's Open."""
    daily = df_daily.set_index("Date")
    w = _agg_periodic(daily, "W-FRI")

    wf = fisher_transform(w["High"], w["Low"], period=10)
    w["W_Fisher"] = wf["fisher"]
    w["W_SMA200"] = sma(w["Close"], 200)
    w["W_EMA10"] = ema(w["Close"], 10)
    w["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()

    min_low_200_8w = (w["Low"] / w["W_SMA200"]).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    max_vol_4w = w["W_Vol_R"].rolling(4, min_periods=1).max()

    cond = (
        (min_low_200_8w <= 1.10)
        & (min_wf_4w < -1.5)
        & (max_vol_4w >= 1.6)
        & (w["Close"] > w["W_EMA10"])
    )

    w_dates = w.index
    d_dates = daily.index
    opens = daily["Open"].to_numpy(dtype=np.float64)
    lows = daily["Low"].to_numpy(dtype=np.float64)
    closes = daily["Close"].to_numpy(dtype=np.float64)
    n_d = len(daily)

    valid_mask = (w_dates >= start_date) & cond.fillna(False).to_numpy(dtype=bool)
    sig_positions = np.where(valid_mask)[0]

    entries: list[dict] = []
    last_ev_pos = -1000
    for pos in sig_positions:
        if (pos - last_ev_pos) <= cluster_weeks:
            continue
        last_ev_pos = pos
        w_dt = w_dates[pos]
        d_pos = int(d_dates.get_loc(w_dt))
        if d_pos + 1 >= n_d:
            continue
        entry_idx = d_pos + 1
        entry_dt = d_dates[entry_idx]
        entry_px = float(opens[entry_idx])

        w6_end = min(n_d, entry_idx + 126)
        mae_6m = (float(np.min(lows[entry_idx:w6_end])) / entry_px - 1.0) * 100.0

        def fwd_ret(bars: int) -> float | None:
            if entry_idx + bars >= n_d:
                return None
            return round((float(closes[entry_idx + bars]) / entry_px - 1.0) * 100.0, 1)

        entries.append(
            {
                "sig_week": w_dt.strftime("%Y-%m-%d"),
                "entry_date": entry_dt.strftime("%Y-%m-%d"),
                "entry_m": entry_dt.strftime("%Y-%m"),
                "entry_px": round(entry_px, 2),
                "max4w_vol_r": round(float(max_vol_4w.iloc[pos]), 2),
                "mae_6m": round(mae_6m, 1),
                "fwd_3m": fwd_ret(63),
                "fwd_6m": fwd_ret(126),
                "fwd_12m": fwd_ret(252),
            }
        )
    return entries


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SOXL 月线 LOG — 顶部与底部区域及最优策略标注</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-bg: rgba(88, 166, 255, 0.14);
      --green: #26a69a;
      --red: #ef5350;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 12px;
      padding: 10px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      font-size: 12px;
    }
    .title-badge {
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      margin-right: 4px;
    }
    .group {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .group-label {
      color: var(--muted);
      font-weight: 600;
    }
    input[type="number"] {
      width: 62px;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 4px 6px;
      font-size: 12px;
      text-align: right;
    }
    .strategy-select {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .toggle-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 10px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--bg);
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      font-size: 12px;
    }
    .toggle-pill input { display: none; }
    .toggle-pill.checked {
      border-color: var(--accent);
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
    }
    .divider {
      width: 1px;
      height: 20px;
      background: var(--border);
    }
    #chart {
      width: 100%;
      height: 68vh;
      min-height: 460px;
    }
    .table-wrap {
      padding: 12px 16px 24px;
      background: var(--bg);
      border-top: 1px solid var(--border);
      overflow-x: auto;
    }
    .table-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 6px 10px;
      border-bottom: 1px solid #21262d;
      text-align: left;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: var(--panel);
    }
    .tag-top {
      color: var(--red);
      font-weight: 600;
    }
    .tag-bot {
      color: var(--green);
      font-weight: 600;
    }
    .tag-warn {
      color: #f0b90b;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <span class="title-badge">SOXL 月线顶底区域</span>

    <label class="toggle-pill checked" id="logLabel">
      <input type="checkbox" id="logToggle" checked />
      <span>LOG 对数坐标</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showZigZag" checked />
      <span>ZigZag 连线</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showSma" checked />
      <span>30周 / 200周均线</span>
    </label>

    <label class="toggle-pill checked">
      <input type="checkbox" class="opt-check" id="showFisher" checked />
      <span>月线 Fisher(10)</span>
    </label>

    <label class="toggle-pill">
      <input type="checkbox" class="opt-check" id="showRsi" />
      <span>月线 RSI(14)</span>
    </label>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">策略标注</span>
      <select id="strategySelect" class="strategy-select">
        <option value="C2_F1_or_MFishLin_TP3h" selected>SOXL防踏空旗舰: C2 (F1+月Fisher兜底)+TP3h (12年$9049万 / 15.5年$1.66亿, 0踏空)</option>
        <option value="F1_TP3h">SOXL极致大周期: F1+TP3h (5笔全胜, 12年$9049万, 回撤-49.0%)</option>
        <option value="F0_TP3h">通用大周期止盈: F0+TP3h (6笔, 12年$6088万, 回撤-50.3%)</option>
        <option value="D1_X1a_WTSL35">策略3 极简大周期: D1_X1a_WTSL35 (6笔, 12年$3804万)</option>
        <option value="D1_X1a_WTSL35_Split50">策略3 分批建仓版: D1_X1a_WTSL35_Split50 (6笔, 12年$3012万, 回撤-54.4%)</option>
        <option value="H5_E2_Promote_S1_AdaptHV">策略4 双引擎旗舰: H5_E2_Promote_S1_AdaptHV (32笔, 12年$3791万, 回撤-52.0%)</option>
        <option value="C7_Vol">仅大底入口: C7_Vol (5次, 0误报)</option>
        <option value="NONE">不显示策略标注</option>
      </select>
    </div>

    <div class="divider"></div>

    <div class="group">
      <span class="group-label">上涨确认(%)</span>
      <input type="number" id="upPct" value="120" min="20" max="500" step="10" />
    </div>
    <div class="group">
      <span class="group-label">回撤确认(%)</span>
      <input type="number" id="downPct" value="55" min="15" max="90" step="5" />
    </div>
    <div class="group">
      <span class="group-label">区域容差(±月 / 价格%)</span>
      <input type="number" id="zoneMonths" value="2" min="0" max="5" step="1" />
      <input type="number" id="zoneTolPct" value="20" min="5" max="50" step="5" />
    </div>
  </div>

  <div id="chart"></div>

  <div class="table-wrap">
    <div class="table-title" id="summaryTitle">识别到的月线顶部与底部区域</div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>类型</th>
          <th>拐点月份</th>
          <th>区域范围</th>
          <th>极值价格 (前复权)</th>
          <th>相对上一拐点涨跌幅</th>
        </tr>
      </thead>
      <tbody id="zoneTableBody"></tbody>
    </table>

    <div id="strategyTableWrap">
      <div class="table-title" id="entryTableTitle" style="margin-top:20px;">策略交易明细</div>
      <table id="entryTable">
        <thead id="entryTableHead"></thead>
        <tbody id="entryTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const MONTHLY = __MONTHLY_JSON__;
    const WEEKLY_SMA = __WEEKLY_SMA_JSON__;
    const C7_ENTRIES = __C7_ENTRIES_JSON__;
    const STRATEGIES = __STRATEGIES_JSON__;

    function detectZigZag(bars, upTh, downTh) {
      const pivots = [];
      let trend = null;
      let lastHIdx = 0, lastH = bars[0].h;
      let lastLIdx = 0, lastL = bars[0].l;

      for (let i = 1; i < bars.length; i++) {
        const h = bars[i].h;
        const l = bars[i].l;

        if (trend === null) {
          if (h >= lastL * (1 + upTh)) {
            pivots.push({ idx: lastLIdx, type: "L", price: lastL });
            trend = "UP";
            lastHIdx = i;
            lastH = h;
          } else if (l <= lastH * (1 - downTh)) {
            pivots.push({ idx: lastHIdx, type: "H", price: lastH });
            trend = "DOWN";
            lastLIdx = i;
            lastL = l;
          } else {
            if (h > lastH) { lastHIdx = i; lastH = h; }
            if (l < lastL) { lastLIdx = i; lastL = l; }
          }
        } else if (trend === "UP") {
          if (h >= lastH) {
            lastHIdx = i;
            lastH = h;
          } else if (l <= lastH * (1 - downTh)) {
            pivots.push({ idx: lastHIdx, type: "H", price: lastH });
            trend = "DOWN";
            lastLIdx = i;
            lastL = l;
          }
        } else if (trend === "DOWN") {
          if (l <= lastL) {
            lastLIdx = i;
            lastL = l;
          } else if (h >= lastL * (1 + upTh)) {
            pivots.push({ idx: lastLIdx, type: "L", price: lastL });
            trend = "UP";
            lastHIdx = i;
            lastH = h;
          }
        }
      }
      return pivots;
    }

    function expandZones(bars, pivots, maxOffset, tolRatio) {
      return pivots.map((p, k) => {
        let start = p.idx;
        let end = p.idx;
        if (p.type === "H") {
          const minPrice = p.price * (1 - tolRatio);
          for (let i = p.idx - 1; i >= Math.max(0, p.idx - maxOffset); i--) {
            if (bars[i].h >= minPrice || bars[i].c >= minPrice) start = i;
            else break;
          }
          for (let i = p.idx + 1; i <= Math.min(bars.length - 1, p.idx + maxOffset); i++) {
            if (bars[i].h >= minPrice || bars[i].c >= minPrice) end = i;
            else break;
          }
        } else {
          const maxPrice = p.price * (1 + tolRatio);
          for (let i = p.idx - 1; i >= Math.max(0, p.idx - maxOffset); i--) {
            if (bars[i].l <= maxPrice || bars[i].c <= maxPrice) start = i;
            else break;
          }
          for (let i = p.idx + 1; i <= Math.min(bars.length - 1, p.idx + maxOffset); i++) {
            if (bars[i].l <= maxPrice || bars[i].c <= maxPrice) end = i;
            else break;
          }
        }
        const prev = k > 0 ? pivots[k - 1] : null;
        const pctChange = prev ? ((p.price - prev.price) / prev.price) * 100 : null;
        return {
          ...p,
          startIdx: start,
          endIdx: end,
          startMonth: bars[start].m,
          endMonth: bars[end].m,
          pivotMonth: bars[p.idx].m,
          pctChange,
        };
      });
    }

    function calcFisher(highs, lows, period = 10) {
      const n = highs.length;
      const fisher = new Array(n).fill(null);
      const signal = new Array(n).fill(null);
      const med = highs.map((h, i) => (h + lows[i]) / 2);
      let prevX = 0, prevF = 0;
      for (let i = period - 1; i < n; i++) {
        let minV = Infinity, maxV = -Infinity;
        for (let j = i - period + 1; j <= i; j++) {
          if (med[j] < minV) minV = med[j];
          if (med[j] > maxV) maxV = med[j];
        }
        const span = maxV - minV;
        const norm = span === 0 ? 0 : (med[i] - minV) / span - 0.5;
        let x = 0.33 * 2 * norm + 0.67 * prevX;
        if (x > 0.999) x = 0.999;
        if (x < -0.999) x = -0.999;
        const f = 0.5 * Math.log((1 + x) / (1 - x)) + 0.5 * prevF;
        fisher[i] = f;
        signal[i] = prevF;
        prevX = x;
        prevF = f;
      }
      return { fisher, signal };
    }

    function calcRSI(arr, period = 14) {
      const out = new Array(arr.length).fill(null);
      const alpha = 1 / period;
      let avgGain = 0, avgLoss = 0;
      for (let i = 1; i < arr.length; i++) {
        const delta = arr[i] - arr[i - 1];
        const gain = delta > 0 ? delta : 0;
        const loss = delta < 0 ? -delta : 0;
        if (i === 1) { avgGain = gain; avgLoss = loss; }
        else {
          avgGain = alpha * gain + (1 - alpha) * avgGain;
          avgLoss = alpha * loss + (1 - alpha) * avgLoss;
        }
        if (i >= period) {
          out[i] = avgLoss === 0 ? (avgGain > 0 ? 100 : 50) : 100 - 100 / (1 + avgGain / avgLoss);
        }
      }
      return out;
    }

    function fmtWan(val) {
      if (val >= 100000000) return `$${(val / 100000000).toFixed(2)}亿`;
      if (val >= 10000) return `$${(val / 10000).toFixed(1)}万`;
      return `$${val.toFixed(2)}`;
    }

    function addStrategyTraces(traces, stratId) {
      if (stratId === "NONE") return;

      if (stratId === "C7_Vol" && C7_ENTRIES.length > 0) {
        traces.push({
          type: "scatter",
          mode: "markers",
          name: "大底入口 (C7_Vol 次日开盘)",
          x: C7_ENTRIES.map(e => e.entry_m),
          y: C7_ENTRIES.map(e => e.entry_px),
          text: C7_ENTRIES.map(e => {
            const f6 = e.fwd_6m === null ? "—" : `${e.fwd_6m >= 0 ? "+" : ""}${e.fwd_6m.toFixed(1)}%`;
            const f12 = e.fwd_12m === null ? "—" : `${e.fwd_12m >= 0 ? "+" : ""}${e.fwd_12m.toFixed(1)}%`;
            return `大底入口 [C7_Vol]<br>信号周: ${e.sig_week}<br>买入日: ${e.entry_date} (开盘 $${e.entry_px.toFixed(2)})<br>近4周峰值量: ${e.max4w_vol_r.toFixed(2)}x<br>6月最大再跌: ${e.mae_6m.toFixed(1)}%<br>6个月收益: ${f6} | 12个月收益: ${f12}`;
          }),
          hoverinfo: "text",
          marker: { symbol: "triangle-up", size: 13, color: "#00e676", line: { color: "#0d1117", width: 1.5 } },
          yaxis: "y",
        });
        return;
      }

      const strat = STRATEGIES[stratId];
      if (!strat || !strat.trades || strat.trades.length === 0) return;

      const botBuys = [];
      const split2Buys = [];
      const trendBuys = [];
      const altBuys = [];
      const normalSells = [];
      const slSells = [];

      strat.trades.forEach((t, idx) => {
        const num = idx + 1;
        const retStr = `${t.ret_pct >= 0 ? "+" : ""}${t.ret_pct.toFixed(1)}%`;
        const avgInfo = t.tranche2_m ? ` | 均价 $${t.avg_entry_px.toFixed(2)}` : "";
        const entryTip = `#${num} 买入 [${t.mode}]<br>买入日: ${t.entry_date} (开盘 $${t.entry_px.toFixed(2)}${avgInfo})<br>对应卖出: ${t.exit_date} @ $${t.exit_px.toFixed(2)}<br>整笔收益: ${retStr} (${t.exit_reason})`;

        if (t.mode.startsWith("BOT")) {
          botBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        } else if (t.mode === "TREND") {
          trendBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        } else if (t.mode === "ALT") {
          altBuys.push({ x: t.entry_m, y: t.entry_px, text: entryTip });
        }

        if (t.tranche2_m && t.tranche2_px !== null) {
          split2Buys.push({
            x: t.tranche2_m,
            y: t.tranche2_px,
            text: `#${num} 二段补仓 (余50%仓)<br>补仓日: ${t.tranche2_date} (开盘 $${t.tranche2_px.toFixed(2)})<br>持仓均价: $${t.avg_entry_px.toFixed(2)}<br>整笔收益: ${retStr} (${t.exit_reason})`,
          });
        }

        if (!t.is_open) {
          const sellTip = `#${num} 卖出 [${t.mode}]<br>卖出日: ${t.exit_date} (开盘 $${t.exit_px.toFixed(2)})<br>入场: ${t.entry_date} (均价 $${t.avg_entry_px.toFixed(2)})<br>单笔收益: ${retStr}<br>原因: ${t.exit_reason}`;
          if (t.exit_reason.includes("SL")) {
            slSells.push({ x: t.exit_m, y: t.exit_px, text: sellTip });
          } else {
            normalSells.push({ x: t.exit_m, y: t.exit_px, text: sellTip });
          }
        }
      });

      const pushScatter = (pts, name, symbol, color, size) => {
        if (pts.length === 0) return;
        traces.push({
          type: "scatter",
          mode: "markers",
          name,
          x: pts.map(p => p.x),
          y: pts.map(p => p.y),
          text: pts.map(p => p.text),
          hoverinfo: "text",
          marker: { symbol, size, color, line: { color: "#0d1117", width: 1.5 } },
          yaxis: "y",
        });
      };

      pushScatter(botBuys, stratId.includes("Split50") ? "大底首仓买入 (50% BOT)" : "大底买入 (BOT)", "triangle-up", "#00e676", 13);
      pushScatter(split2Buys, "二段补仓 (EMA10 补满50%)", "diamond", "#38bdf8", 11);
      pushScatter(trendBuys, "趋势买入 (TREND)", "triangle-up", "#22c55e", 10);
      pushScatter(altBuys, "恐慌底买入 (ALT)", "triangle-up", "#facc15", 10);
      pushScatter(normalSells, "大顶/趋势/止盈卖出", "triangle-down", "#ff5252", 12);
      pushScatter(slSells, "止损卖出 (SL)", "x", "#fb923c", 11);
    }

    function renderStrategyTable(stratId) {
      const wrap = document.getElementById("strategyTableWrap");
      const titleEl = document.getElementById("entryTableTitle");
      const headEl = document.getElementById("entryTableHead");
      const bodyEl = document.getElementById("entryTableBody");
      const fmtPct = v => (v === null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`);

      if (stratId === "NONE") {
        wrap.style.display = "none";
        return;
      }
      wrap.style.display = "block";

      if (stratId === "C7_Vol") {
        titleEl.textContent = `大底入口明细 [C7_Vol]（近8周Low≤1.10×200W均线 & 近4周Fisher<-1.5 & 近4周峰值量≥1.6x & 站上10周EMA，共 ${C7_ENTRIES.length} 次）`;
        headEl.innerHTML = `<tr>
          <th>#</th><th>信号周</th><th>买入日 (次日开盘)</th><th>开盘价 ($)</th>
          <th>近4周峰值量</th><th>6月最大再跌 (MAE)</th><th>3个月收益</th><th>6个月收益</th><th>12个月收益</th>
        </tr>`;
        bodyEl.innerHTML = C7_ENTRIES.map((e, i) => `<tr>
          <td>${i + 1}</td><td>${e.sig_week}</td><td>${e.entry_date}</td><td>${e.entry_px.toFixed(2)}</td>
          <td>${e.max4w_vol_r.toFixed(2)}x</td><td class="tag-top">${e.mae_6m.toFixed(1)}%</td>
          <td class="${e.fwd_3m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_3m)}</td>
          <td class="${e.fwd_6m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_6m)}</td>
          <td class="${e.fwd_12m >= 0 ? "tag-bot" : "tag-top"}">${fmtPct(e.fwd_12m)}</td>
        </tr>`).join("");
        return;
      }

      const strat = STRATEGIES[stratId];
      const st = strat.stats;
      titleEl.textContent = `${strat.name}（2015–2026：$1万 → ${fmtWan(st.final)}，CAGR ${st.cagr.toFixed(2)}%，MaxDD ${st.max_dd.toFixed(2)}%，Calmar ${st.calmar.toFixed(3)}，${st.trades} 笔，胜率 ${st.win_rate.toFixed(1)}% ｜ 规则：${strat.rule_desc}）`;
      headEl.innerHTML = `<tr>
        <th>#</th><th>仓位类型</th><th>首次买入日</th><th>补仓日</th><th>买入价 ($)</th>
        <th>持仓均价 ($)</th><th>卖出日</th><th>卖出价 ($)</th><th>单笔收益</th><th>离场原因</th>
      </tr>`;
      bodyEl.innerHTML = strat.trades.map((t, i) => {
        const retCls = t.ret_pct >= 0 ? "tag-bot" : "tag-top";
        const modeCls = t.mode.startsWith("BOT") ? "tag-bot" : (t.mode === "ALT" ? "tag-warn" : "");
        return `<tr>
          <td>${i + 1}</td>
          <td class="${modeCls}">${t.mode}</td>
          <td>${t.entry_date}</td>
          <td>${t.tranche2_date}</td>
          <td>${t.entry_px.toFixed(2)}</td>
          <td>${t.avg_entry_px.toFixed(2)}</td>
          <td>${t.exit_date}</td>
          <td>${t.exit_px.toFixed(2)}</td>
          <td class="${retCls}">${fmtPct(t.ret_pct)}</td>
          <td>${t.exit_reason}</td>
        </tr>`;
      }).join("");
    }

    function render() {
      const isLog = document.getElementById("logToggle").checked;
      const showZigZag = document.getElementById("showZigZag").checked;
      const showSma = document.getElementById("showSma").checked;
      const showFisher = document.getElementById("showFisher").checked;
      const showRsi = document.getElementById("showRsi").checked;
      const stratId = document.getElementById("strategySelect").value;

      const upTh = Math.max(0.1, Number(document.getElementById("upPct").value) / 100);
      const downTh = Math.min(0.95, Math.max(0.1, Number(document.getElementById("downPct").value) / 100));
      const zoneMonths = Math.max(0, Number(document.getElementById("zoneMonths").value));
      const zoneTol = Math.max(0.01, Number(document.getElementById("zoneTolPct").value) / 100);

      const dates = MONTHLY.map(b => b.m);
      const opens = MONTHLY.map(b => b.o);
      const highs = MONTHLY.map(b => b.h);
      const lows = MONTHLY.map(b => b.l);
      const closes = MONTHLY.map(b => b.c);

      const pivots = detectZigZag(MONTHLY, upTh, downTh);
      const zones = expandZones(MONTHLY, pivots, zoneMonths, zoneTol);

      const subPanels = [];
      if (showFisher) subPanels.push("fisher");
      if (showRsi) subPanels.push("rsi");
      const subH = subPanels.length === 0 ? 0 : Math.min(0.20, 0.38 / subPanels.length);
      const gap = 0.04;
      const mainBottom = subPanels.length * (subH + gap);

      const traces = [];
      const shapes = [];
      const annotations = [];

      zones.forEach(z => {
        const isTop = z.type === "H";
        shapes.push({
          type: "rect",
          xref: "x",
          x0: z.startIdx - 0.45,
          x1: z.endIdx + 0.45,
          yref: "paper",
          y0: mainBottom,
          y1: 1,
          fillcolor: isTop ? "rgba(239, 83, 80, 0.16)" : "rgba(38, 166, 154, 0.16)",
          line: {
            color: isTop ? "rgba(239, 83, 80, 0.45)" : "rgba(38, 166, 154, 0.45)",
            width: 1,
          },
          layer: "below",
        });

        annotations.push({
          xref: "x",
          x: z.pivotMonth,
          yref: "y",
          y: isLog ? Math.log10(z.price) : z.price,
          text: isTop ? `▼ 顶 ${z.price.toFixed(2)}` : `▲ 底 ${z.price.toFixed(2)}`,
          showarrow: true,
          arrowhead: 0,
          arrowcolor: isTop ? "#ef5350" : "#26a69a",
          ax: 0,
          ay: isTop ? -22 : 22,
          font: { color: isTop ? "#ff7b72" : "#3fb950", size: 10, weight: 700 },
          bgcolor: "rgba(13,17,23,0.75)",
        });
      });

      traces.push({
        type: "candlestick",
        name: "SOXL 月线",
        x: dates,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        increasing: { line: { color: "#26a69a" }, fillcolor: "#26a69a" },
        decreasing: { line: { color: "#ef5350" }, fillcolor: "#ef5350" },
        yaxis: "y",
      });

      if (showZigZag && zones.length > 0) {
        traces.push({
          type: "scatter",
          mode: "lines+markers",
          name: "ZigZag 拐点",
          x: zones.map(z => z.pivotMonth),
          y: zones.map(z => z.price),
          line: { width: 1.5, color: "#f0b90b", dash: "dot" },
          marker: { size: 7, color: zones.map(z => (z.type === "H" ? "#ef5350" : "#26a69a")) },
          yaxis: "y",
        });
      }

      if (showSma) {
        const smaMap = new Map(WEEKLY_SMA.map(w => [w.m, w]));
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "30周均线",
          x: dates,
          y: dates.map(d => (smaMap.get(d) ? smaMap.get(d).s30 : null)),
          line: { width: 1.4, color: "#58a6ff" },
          yaxis: "y",
        });
        traces.push({
          type: "scatter",
          mode: "lines",
          name: "200周均线",
          x: dates,
          y: dates.map(d => (smaMap.get(d) ? smaMap.get(d).s200 : null)),
          line: { width: 1.5, color: "#d2a8ff" },
          yaxis: "y",
        });
      }

      addStrategyTraces(traces, stratId);

      const layout = {
        paper_bgcolor: "#0d1117",
        plot_bgcolor: "#0d1117",
        font: { color: "#e6edf3", size: 11 },
        margin: { l: 64, r: 24, t: 36, b: 36 },
        hovermode: "x unified",
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.04, bgcolor: "rgba(0,0,0,0)" },
        xaxis: {
          type: "category",
          rangeslider: { visible: false },
          gridcolor: "#21262d",
          nticks: 16,
        },
        yaxis: {
          domain: [mainBottom, 1],
          type: isLog ? "log" : "linear",
          gridcolor: "#21262d",
          title: isLog ? "SOXL 价格 (LOG)" : "SOXL 价格",
          autorange: true,
        },
        shapes,
        annotations,
      };

      subPanels.forEach((panel, idx) => {
        const axisNum = idx + 2;
        const yName = `yaxis${axisNum}`;
        const yRef = `y${axisNum}`;
        const top = mainBottom - gap - idx * (subH + gap);
        const bottom = Math.max(0, top - subH);

        if (panel === "fisher") {
          layout[yName] = { domain: [bottom, top], gridcolor: "#21262d", title: "月线 Fisher" };
          const f = calcFisher(highs, lows, 10);
          traces.push({
            type: "scatter", mode: "lines", name: "Fisher(10)",
            x: dates, y: f.fisher, line: { width: 1.4, color: "#2dd4bf" }, yaxis: yRef,
          });
          traces.push({
            type: "scatter", mode: "lines", name: "Fisher Signal",
            x: dates, y: f.signal, line: { width: 1.2, color: "#f97316", dash: "dot" }, yaxis: yRef,
          });
          layout.shapes.push(
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 0, y1: 0, line: { color: "#484f58", width: 1, dash: "dash" } }
          );
        } else if (panel === "rsi") {
          layout[yName] = { domain: [bottom, top], gridcolor: "#21262d", title: "月线 RSI(14)", range: [0, 100] };
          traces.push({
            type: "scatter", mode: "lines", name: "RSI(14)",
            x: dates, y: calcRSI(closes, 14), line: { width: 1.4, color: "#c084fc" }, yaxis: yRef,
          });
          layout.shapes.push(
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 70, y1: 70, line: { color: "#ef5350", width: 1, dash: "dot" } },
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: yRef, y0: 30, y1: 30, line: { color: "#26a69a", width: 1, dash: "dot" } }
          );
        }
      });

      Plotly.newPlot("chart", traces, layout, { responsive: true, scrollZoom: true });

      const tbody = document.getElementById("zoneTableBody");
      tbody.innerHTML = zones.map((z, i) => {
        const isTop = z.type === "H";
        const cls = isTop ? "tag-top" : "tag-bot";
        const label = isTop ? "顶部区域" : "底部区域";
        const pct = z.pctChange === null
          ? "—"
          : `${z.pctChange >= 0 ? "+" : ""}${z.pctChange.toFixed(1)}%`;
        return `<tr>
          <td>${i + 1}</td>
          <td class="${cls}">${label}</td>
          <td>${z.pivotMonth}</td>
          <td>${z.startMonth} ~ ${z.endMonth}</td>
          <td>${z.price.toFixed(2)}</td>
          <td class="${cls}">${pct}</td>
        </tr>`;
      }).join("");

      let stratInfo = "";
      if (stratId === "C7_Vol") {
        stratInfo = ` ｜ 大底入口 C7_Vol（共 ${C7_ENTRIES.length} 次触发，6月/12月胜率 100%）`;
      } else if (stratId !== "NONE" && STRATEGIES[stratId]) {
        const st = STRATEGIES[stratId].stats;
        stratInfo = ` ｜ ${STRATEGIES[stratId].short_label}（$1万→${fmtWan(st.final)}, CAGR ${st.cagr.toFixed(2)}%, MaxDD ${st.max_dd.toFixed(2)}%, ${st.trades}笔, 胜率 ${st.win_rate.toFixed(1)}%）`;
      }
      document.getElementById("summaryTitle").textContent =
        `共识别 ${zones.length} 个月线顶底区域（上涨 ≥ ${(upTh * 100).toFixed(0)}% / 回撤 ≥ ${(downTh * 100).toFixed(0)}%）${stratInfo}`;

      renderStrategyTable(stratId);
    }

    document.getElementById("logToggle").addEventListener("change", e => {
      document.getElementById("logLabel").classList.toggle("checked", e.target.checked);
      render();
    });
    document.querySelectorAll(".opt-check").forEach(el => {
      el.addEventListener("change", () => {
        el.parentElement.classList.toggle("checked", el.checked);
        render();
      });
    });
    document.getElementById("strategySelect").addEventListener("change", render);
    ["upPct", "downPct", "zoneMonths", "zoneTolPct"].forEach(id => {
      document.getElementById(id).addEventListener("change", render);
    });

    render();
  </script>
</body>
</html>
"""


def render_html(
    monthly: list[dict],
    weekly_sma: list[dict],
    c7_entries: list[dict],
    strategies: dict[str, dict],
) -> str:
    """Render standalone HTML with Plotly.js."""
    m_json = json.dumps(monthly, separators=(",", ":"), allow_nan=False)
    w_json = json.dumps(weekly_sma, separators=(",", ":"), allow_nan=False)
    c7_json = json.dumps(c7_entries, separators=(",", ":"), allow_nan=False)
    s_json = json.dumps(strategies, separators=(",", ":"), allow_nan=False, ensure_ascii=False)

    return (
        HTML_TEMPLATE.replace("__MONTHLY_JSON__", m_json)
        .replace("__WEEKLY_SMA_JSON__", w_json)
        .replace("__C7_ENTRIES_JSON__", c7_json)
        .replace("__STRATEGIES_JSON__", s_json)
    )


def main() -> None:
    df_soxl = load_daily_adjusted(DATA_PATH)
    df_smh = load_daily_adjusted(SMH_PATH)

    monthly, weekly_sma = build_monthly_and_weekly(df_soxl)
    c7_entries = detect_c7_vol_entries(df_soxl)

    feats = build_causal_features(df_soxl, df_smh)
    c2_tp3h = simulate_tp3h(
        feats,
        use_f1_filter=True,
        entry_col_override="C2_F1_or_MFishLin",
        mode_tag_override="BOT_C2",
        strategy_id="C2_F1_or_MFishLin_TP3h",
    )
    c2_15yr = simulate_tp3h(
        feats,
        use_f1_filter=True,
        start_date="2011-06-01",
        end_date=BT_END,
        entry_col_override="C2_F1_or_MFishLin",
        mode_tag_override="BOT_C2",
        strategy_id="C2_F1_or_MFishLin_TP3h",
    )
    f1_tp3h = simulate_tp3h(feats, use_f1_filter=True)
    f0_tp3h = simulate_tp3h(feats, use_f1_filter=False)
    d1_full = simulate_d1_x1a_wtsl35(feats, use_split_50_50=False)
    d1_split = simulate_d1_x1a_wtsl35(feats, use_split_50_50=True)
    h5 = simulate_h5_adapt_hv(feats)

    assert len(c7_entries) == 5, f"Expected 5 C7_Vol entries, got {len(c7_entries)}"
    assert c2_tp3h["stats"]["trades"] == 5 and abs(c2_tp3h["stats"]["final"] - 90489657.17) < 1.0, c2_tp3h["stats"]
    assert c2_15yr["stats"]["trades"] == 7 and abs(c2_15yr["stats"]["final"] - 166044243.09) < 1.0, c2_15yr["stats"]
    assert f1_tp3h["stats"]["trades"] == 5 and abs(f1_tp3h["stats"]["final"] - 90489657.17) < 1.0 and f1_tp3h["stats"]["win_rate"] == 100.0, f1_tp3h["stats"]
    assert f0_tp3h["stats"]["trades"] == 6 and abs(f0_tp3h["stats"]["final"] - 60877424.74) < 1.0 and f0_tp3h["stats"]["win_rate"] == 83.3, f0_tp3h["stats"]
    assert d1_full["stats"]["trades"] == 6 and abs(d1_full["stats"]["final"] - 38035375.82) < 1.0, d1_full["stats"]
    assert d1_split["stats"]["trades"] == 6 and abs(d1_split["stats"]["final"] - 30120328.33) < 1.0, d1_split["stats"]
    assert h5["stats"]["trades"] == 32 and abs(h5["stats"]["final"] - 37913860.48) < 1.0, h5["stats"]

    strategies = {
        "C2_F1_or_MFishLin_TP3h": c2_tp3h,
        "F1_TP3h": f1_tp3h,
        "F0_TP3h": f0_tp3h,
        "D1_X1a_WTSL35": d1_full,
        "D1_X1a_WTSL35_Split50": d1_split,
        "H5_E2_Promote_S1_AdaptHV": h5,
    }

    for key, payload in strategies.items():
        st = payload["stats"]
        print(
            f"[{key}] Final=${st['final']:,.2f} | CAGR={st['cagr']}% | MaxDD={st['max_dd']}% | Calmar={st['calmar']} | Trades={st['trades']} | WinRate={st['win_rate']}%"
        )

    html = render_html(monthly, weekly_sma, c7_entries, strategies)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML} ({len(monthly)} monthly bars)")


if __name__ == "__main__":
    main()
