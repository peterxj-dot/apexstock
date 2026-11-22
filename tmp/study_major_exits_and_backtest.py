"""Major-top exit study and strictly causal Entry x Exit full backtest for SOXL and TQQQ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from indicators import ema, fisher_transform, sma
from scan_soxl_strategies import FEE_RATE, INITIAL_CAPITAL, load_adjusted
from study_major_bottoms import build_weekly_dataset
from verify_bottom_volume import add_volume_features


def prepare_backtest_data(ticker: str, bench_ticker: str) -> dict:
    """Prepare strictly causal daily and weekly arrays for cycle entry/exit backtesting."""
    daily_df = load_adjusted(ticker)
    bench_df = load_adjusted(bench_ticker)

    w, daily_df = build_weekly_dataset(daily_df, bench_df)
    w = add_volume_features(w, daily_df, bench_df)

    w_bench, _ = build_weekly_dataset(bench_df, bench_df)
    w["Bench_vs_30W%"] = w_bench["Close_vs_30W%"].reindex(w.index)
    w["Bench_M_Fish_prev"] = w_bench["M_Fish_prev"].reindex(w.index)

    # Strictly causal completed monthly Fisher + Signal + diff aligned to weekly & daily indices
    with_dt = daily_df.assign(_dt=daily_df.index)
    m = (
        with_dt.groupby(pd.Grouper(freq="ME"))
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "_dt": "max"})
        .dropna(subset=["Open", "Close", "_dt"])
    )
    m.index = pd.DatetimeIndex(m["_dt"])

    mf = fisher_transform(m["High"], m["Low"], period=10)
    m_ind = pd.DataFrame(
        {
            "M_Fish_prev": mf["fisher"],
            "M_Fish_Sig_prev": mf["fisher_signal"],
            "M_Fish_prev_diff": mf["fisher"].diff(),
        },
        index=m.index,
    )
    m_on_w = m_ind.reindex(w.index, method="ffill")
    # Parity assertion against build_weekly_dataset
    valid_mask = w["M_Fish_prev"].notna() & m_on_w["M_Fish_prev"].notna()
    if not np.allclose(w.loc[valid_mask, "M_Fish_prev"], m_on_w.loc[valid_mask, "M_Fish_prev"], atol=1e-12):
        raise AssertionError("Monthly Fisher alignment mismatch vs build_weekly_dataset!")

    w["M_Fish_Sig_prev"] = m_on_w["M_Fish_Sig_prev"]
    w["M_Fish_prev_diff"] = m_on_w["M_Fish_prev_diff"]

    # Entry features on completed weekly bars
    min_low_200_8w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    w["pre_c7_vol"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (w["max4w_Vol_R"] >= 1.6)
    w["W_EMA5"] = ema(w["Close"], 5)
    w["prev_W_High"] = w["High"].shift(1)
    w["pre_8w_low"] = w["Low"].rolling(8, min_periods=1).min()

    w["E1_C7Vol_EMA10"] = w["pre_c7_vol"] & (w["Close"] > w["W_EMA10"])
    w["E2_PrevHigh"] = w["pre_c7_vol"] & (w["Close"] > w["prev_W_High"])
    w["E3_EMA5_FishLog"] = w["pre_c7_vol"] & (
        (w["Close"] > w["W_EMA5"]) | ((w["dW_Fish_Log"] > 0) & (w["Close"] > w["prev_W_High"]))
    )
    w["E4_EMA5"] = w["pre_c7_vol"] & (w["Close"] > w["W_EMA5"])

    # Daily aligned benchmark & causal monthly Fisher
    bench_aligned = bench_df.reindex(daily_df.index, method="ffill")
    bench_sma200 = sma(bench_aligned["Close"], 200)
    daily_df["Bench_below_SMA200_99"] = (bench_aligned["Close"] < bench_sma200 * 0.99).fillna(False)

    # Align completed monthly Fisher to daily via completed weeks -> daily? No, directly from completed months m_ind!
    # Double check: on days before final trading day of a month, ffill gives prior completed month.
    # For the final partial month in data (ending on last row daily_df.index[-1]), signal loop never trades on last row anyway.
    daily_df["M_Fish_prev_ffill"] = m_ind["M_Fish_prev"].reindex(daily_df.index, method="ffill")

    # Map every weekly last_trade_date to its exact daily index
    daily_pos_map = {dt: pos for pos, dt in enumerate(daily_df.index)}
    week_end_daily_pos: dict[int, int] = {}
    for w_pos, w_dt in enumerate(w.index):
        if w_dt not in daily_pos_map:
            raise AssertionError(f"Weekly date {w_dt} missing from daily_df.index!")
        week_end_daily_pos[daily_pos_map[w_dt]] = w_pos

    return {
        "ticker": ticker,
        "bench_ticker": bench_ticker,
        "daily": daily_df,
        "weekly": w,
        "week_end_daily_pos": week_end_daily_pos,
    }


def compute_metrics(
    equity: np.ndarray,
    trade_logs: list[dict],
    days_invested: int,
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    n = len(equity)
    final_val = float(equity[-1])
    years_span = n / 252.0
    ret_pct = (final_val / initial_capital - 1.0) * 100.0
    cagr = ((final_val / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0

    cum_max = np.maximum.accumulate(equity)
    max_dd = float(((equity / cum_max) - 1.0).min() * 100.0)
    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    daily_ret = np.diff(equity) / equity[:-1]
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252.0)) if std > 1e-12 else 0.0
    downside = np.minimum(daily_ret, 0.0)
    d_std = float(np.sqrt(np.mean(downside * downside)))
    sortino = float(daily_ret.mean() / d_std * np.sqrt(252.0)) if d_std > 1e-12 else 0.0

    trades_cnt = len(trade_logs)
    win_rate = (sum(1 for t in trade_logs if t["ret_pct"] > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = days_invested / n * 100.0

    # Yearly returns
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
        "Final": round(final_val, 2),
        "Ret%": round(ret_pct, 1),
        "CAGR": round(cagr, 2),
        "MaxDD": round(max_dd, 2),
        "Calmar": round(calmar, 3),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Trades": trades_cnt,
        "WinRate": round(win_rate, 1),
        "Exposure": round(exposure, 1),
        "WorstYear": round(worst_year, 1),
        "yearly_rets": yearly_rets,
    }


def run_buy_and_hold(
    data: dict,
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> dict:
    df = data["daily"]
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    sub = df.loc[mask]
    opens = sub["Open"].to_numpy(dtype=np.float64)
    closes = sub["Close"].to_numpy(dtype=np.float64)

    shares = (initial_capital * (1.0 - fee_rate)) / opens[0]
    equity = shares * closes
    ret = (closes[-1] / (opens[0] * (1.0 + fee_rate)) - 1.0) * 100.0
    trades = [
        {
            "entry_date": sub.index[0].strftime("%Y-%m-%d"),
            "tranche2_date": "-",
            "entry_px": round(float(opens[0]), 2),
            "exit_date": sub.index[-1].strftime("%Y-%m-%d") + "*",
            "exit_px": round(float(closes[-1]), 2),
            "ret_pct": round(float(ret), 1),
            "reason": "OPEN_B&H",
        }
    ]
    res = compute_metrics(equity, trades, len(sub), sub.index, initial_capital)
    res.update({"Entry": "Buy&Hold", "SL": "-", "Exit": "Hold", "trade_logs": trades})
    return res


def check_layer_b_exit(
    exit_rule: str,
    is_week_end: bool,
    w_row: pd.Series | None,
    d_row: pd.Series,
    peak_m_fish: float,
    peak_vs_30w: float,
    peak_w_close: float,
    peak_d_close: float,
) -> tuple[bool, str]:
    """Evaluate strictly causal Layer B major-top exit."""
    if exit_rule == "X3e_Pure_BenchSMA200":
        if bool(d_row["Bench_below_SMA200_99"]):
            return True, "Bench<SMA200*0.99"
        return False, ""

    if exit_rule == "X3f_Pure_DTSL40":
        if peak_d_close > 0 and float(d_row["Close"]) <= peak_d_close * 0.60:
            return True, "DTSL40%"
        return False, ""

    if exit_rule == "X3d_30W6_DTSL30":
        mf_ffill = d_row["M_Fish_prev_ffill"]
        if pd.notna(mf_ffill) and float(mf_ffill) >= 2.0 and peak_d_close > 0:
            if float(d_row["Close"]) <= peak_d_close * 0.70:
                return True, "M>=2&DTSL30%"
        if is_week_end and w_row is not None:
            mf_w = w_row["M_Fish_prev"]
            vs30 = w_row["Close_vs_30W%"]
            if pd.notna(mf_w) and pd.notna(vs30) and float(mf_w) >= 2.0 and float(vs30) < -6.0:
                return True, "M>=2&<30W-6%"
        return False, ""

    # All remaining Layer B rules are weekly-close evaluated
    if not is_week_end or w_row is None:
        return False, ""

    mf = w_row["M_Fish_prev"]
    mf_log = w_row["M_Fish_Log_prev"]
    mf_sig = w_row["M_Fish_Sig_prev"]
    mf_diff = w_row["M_Fish_prev_diff"]
    vs30 = w_row["Close_vs_30W%"]
    b_vs30 = w_row["Bench_vs_30W%"]
    w_close = float(w_row["Close"])
    w_ema10 = w_row["W_EMA10"]
    w_ema5 = w_row["W_EMA5"]

    if exit_rule == "X1a_M20_30W0":
        if pd.notna(mf) and pd.notna(vs30) and float(mf) >= 2.0 and float(vs30) < 0.0:
            return True, "M>=2&<30W"
    elif exit_rule == "X1b_M20_30W6":
        if pd.notna(mf) and pd.notna(vs30) and float(mf) >= 2.0 and float(vs30) < -6.0:
            return True, "M>=2&<30W-6%"
    elif exit_rule == "X1c_M20_30W8":
        if pd.notna(mf) and pd.notna(vs30) and float(mf) >= 2.0 and float(vs30) < -8.0:
            return True, "M>=2&<30W-8%"
    elif exit_rule == "X1d_M20_Bench30W":
        if pd.notna(mf) and pd.notna(b_vs30) and float(mf) >= 2.0 and float(b_vs30) < 0.0:
            return True, "M>=2&Bench<30W"
    elif exit_rule == "X1e_M20_30W6_or_Bench30W":
        if pd.notna(mf) and float(mf) >= 2.0:
            c1 = pd.notna(vs30) and float(vs30) < -6.0
            c2 = pd.notna(b_vs30) and float(b_vs30) < 0.0
            if c1 or c2:
                return True, "M>=2&(30W-6|Bench30W)"
    elif exit_rule == "X1f_MLog22_30W6":
        if pd.notna(mf_log) and pd.notna(vs30) and float(mf_log) >= 2.2 and float(vs30) < -6.0:
            return True, "MLog>=2.2&<30W-6%"
    elif exit_rule == "X2a_MFishRoll_EMA10":
        if peak_m_fish >= 2.0 and pd.notna(mf_diff) and pd.notna(w_ema10):
            if float(mf_diff) < 0.0 and w_close < float(w_ema10):
                return True, "MFishRoll&<W_EMA10"
    elif exit_rule == "X2b_MFishCross":
        if peak_m_fish >= 2.0 and pd.notna(mf) and pd.notna(mf_sig) and float(mf) < float(mf_sig):
            return True, "MFish<Sig"
    elif exit_rule == "X3a_30W6_Blowoff80":
        if pd.notna(mf) and pd.notna(vs30) and float(mf) >= 2.0 and float(vs30) < -6.0:
            return True, "M>=2&<30W-6%"
        if peak_vs_30w >= 80.0 and pd.notna(w_ema5) and w_close < float(w_ema5):
            return True, "Blowoff>=80%&<W_EMA5"
    elif exit_rule == "X3b_30W6_WTSL35":
        if pd.notna(mf) and pd.notna(vs30) and float(mf) >= 2.0 and float(vs30) < -6.0:
            return True, "M>=2&<30W-6%"
        if pd.notna(mf) and float(mf) >= 2.0 and peak_w_close > 0.0 and w_close <= peak_w_close * 0.65:
            return True, "M>=2&WTSL35%"
    elif exit_rule == "X3c_Bench30W_WTSL35":
        if pd.notna(mf) and pd.notna(b_vs30) and float(mf) >= 2.0 and float(b_vs30) < 0.0:
            return True, "M>=2&Bench<30W"
        if pd.notna(mf) and float(mf) >= 2.0 and peak_w_close > 0.0 and w_close <= peak_w_close * 0.65:
            return True, "M>=2&WTSL35%"
    else:
        raise ValueError(f"Unknown exit rule: {exit_rule}")

    return False, ""


def run_cycle_backtest(
    data: dict,
    entry_rule: str,
    sl_rule: str,
    exit_rule: str,
    start_date: str,
    end_date: str,
    sl_cooldown_days: int = 10,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> dict:
    """Run strictly causal daily simulation over [start_date, end_date]."""
    daily_df: pd.DataFrame = data["daily"]
    w_df: pd.DataFrame = data["weekly"]
    week_end_daily_pos: dict[int, int] = data["week_end_daily_pos"]

    idx = daily_df.index
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    bt_positions = np.where((idx >= start_ts) & (idx <= end_ts))[0]
    if len(bt_positions) == 0:
        raise ValueError(f"No trading days in [{start_date}, {end_date}]")
    i_start = int(bt_positions[0])
    i_end = int(bt_positions[-1])
    n_bt = i_end - i_start + 1

    opens = daily_df["Open"].to_numpy(dtype=np.float64)
    closes = daily_df["Close"].to_numpy(dtype=np.float64)

    cash = float(initial_capital)
    shares = 0.0
    tranches = 0
    avg_entry_px = 0.0
    entry_date_str = ""
    tranche2_date_str = "-"
    entry_fill_pos = -1
    entry_pre_8w_low = 0.0
    trade_cash_start = float(initial_capital)

    peak_d_close = 0.0
    peak_w_close = 0.0
    peak_m_fish = -99.0
    peak_vs_30w = -999.0
    exit_reason_pending = ""
    pending_action = ""  # "", "BUY_FULL", "BUY_HALF", "BUY_TOPUP", "SELL"
    pending_8w_low = 0.0
    cooldown_until_pos = -1

    equity = np.empty(n_bt, dtype=np.float64)
    trade_logs: list[dict] = []
    days_invested = 0

    for step, pos in enumerate(range(i_start, i_end + 1)):
        o = opens[pos]
        c = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")

        # 1. Execute pending action at today's Open
        just_filled_exit = False
        if pending_action == "BUY_FULL" and tranches == 0:
            trade_cash_start = cash
            cost = cash
            new_shares = (cost * (1.0 - fee_rate)) / o
            cash = 0.0
            shares = new_shares
            tranches = 2
            avg_entry_px = o
            entry_date_str = dt_str
            tranche2_date_str = "-"
            entry_fill_pos = pos
            entry_pre_8w_low = pending_8w_low
            peak_d_close = c
            peak_w_close = 0.0
            peak_m_fish = -99.0
            peak_vs_30w = -999.0
        elif pending_action == "BUY_HALF" and tranches == 0:
            trade_cash_start = cash
            cost = 0.5 * cash
            new_shares = (cost * (1.0 - fee_rate)) / o
            cash -= cost
            shares = new_shares
            tranches = 1
            avg_entry_px = o
            entry_date_str = dt_str
            tranche2_date_str = "-"
            entry_fill_pos = pos
            entry_pre_8w_low = pending_8w_low
            peak_d_close = c
            peak_w_close = 0.0
            peak_m_fish = -99.0
            peak_vs_30w = -999.0
        elif pending_action == "BUY_TOPUP" and tranches == 1 and shares > 0:
            cost = cash
            new_shares = (cost * (1.0 - fee_rate)) / o
            total_shares = shares + new_shares
            avg_entry_px = (shares * avg_entry_px + new_shares * o) / total_shares
            shares = total_shares
            cash = 0.0
            tranches = 2
            tranche2_date_str = dt_str
            if c > peak_d_close:
                peak_d_close = c
        elif pending_action == "SELL" and shares > 0:
            if pos <= entry_fill_pos:
                raise AssertionError(f"Same-day/invalid buy-sell order on {dt_str}")
            proceeds = shares * o * (1.0 - fee_rate)
            cash += proceeds
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trade_logs.append(
                {
                    "entry_date": entry_date_str,
                    "tranche2_date": tranche2_date_str,
                    "entry_px": round(avg_entry_px, 2),
                    "exit_date": dt_str,
                    "exit_px": round(float(o), 2),
                    "ret_pct": round(float(ret_pct), 1),
                    "reason": exit_reason_pending,
                }
            )
            if exit_reason_pending.startswith("SL_"):
                cooldown_until_pos = pos + sl_cooldown_days
            shares = 0.0
            tranches = 0
            avg_entry_px = 0.0
            just_filled_exit = True
        pending_action = ""
        exit_reason_pending = ""

        # 2. Mark-to-market at today's Close
        equity[step] = cash + shares * c
        if shares > 0:
            days_invested += 1
            if c > peak_d_close:
                peak_d_close = c

        # 3. Evaluate next-day signals (do not emit on last bar of slice)
        if pos >= i_end:
            continue

        is_w_end = pos in week_end_daily_pos
        w_row = w_df.iloc[week_end_daily_pos[pos]] if is_w_end else None
        d_row = daily_df.iloc[pos]

        if shares > 0:
            if is_w_end and w_row is not None:
                wc = float(w_row["Close"])
                if wc > peak_w_close:
                    peak_w_close = wc
                if pd.notna(w_row["M_Fish_prev"]):
                    peak_m_fish = max(peak_m_fish, float(w_row["M_Fish_prev"]))
                if pd.notna(w_row["Close_vs_30W%"]):
                    peak_vs_30w = max(peak_vs_30w, float(w_row["Close_vs_30W%"]))

            # Layer A: Failed-bottom stop (active before trade matures)
            m_fish_cur = d_row["M_Fish_prev_ffill"]
            m_unmatured = pd.isna(m_fish_cur) or (float(m_fish_cur) < 1.5)
            trade_unmatured = m_unmatured and (peak_d_close < avg_entry_px * 1.50)

            hit_sl = False
            sl_reason = ""
            if trade_unmatured and sl_rule != "SL_None":
                if sl_rule == "SL_D25" and c < avg_entry_px * 0.75:
                    hit_sl, sl_reason = True, "SL_D25%"
                elif is_w_end and w_row is not None:
                    wc = float(w_row["Close"])
                    if sl_rule == "SL_W25" and wc < avg_entry_px * 0.75:
                        hit_sl, sl_reason = True, "SL_W25%"
                    elif sl_rule == "SL_W22" and wc < avg_entry_px * 0.78:
                        hit_sl, sl_reason = True, "SL_W22%"
                    elif sl_rule == "SL_W30" and wc < avg_entry_px * 0.70:
                        hit_sl, sl_reason = True, "SL_W30%"
                    elif sl_rule == "SL_BreakLow" and wc < entry_pre_8w_low:
                        hit_sl, sl_reason = True, "SL_BreakLow"

            if hit_sl:
                pending_action = "SELL"
                exit_reason_pending = sl_reason
            else:
                # Layer B: Major-top exit (prevent exit trigger on exact entry day unless daily stop already handled)
                hit_b, b_reason = check_layer_b_exit(
                    exit_rule,
                    is_w_end and pos > entry_fill_pos,
                    w_row,
                    d_row,
                    peak_m_fish,
                    peak_vs_30w,
                    peak_w_close,
                    peak_d_close,
                )
                if hit_b:
                    pending_action = "SELL"
                    exit_reason_pending = b_reason
                elif entry_rule == "E5_Split_50_50" and tranches == 1 and is_w_end and w_row is not None:
                    if bool(w_row["E1_C7Vol_EMA10"]):
                        pending_action = "BUY_TOPUP"

        # Flat entry check (no buy on same day an exit just filled -> pos > exit fill wait? Plan says no same-day buy & sell: if exit executes at t's Open, entry cannot fill until at least t+1 -> signal at t close fills at t+1 Open, Wait: let's also keep `not just_filled_exit` on weekly signal so we never exit on open & re-enter same day's close)
        if shares == 0 and not just_filled_exit and pos >= cooldown_until_pos:
            if is_w_end and w_row is not None:
                if entry_rule == "E5_Split_50_50":
                    if bool(w_row["E1_C7Vol_EMA10"]):
                        pending_action = "BUY_FULL"
                        pending_8w_low = float(w_row["pre_8w_low"])
                    elif bool(w_row["E2_PrevHigh"]):
                        pending_action = "BUY_HALF"
                        pending_8w_low = float(w_row["pre_8w_low"])
                else:
                    if bool(w_row[entry_rule]):
                        pending_action = "BUY_FULL"
                        pending_8w_low = float(w_row["pre_8w_low"])

    # Record open trade at period end
    if shares > 0:
        final_eq = cash + shares * closes[i_end]
        ret_pct = (final_eq / trade_cash_start - 1.0) * 100.0
        trade_logs.append(
            {
                "entry_date": entry_date_str,
                "tranche2_date": tranche2_date_str,
                "entry_px": round(avg_entry_px, 2),
                "exit_date": idx[i_end].strftime("%Y-%m-%d") + "*",
                "exit_px": round(float(closes[i_end]), 2),
                "ret_pct": round(float(ret_pct), 1),
                "reason": "OPEN_END",
            }
        )

    res = compute_metrics(equity, trade_logs, days_invested, idx[i_start : i_end + 1], initial_capital)
    res.update({"Entry": entry_rule, "SL": sl_rule, "Exit": exit_rule, "trade_logs": trade_logs})
    return res


def format_summary_df(results: list[dict]) -> pd.DataFrame:
    cols = ["Entry", "SL", "Exit", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate", "Exposure", "WorstYear"]
    return pd.DataFrame([{k: r[k] for k in cols} for r in results])


def print_trade_log(title: str, res: dict) -> None:
    print(
        f"\n--- {title} | Final=${res['Final']:,.2f} | CAGR={res['CAGR']}% | MaxDD={res['MaxDD']}% | Calmar={res['Calmar']} | Trades={res['Trades']} | WinRate={res['WinRate']}% ---"
    )
    df = pd.DataFrame(res["trade_logs"])
    print(df.to_string(index=False))


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    soxl = prepare_backtest_data("SOXL", "SMH")
    tqqq = prepare_backtest_data("TQQQ", "QQQ")
    print("Causality & Monthly Fisher parity assertions PASSED on SOXL and TQQQ.")

    layer_b_exits = [
        "X1a_M20_30W0",
        "X1b_M20_30W6",
        "X1c_M20_30W8",
        "X1d_M20_Bench30W",
        "X1e_M20_30W6_or_Bench30W",
        "X1f_MLog22_30W6",
        "X2a_MFishRoll_EMA10",
        "X2b_MFishCross",
        "X3a_30W6_Blowoff80",
        "X3b_30W6_WTSL35",
        "X3c_Bench30W_WTSL35",
        "X3d_30W6_DTSL30",
        "X3e_Pure_BenchSMA200",
        "X3f_Pure_DTSL40",
    ]

    windows = [
        ("SOXL", soxl, "2015-01-01", "2025-12-31", "SOXL 2015-2025 (11yr)"),
        ("SOXL", soxl, "2015-01-01", "2026-11-19", "SOXL 2015-2026 (12yr)"),
        ("TQQQ", tqqq, "2015-01-01", "2025-12-31", "TQQQ 2015-2025 (11yr)"),
        ("TQQQ", tqqq, "2015-01-01", "2026-11-19", "TQQQ 2015-2026 (12yr)"),
    ]

    # =========================================================================
    # STAGE 1: Exit Architecture Scan (SL_W25 fixed, E1_C7Vol_EMA10 vs E2_PrevHigh)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 1: EXIT ARCHITECTURE SCAN (SL_W25 fixed; comparing E1_C7Vol_EMA10 vs E2_PrevHigh)")
    print("=" * 100)
    stage1_by_win: dict[str, pd.DataFrame] = {}
    for _, ds, s, e, label in windows:
        rows = [run_buy_and_hold(ds, s, e)]
        for ent in ("E1_C7Vol_EMA10", "E2_PrevHigh"):
            for ex in layer_b_exits:
                rows.append(run_cycle_backtest(ds, ent, "SL_W25", ex, s, e))
        df = format_summary_df(rows)
        stage1_by_win[label] = df
        print(f"\n=== {label} ===")
        print(df.to_string(index=False))

    # Pick top 4 exits dynamically based on balanced CAGR + Calmar across E1 and E2 on SOXL 2015-2026 + TQQQ 2015-2026
    s12_e2 = stage1_by_win["SOXL 2015-2026 (12yr)"].query("Entry == 'E2_PrevHigh'").copy()
    t12_e2 = stage1_by_win["TQQQ 2015-2026 (12yr)"].query("Entry == 'E2_PrevHigh'")[["Exit", "CAGR", "Calmar", "MaxDD"]]
    comb = s12_e2.merge(t12_e2, on="Exit", suffixes=("_S", "_T"))
    comb["RankScore"] = comb["Calmar_S"] * 10.0 + comb["Calmar_T"] * 10.0 + comb["CAGR_S"] * 0.25 + comb["CAGR_T"] * 0.25
    comb = comb.sort_values("RankScore", ascending=False)
    top_exits = comb["Exit"].head(4).tolist()
    for must in ("X1b_M20_30W6", "X3a_30W6_Blowoff80"):
        if must not in top_exits:
            top_exits.append(must)
    top_exits = top_exits[:5]
    print(f"\nSelected Top Exits for Stage 2/3: {top_exits}")

    # =========================================================================
    # STAGE 2: Failed-Bottom Stop Sensitivity (on E2_PrevHigh & E1_C7Vol_EMA10)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 2: FAILED-BOTTOM STOP SENSITIVITY (2022-05-31 handling & bottom safety check)")
    print("=" * 100)
    sl_rules = ["SL_None", "SL_W22", "SL_W25", "SL_W30", "SL_BreakLow", "SL_D25"]
    for label, ds, s, e in [
        ("SOXL 2015-2026 (12yr)", soxl, "2015-01-01", "2026-11-19"),
        ("TQQQ 2015-2026 (12yr)", tqqq, "2015-01-01", "2026-11-19"),
    ]:
        rows = []
        for ex in top_exits[:2]:
            for sl in sl_rules:
                rows.append(run_cycle_backtest(ds, "E2_PrevHigh", sl, ex, s, e))
        print(f"\n=== {label} | Entry=E2_PrevHigh ===")
        print(format_summary_df(rows).to_string(index=False))

    # =========================================================================
    # STAGE 3: Head-to-Head 5 Entries x Top Exits (SOXL & TQQQ, 11yr & 12yr)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 3: HEAD-TO-HEAD 5 ENTRIES x TOP EXITS (with SL_W25 vs SL_None for E1 baseline)")
    print("=" * 100)
    entries = ["E1_C7Vol_EMA10", "E2_PrevHigh", "E3_EMA5_FishLog", "E4_EMA5", "E5_Split_50_50"]
    for _, ds, s, e, label in windows:
        rows = [run_buy_and_hold(ds, s, e)]
        for ex in top_exits[:4]:
            # Check E1 with SL_None as well to be 100% fair if SL_W25 ever stops E1 (e.g. 2022-11 -32.5% MAE)
            rows.append(run_cycle_backtest(ds, "E1_C7Vol_EMA10", "SL_None", ex, s, e))
            for ent in entries:
                rows.append(run_cycle_backtest(ds, ent, "SL_W25", ex, s, e))
        print(f"\n=== {label} ===")
        print(format_summary_df(rows).to_string(index=False))

    # =========================================================================
    # STAGE 4: Full Trade-by-Trade Logs on SOXL & TQQQ (2015-2026)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STAGE 4: FULL TRADE-BY-TRADE LOGS (2015-2026)")
    print("=" * 100)
    best_exit = top_exits[0]
    alt_exit = top_exits[1]
    for ex in (best_exit, alt_exit):
        for ent, sl in [("E1_C7Vol_EMA10", "SL_None"), ("E1_C7Vol_EMA10", "SL_W25"), ("E2_PrevHigh", "SL_W25"), ("E5_Split_50_50", "SL_W25")]:
            r_soxl = run_cycle_backtest(soxl, ent, sl, ex, "2015-01-01", "2026-11-19")
            print_trade_log(f"SOXL | {ent} + {sl} + {ex}", r_soxl)
        for ent, sl in [("E1_C7Vol_EMA10", "SL_None"), ("E2_PrevHigh", "SL_W25"), ("E5_Split_50_50", "SL_W25")]:
            r_tqqq = run_cycle_backtest(tqqq, ent, sl, ex, "2015-01-01", "2026-11-19")
            print_trade_log(f"TQQQ | {ent} + {sl} + {ex}", r_tqqq)


if __name__ == "__main__":
    main()
