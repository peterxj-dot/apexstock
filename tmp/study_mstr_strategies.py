"""Explore and backtest MSTR strategies (2020-2026) using BTC-driven and MSTR-native signals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import ema, fisher_transform, historical_volatility, sma
from tools.build_soxl_zones import FEE_RATE, INITIAL_CAPITAL, load_daily_adjusted

HALVINGS = [
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-04-10"),
]


def _ds_du(dt: pd.Timestamp) -> tuple[int, int]:
    prev_h = max((h for h in HALVINGS if h <= dt), default=None)
    next_h = min((h for h in HALVINGS if h > dt), default=None)
    ds = (dt - prev_h).days if prev_h is not None else 9999
    du = (next_h - dt).days if next_h is not None else 9999
    return ds, du


def _pine_gaussian_filter(src: pd.Series, period: int, poles: int) -> pd.Series:
    beta = 0.5 * (1.0 - np.cos(2.0 * np.pi / period))
    g = src.copy()
    for _ in range(poles):
        g = g.ewm(alpha=beta, adjust=False).mean()
    return g


def compute_stats(equity: np.ndarray, trades: list[dict], days_inv: int, dates: pd.DatetimeIndex) -> dict:
    n = len(equity)
    final_val = float(equity[-1])
    years = n / 252.0
    cagr = ((final_val / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0)) * 100.0
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else 0.0
    rets = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252.0)) if (len(rets) > 0 and np.std(rets) > 0) else 0.0
    wr = (sum(1 for t in trades if t["ret_pct"] > 0.0) / len(trades) * 100.0) if trades else 0.0
    exp = days_inv / n * 100.0 if n > 0 else 0.0
    s_eq = pd.Series(equity, index=dates)
    ann = s_eq.resample("YE").last().pct_change().dropna() * 100.0
    worst_yr = float(ann.min()) if not ann.empty else 0.0
    return {
        "final": round(final_val, 2),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "win_rate": round(wr, 1),
        "exposure": round(exp, 1),
        "worst_yr": round(worst_yr, 1),
    }


def prepare_data() -> pd.DataFrame:
    df_mstr = load_daily_adjusted(ROOT / "data" / "MSTR.csv").sort_values("Date").reset_index(drop=True)
    df_btc = load_daily_adjusted(ROOT / "data" / "BTC-USD.csv").sort_values("Date").reset_index(drop=True)

    # BTC indicators on full 365d calendar
    b = df_btc.copy()
    b["b_sma150"] = sma(b["Close"], 150)
    b["b_sma200"] = sma(b["Close"], 200)
    b["b_ema150"] = ema(b["Close"], 150)
    b["b_hv20"] = historical_volatility(b["Close"], period=20, annualization_factor=365)
    b["b_hv60"] = historical_volatility(b["Close"], period=60, annualization_factor=365)
    b["b_hvr"] = b["b_hv20"] / b["b_hv60"]
    b["b_hvr_spike35"] = (b["b_hvr"] >= 1.30).rolling(35, min_periods=1).max().astype(bool)
    bg_mid = _pine_gaussian_filter(b["Close"], 10, 2)
    bg_dev = _pine_gaussian_filter((b["Close"] - bg_mid).abs(), 10, 2)
    b["b_gac_mid"] = bg_mid
    b["b_gac_upper"] = bg_mid + 1.5 * bg_dev
    b["b_close"] = b["Close"]
    # Strict causal alignment: at MSTR date t, only use BTC bars strictly before date t (or <= t-1d)
    b["Date_avail"] = b["Date"] + pd.Timedelta(days=1)

    m = df_mstr.copy()
    m["m_sma50"] = sma(m["Close"], 50)
    m["m_sma150"] = sma(m["Close"], 150)
    m["m_sma200"] = sma(m["Close"], 200)
    m["m_ema150"] = ema(m["Close"], 150)
    m["m_hv20"] = historical_volatility(m["Close"], period=20, annualization_factor=252)
    m["m_hv60"] = historical_volatility(m["Close"], period=60, annualization_factor=252)
    m["m_hvr"] = m["m_hv20"] / m["m_hv60"]
    m["m_hvr_spike25"] = (m["m_hvr"] >= 1.30).rolling(25, min_periods=1).max().astype(bool)

    mg_mid = _pine_gaussian_filter(m["Close"], 10, 2)
    mg_dev = _pine_gaussian_filter((m["Close"] - mg_mid).abs(), 10, 2)
    m["m_gac_mid"] = mg_mid
    m["m_gac_upper"] = mg_mid + 1.5 * mg_dev

    # Weekly Fisher(10) on MSTR (strictly completed W-FRI)
    mw = m.set_index("Date")
    wk = mw.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    wk_fish = fisher_transform(wk["High"], wk["Low"], period=10)
    wk["w_fish"] = wk_fish.iloc[:, 0]
    wk["w_fish_sig"] = wk_fish.iloc[:, 1]
    # Map weekly bar end-of-week strictly to next trading days
    wk_avail = wk[["w_fish", "w_fish_sig"]].copy()
    wk_avail.index = wk_avail.index + pd.Timedelta(days=1)
    m = pd.merge_asof(
        m.sort_values("Date"),
        wk_avail.reset_index().rename(columns={"Date": "wk_dt"}).sort_values("wk_dt"),
        left_on="Date",
        right_on="wk_dt",
        direction="backward",
    )

    b_cols = [
        "Date_avail", "b_close", "b_sma150", "b_sma200", "b_ema150",
        "b_hv20", "b_hvr", "b_hvr_spike35", "b_gac_mid", "b_gac_upper",
    ]
    merged = pd.merge_asof(
        m.sort_values("Date"),
        b[b_cols].sort_values("Date_avail"),
        left_on="Date",
        right_on="Date_avail",
        direction="backward",
    )
    return merged


def run_strategy(df: pd.DataFrame, mode: str, start_date: str = "2020-01-02") -> tuple[dict, list[dict]]:
    d = df.set_index("Date")
    idx = d.index
    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = len(idx) - 1
    n_bt = i_end - i_start + 1

    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    b_close = d["b_close"].to_numpy(dtype=np.float64)
    b_sma150 = d["b_sma150"].to_numpy(dtype=np.float64)
    b_sma200 = d["b_sma200"].to_numpy(dtype=np.float64)
    b_ema150 = d["b_ema150"].to_numpy(dtype=np.float64)
    b_hv20 = d["b_hv20"].to_numpy(dtype=np.float64)
    b_hvr = d["b_hvr"].to_numpy(dtype=np.float64)
    b_sp35 = d["b_hvr_spike35"].to_numpy(dtype=bool)
    b_gac_m = d["b_gac_mid"].to_numpy(dtype=np.float64)
    b_gac_u = d["b_gac_upper"].to_numpy(dtype=np.float64)

    m_sma50 = d["m_sma50"].to_numpy(dtype=np.float64)
    m_sma200 = d["m_sma200"].to_numpy(dtype=np.float64)
    m_ema150 = d["m_ema150"].to_numpy(dtype=np.float64)
    m_hv20 = d["m_hv20"].to_numpy(dtype=np.float64)
    m_hvr = d["m_hvr"].to_numpy(dtype=np.float64)
    m_sp25 = d["m_hvr_spike25"].to_numpy(dtype=bool)
    m_gac_m = d["m_gac_mid"].to_numpy(dtype=np.float64)
    m_gac_u = d["m_gac_upper"].to_numpy(dtype=np.float64)
    w_fish = d["w_fish"].to_numpy(dtype=np.float64)
    w_fish_sig = d["w_fish_sig"].to_numpy(dtype=np.float64)

    cash = INITIAL_CAPITAL
    shares = 0.0
    trade_cash_start = INITIAL_CAPITAL
    entry_px = 0.0
    entry_dt = ""
    pending = ""
    pending_reason = ""
    days_inv = 0
    trades: list[dict] = []
    equity = np.empty(n_bt, dtype=np.float64)

    peak_close = 0.0
    vol_brake_exit = False

    def eval_sig(pos: int, holding: bool) -> tuple[str, str]:
        nonlocal peak_close, vol_brake_exit
        ds, du = _ds_du(idx[pos])
        c = closes[pos]
        bc = b_close[pos]

        if mode == "BUY_HOLD":
            if not holding and pos == i_start:
                return "BUY", "BH"
            return "HOLD", ""

        if mode == "BTC_Halv_HV_Combo":
            in_buy = (du <= 540) and (ds > 560)
            in_sell = (ds >= 480) and (du > 540)
            if not holding:
                if in_buy:
                    if b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0:
                        return "BUY", "BTC_HV_Cool"
                    if du <= 490:
                        return "BUY", "H-490_Fallback"
                elif ds < 480:
                    return "BUY", "Halv_Bull"
            else:
                if in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30:
                    return "SELL", "BTC_HV_Spike"
                if ds >= 540 and du > 540:
                    return "SELL", "H+540_Fallback"

        elif mode == "BTC_Phase_Halv_SMA150":
            if not holding:
                if (du <= 520 and ds > 535) or (ds <= 200):
                    return "BUY", "Accum"
                if (200 < ds <= 535) and not np.isnan(b_sma150[pos]) and bc > b_sma150[pos] and b_hv20[pos] <= 95.0:
                    return "BUY", "BTC_Trend"
            else:
                if ds > 535 and du > 520:
                    return "SELL", "Bear"
                if 200 < ds <= 250 and ((not np.isnan(b_sma150[pos]) and bc < b_sma150[pos] * 0.99) or b_hv20[pos] > 105.0):
                    return "SELL", "MidExit"
                if 250 < ds <= 535 and (not np.isnan(b_sma150[pos]) and bc < b_sma150[pos] * 0.95):
                    return "SELL", "LateSMA150"

        elif mode == "BTC_Daily_GAC_10_2_15":
            in_bull = (du <= 500) or (ds <= 530)
            if not holding:
                if in_bull and not np.isnan(b_gac_u[pos]) and bc > b_gac_u[pos]:
                    return "BUY", "BTC_GAC_Upper"
            else:
                if (not in_bull) or (not np.isnan(b_gac_m[pos]) and bc < b_gac_m[pos]):
                    return "SELL", "BTC_GAC_Mid/Bear"

        elif mode == "MSTR_Halv_GAC_10_2_15":
            in_bull = (du <= 500) or (ds <= 530)
            if not holding:
                if in_bull and not np.isnan(m_gac_u[pos]) and c > m_gac_u[pos]:
                    return "BUY", "MSTR_GAC_Upper"
            else:
                if (not in_bull) or (not np.isnan(m_gac_m[pos]) and c < m_gac_m[pos]):
                    return "SELL", "MSTR_GAC_Mid/Bear"

        elif mode == "MSTR_GAC_Always":
            if not holding:
                if not np.isnan(m_gac_u[pos]) and c > m_gac_u[pos]:
                    return "BUY", "MSTR_GAC_Upper"
            else:
                if not np.isnan(m_gac_m[pos]) and c < m_gac_m[pos]:
                    return "SELL", "MSTR_GAC_Mid"

        elif mode == "BTC_SMA200_MSTR":
            if not holding:
                if not np.isnan(b_sma200[pos]) and bc > b_sma200[pos] * 1.01:
                    return "BUY", "BTC>SMA200"
            else:
                if not np.isnan(b_sma200[pos]) and bc < b_sma200[pos] * 0.99:
                    return "SELL", "BTC<SMA200"

        elif mode == "BTC_SMA150_MSTR":
            if not holding:
                if not np.isnan(b_sma150[pos]) and bc > b_sma150[pos] * 1.01:
                    return "BUY", "BTC>SMA150"
            else:
                if not np.isnan(b_sma150[pos]) and bc < b_sma150[pos] * 0.99:
                    return "SELL", "BTC<SMA150"

        elif mode == "BTC_EMA150_MSTR_TSL30":
            if not holding:
                if not np.isnan(b_ema150[pos]) and bc > b_ema150[pos] * 1.02:
                    return "BUY", "BTC>EMA150"
            else:
                if (not np.isnan(b_ema150[pos]) and bc < b_ema150[pos] * 0.98) or (c < peak_close * 0.70):
                    return "SELL", "BTC_EMA150_or_TSL30"

        elif mode == "MSTR_SMA200_HV115":
            hv = m_hv20[pos]
            hv_lim = 95.0 if vol_brake_exit else 105.0
            if not holding:
                if not np.isnan(m_sma200[pos]) and c > m_sma200[pos] * 1.01 and hv <= hv_lim:
                    return "BUY", "MSTR>SMA200"
            else:
                if hv > 125.0:
                    vol_brake_exit = True
                    return "SELL", "MSTR_HV>125"
                if not np.isnan(m_sma200[pos]) and c < m_sma200[pos] * 0.99:
                    vol_brake_exit = False
                    return "SELL", "MSTR<SMA200"

        elif mode == "MSTR_EMA150_TSL30":
            if not holding:
                if not np.isnan(m_ema150[pos]) and c > m_ema150[pos] * 1.02:
                    return "BUY", "MSTR>EMA150"
            else:
                if (not np.isnan(m_ema150[pos]) and c < m_ema150[pos] * 0.98) or (c < peak_close * 0.70):
                    return "SELL", "MSTR_EMA150_or_TSL30"

        elif mode == "MSTR_Halv_Combo_TSL35":
            # Halving buy zone (using BTC HV cool or H-490), exit on MSTR TSL35% after H+200 or Halv sell zone
            in_buy = (du <= 540) and (ds > 560)
            in_sell = (ds >= 480) and (du > 540)
            if not holding:
                if in_buy:
                    if b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0:
                        return "BUY", "BTC_HV_Cool"
                    if du <= 490:
                        return "BUY", "H-490_Fallback"
                elif ds < 480:
                    return "BUY", "Halv_Bull"
            else:
                if ds >= 200 and c < peak_close * 0.65:
                    return "SELL", "MSTR_TSL35"
                if in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30:
                    return "SELL", "BTC_HV_Spike"
                if ds >= 540 and du > 540:
                    return "SELL", "H+540_Fallback"

        elif mode == "MSTR_Halv_Combo_MSTR_SMA50":
            # Buy in Halv buy zone (or re-enter if MSTR > GAC_upper inside bull), exit if ds>=250 & MSTR < SMA50 or Bear
            in_buy = (du <= 540) and (ds > 560)
            in_bull = (du <= 500) or (ds <= 530)
            if not holding:
                if in_buy and ((b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0) or du <= 490):
                    return "BUY", "Halv_Bottom"
                if in_bull and ds >= 0 and ds <= 400 and not np.isnan(m_gac_u[pos]) and c > m_gac_u[pos]:
                    return "BUY", "Bull_GAC_Reentry"
            else:
                if ds > 535 and du > 540:
                    return "SELL", "Bear"
                if ds >= 200 and not np.isnan(m_sma50[pos]) and c < m_sma50[pos] * 0.98:
                    return "SELL", "Late_MSTR<SMA50"

        return "HOLD", ""

    for k, pos in enumerate(range(i_start, i_end + 1)):
        o_px = opens[pos]
        c_px = closes[pos]
        dt_str = idx[pos].strftime("%Y-%m-%d")

        if pending == "BUY" and shares == 0.0:
            trade_cash_start = cash
            shares = (cash * (1.0 - FEE_RATE)) / o_px
            cash = 0.0
            entry_px = o_px
            entry_dt = dt_str
            peak_close = c_px
            pending = ""
        elif pending == "SELL" and shares > 0.0:
            cash = shares * o_px * (1.0 - FEE_RATE)
            ret_pct = (cash / trade_cash_start - 1.0) * 100.0
            trades.append({
                "entry_date": entry_dt, "entry_px": round(entry_px, 2),
                "exit_date": dt_str, "exit_px": round(o_px, 2),
                "ret_pct": round(ret_pct, 1), "reason": pending_reason,
            })
            shares = 0.0
            entry_px = 0.0
            peak_close = 0.0
            pending = ""

        if shares > 0.0:
            days_inv += 1
            if c_px > peak_close:
                peak_close = c_px

        equity[k] = cash + shares * c_px

        if pos < i_end:
            act, rsn = eval_sig(pos, shares > 0.0)
            if act in ("BUY", "SELL"):
                pending = act
                pending_reason = rsn

    if shares > 0.0:
        ret_pct = (shares * closes[i_end] / trade_cash_start - 1.0) * 100.0
        trades.append({
            "entry_date": entry_dt, "entry_px": round(entry_px, 2),
            "exit_date": idx[i_end].strftime("%Y-%m-%d") + "(OPEN)", "exit_px": round(closes[i_end], 2),
            "ret_pct": round(ret_pct, 1), "reason": "OPEN",
        })

    stats = compute_stats(equity, trades, days_inv, idx[i_start:i_end + 1])
    return stats, trades


def main() -> None:
    df = prepare_data()
    modes = [
        "BUY_HOLD",
        "BTC_Halv_HV_Combo",
        "BTC_Phase_Halv_SMA150",
        "BTC_Daily_GAC_10_2_15",
        "MSTR_Halv_GAC_10_2_15",
        "MSTR_GAC_Always",
        "BTC_SMA150_MSTR",
        "BTC_SMA200_MSTR",
        "BTC_EMA150_MSTR_TSL30",
        "MSTR_SMA200_HV115",
        "MSTR_EMA150_TSL30",
        "MSTR_Halv_Combo_TSL35",
        "MSTR_Halv_Combo_MSTR_SMA50",
    ]
    print("=== MSTR Backtest (2020-01-02 ~ 2026-11-23, Initial $10,000, 0.05% fee) ===")
    results = []
    for m in modes:
        st, tr = run_strategy(df, m, start_date="2020-01-02")
        results.append((st["final"], m, st, tr))
    results.sort(reverse=True)
    for _, m, st, _ in results:
        print(
            f"{m:26s} | Final=${st['final']:12,.2f} | CAGR={st['cagr']:6.2f}% | "
            f"MaxDD={st['max_dd']:6.2f}% | Calmar={st['calmar']:5.3f} | Sharpe={st['sharpe']:4.2f} | "
            f"Trades={st['trades']:2d} | WR={st['win_rate']:5.1f}% | Exp={st['exposure']:5.1f}% | WorstYr={st['worst_yr']:6.1f}%"
        )

    print("\n--- Top 3 Strategy Trades ---")
    for _, m, _, tr in results[:4]:
        if m == "BUY_HOLD":
            continue
        print(f"\n[{m}]")
        for t in tr:
            print(f"  {t['entry_date']} @ ${t['entry_px']} -> {t['exit_date']} @ ${t['exit_px']} ({t['ret_pct']:+.1f}%) [{t['reason']}]")


if __name__ == "__main__":
    main()
