"""Large-scale strictly causal SOXL strategy & parameter scanner (2015-2025)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import atr, ema, fisher_transform, historical_volatility, sma

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0005


def load_adjusted(ticker: str) -> pd.DataFrame:
    """Load daily CSV, drop incomplete rows, and split/dividend-adjust OHLC."""
    df = pd.read_csv(ROOT / "data" / f"{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if "Adj Close" in df.columns:
        ratio = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * ratio
    return df.set_index("Date")


def build_periodic(daily_df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Aggregate daily data to causal period bars indexed by each period's actual last trade date."""
    with_dt = daily_df.assign(_dt=daily_df.index)
    grouped = with_dt.groupby(pd.Grouper(freq=freq)).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "_dt": "max",
        }
    )
    grouped = grouped.dropna(subset=["Open", "Close", "_dt"])
    last_dates = pd.DatetimeIndex(grouped["_dt"])
    periodic = grouped.drop(columns=["_dt"]).copy()
    periodic.index = last_dates
    return periodic, last_dates


def compute_asset_features(daily_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> dict:
    """Precompute daily, weekly, and monthly causal feature arrays aligned to target_index."""
    d = daily_df.reindex(target_index, method="ffill").copy()

    # Daily indicators
    feats: dict[str, pd.Series | pd.DataFrame] = {"daily": d}
    for n in (10, 15, 20, 30, 50, 80, 100, 120, 150, 200):
        feats[f"D_SMA{n}"] = sma(d["Close"], n)
        feats[f"D_EMA{n}"] = ema(d["Close"], n)

    df_d_fish = fisher_transform(d["High"], d["Low"], period=10)
    feats["D_Fisher"] = df_d_fish["fisher"]
    feats["D_Fisher_Sig"] = df_d_fish["fisher_signal"]
    feats["D_ATR14"] = atr(d["High"], d["Low"], d["Close"], period=14)
    feats["D_HV20"] = historical_volatility(d["Close"], period=20, annualization_factor=252)

    # Weekly causal indicators
    w, w_last_dates = build_periodic(daily_df, "W-FRI")
    wf = fisher_transform(w["High"], w["Low"], period=10)
    w_ind = pd.DataFrame(index=w.index)
    w_ind["W_Close"] = w["Close"]
    w_ind["W_Fisher"] = wf["fisher"]
    w_ind["W_Fisher_Sig"] = wf["fisher_signal"]
    w_ind["dW_Fisher"] = w_ind["W_Fisher"].diff()
    w_ind["W_Vol_R"] = w["Volume"] / w["Volume"].rolling(20, min_periods=10).mean()
    for n in (15, 20, 30, 40):
        w_ind[f"W_SMA{n}"] = sma(w["Close"], n)

    w_aligned = w_ind.reindex(target_index, method="ffill")
    for col in w_aligned.columns:
        feats[col] = w_aligned[col]
    feats["is_week_end"] = pd.Series(target_index.isin(w_last_dates), index=target_index)

    # Monthly causal indicators
    m, _ = build_periodic(daily_df, "ME")
    mf = fisher_transform(m["High"], m["Low"], period=10)
    m_ind = pd.DataFrame(index=m.index)
    m_ind["M_Fisher"] = mf["fisher"]
    m_ind["M_Vol_R"] = m["Volume"] / m["Volume"].rolling(12, min_periods=6).mean()

    m_aligned = m_ind.reindex(target_index, method="ffill")
    for col in m_aligned.columns:
        feats[col] = m_aligned[col]

    return feats


def simulate_signals(
    opens: np.ndarray,
    closes: np.ndarray,
    atr14: np.ndarray,
    buy_sig: np.ndarray,
    sell_sig: np.ndarray,
    year_end_indices: list[tuple[int, int]],
    fixed_sl: float | None = None,
    trail_sl: float | None = None,
    atr_sl_mult: float | None = None,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
    fee_rate: float = FEE_RATE,
    initial_capital: float = INITIAL_CAPITAL,
    record_details: bool = False,
) -> dict:
    """Fast strictly causal simulation: signal at day i Close executes at day i+1 Open."""
    n = len(closes)
    cash = initial_capital
    shares = 0.0
    in_pos = False
    pending = 0  # 0=none, 1=BUY, -1=SELL
    entry_px = 0.0
    entry_idx = -100000
    exit_idx = -100000
    peak_close = 0.0

    equity = np.empty(n, dtype=np.float64)
    trade_rets: list[float] = []
    trade_logs: list[dict] = []
    days_invested = 0

    for i in range(n):
        o = opens[i]
        c = closes[i]

        # 1. Execute pending order at today's Open
        if pending == 1 and not in_pos:
            shares = (cash * (1.0 - fee_rate)) / o
            cash = 0.0
            in_pos = True
            entry_px = o
            entry_idx = i
            peak_close = c
        elif pending == -1 and in_pos:
            proceeds = shares * o * (1.0 - fee_rate)
            ret = (o * (1.0 - fee_rate)) / (entry_px * (1.0 + fee_rate)) - 1.0
            trade_rets.append(ret)
            if record_details:
                trade_logs.append({"entry_idx": entry_idx, "exit_idx": i, "entry_px": entry_px, "exit_px": o, "ret_pct": ret * 100.0})
            cash = proceeds
            shares = 0.0
            in_pos = False
            exit_idx = i
        pending = 0

        # 2. Mark-to-market at today's Close
        if in_pos:
            days_invested += 1
            equity[i] = shares * c
            if c > peak_close:
                peak_close = c

            # 3. Evaluate stops first (override min_hold_days), then signal exit
            stop_hit = False
            if fixed_sl is not None and c <= entry_px * (1.0 - fixed_sl):
                stop_hit = True
            elif trail_sl is not None and c <= peak_close * (1.0 - trail_sl):
                stop_hit = True
            elif atr_sl_mult is not None and not np.isnan(atr14[i]) and c <= peak_close - atr_sl_mult * atr14[i]:
                stop_hit = True

            if stop_hit or ((i - entry_idx) >= min_hold_days and sell_sig[i]):
                pending = -1
        else:
            equity[i] = cash
            if (i - exit_idx) >= cooldown_days and buy_sig[i]:
                pending = 1

    # Include open trade at period end for trade count / win rate consistency
    if in_pos:
        ret = closes[-1] / (entry_px * (1.0 + fee_rate)) - 1.0
        trade_rets.append(ret)
        if record_details:
            trade_logs.append({"entry_idx": entry_idx, "exit_idx": n - 1, "entry_px": entry_px, "exit_px": closes[-1], "ret_pct": ret * 100.0, "is_open": True})

    final_val = float(equity[-1])
    years_span = n / 252.0
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

    trades_cnt = len(trade_rets)
    win_rate = (sum(1 for r in trade_rets if r > 0.0) / trades_cnt * 100.0) if trades_cnt else 0.0
    exposure = days_invested / n * 100.0

    # Yearly returns
    yearly_rets: dict[int, float] = {}
    prev_eq = initial_capital
    for yr, end_idx in year_end_indices:
        yr_eq = float(equity[end_idx])
        yearly_rets[yr] = (yr_eq / prev_eq - 1.0) * 100.0
        prev_eq = yr_eq
    worst_year = min(yearly_rets.values()) if yearly_rets else 0.0

    res = {
        "Final": round(final_val, 2),
        "CAGR": round(cagr, 2),
        "MaxDD": round(max_dd, 2),
        "Calmar": round(calmar, 3),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Trades": trades_cnt,
        "WinRate": round(win_rate, 1),
        "Exposure": round(exposure, 1),
        "WorstYear": round(worst_year, 1),
    }
    if record_details:
        res["yearly_rets"] = yearly_rets
        res["trade_logs"] = trade_logs
    return res


def main() -> None:
    t0 = time.perf_counter()
    soxl = load_adjusted("SOXL")
    smh = load_adjusted("SMH")
    qqq = load_adjusted("QQQ")

    full_idx = soxl.index
    soxl_f = compute_asset_features(soxl, full_idx)
    smh_f = compute_asset_features(smh, full_idx)
    qqq_f = compute_asset_features(qqq, full_idx)

    # Backtest slice mask
    bt_mask = (full_idx >= START_DATE) & (full_idx <= END_DATE)
    bt_idx = full_idx[bt_mask]
    n_bt = len(bt_idx)

    def sl(series_or_arr) -> np.ndarray:
        if isinstance(series_or_arr, pd.Series):
            return series_or_arr.to_numpy()[bt_mask]
        return np.asarray(series_or_arr)[bt_mask]

    opens = sl(soxl_f["daily"]["Open"]).astype(np.float64)
    closes = sl(soxl_f["daily"]["Close"]).astype(np.float64)
    atr14 = sl(soxl_f["D_ATR14"]).astype(np.float64)
    is_w_end = sl(soxl_f["is_week_end"]).astype(bool)

    # Calendar year end indices inside bt_idx
    yr_series = pd.Series(np.arange(n_bt), index=bt_idx)
    year_end_indices = [(int(yr), int(grp.iloc[-1])) for yr, grp in yr_series.groupby(bt_idx.year)]

    # Benchmarks
    bh_soxl_eq = INITIAL_CAPITAL * (closes / closes[0])
    years_span = n_bt / 252.0
    bh_soxl_cagr = ((bh_soxl_eq[-1] / INITIAL_CAPITAL) ** (1.0 / years_span) - 1.0) * 100.0
    bh_soxl_dd = ((bh_soxl_eq / np.maximum.accumulate(bh_soxl_eq)) - 1.0).min() * 100.0
    bh_soxl_dr = np.diff(bh_soxl_eq) / bh_soxl_eq[:-1]
    bh_soxl_sharpe = float(bh_soxl_dr.mean() / bh_soxl_dr.std() * np.sqrt(252.0))

    smh_closes = sl(smh_f["daily"]["Close"]).astype(np.float64)
    bh_smh_eq = INITIAL_CAPITAL * (smh_closes / smh_closes[0])
    bh_smh_cagr = ((bh_smh_eq[-1] / INITIAL_CAPITAL) ** (1.0 / years_span) - 1.0) * 100.0
    bh_smh_dd = ((bh_smh_eq / np.maximum.accumulate(bh_smh_eq)) - 1.0).min() * 100.0
    bh_smh_dr = np.diff(bh_smh_eq) / bh_smh_eq[:-1]
    bh_smh_sharpe = float(bh_smh_dr.mean() / bh_smh_dr.std() * np.sqrt(252.0))

    print("=== Benchmarks (2015-01-01 ~ 2025-12-31, Initial=$10,000) ===")
    print(f"Buy & Hold SOXL -> Final: ${bh_soxl_eq[-1]:,.2f}, CAGR: {bh_soxl_cagr:.2f}%, MaxDD: {bh_soxl_dd:.2f}%, Calmar: {bh_soxl_cagr/abs(bh_soxl_dd):.3f}, Sharpe: {bh_soxl_sharpe:.2f}")
    print(f"Buy & Hold SMH  -> Final: ${bh_smh_eq[-1]:,.2f}, CAGR: {bh_smh_cagr:.2f}%, MaxDD: {bh_smh_dd:.2f}%, Calmar: {bh_smh_cagr/abs(bh_smh_dd):.3f}, Sharpe: {bh_smh_sharpe:.2f}")

    # Pre-extract arrays used across families
    soxl_wc = sl(soxl_f["W_Close"])
    soxl_wf = sl(soxl_f["W_Fisher"])
    soxl_wfs = sl(soxl_f["W_Fisher_Sig"])
    soxl_dwf = sl(soxl_f["dW_Fisher"])
    soxl_wvr = sl(soxl_f["W_Vol_R"])
    soxl_mf = sl(soxl_f["M_Fisher"])
    soxl_hv20 = sl(soxl_f["D_HV20"])
    soxl_df = sl(soxl_f["D_Fisher"])
    soxl_dfs = sl(soxl_f["D_Fisher_Sig"])

    smh_dc = sl(smh_f["daily"]["Close"])
    smh_wc = sl(smh_f["W_Close"])
    smh_wf = sl(smh_f["W_Fisher"])
    smh_wfs = sl(smh_f["W_Fisher_Sig"])
    smh_df = sl(smh_f["D_Fisher"])
    smh_dfs = sl(smh_f["D_Fisher_Sig"])

    qqq_dc = sl(qqq_f["daily"]["Close"])

    results: list[dict] = []
    # Store generator specs so we can replay top strategies with full details
    replay_specs: dict[str, tuple] = {}

    def record(family: str, name: str, params_str: str, buy_sig: np.ndarray, sell_sig: np.ndarray, sim_kw: dict) -> None:
        m = simulate_signals(opens, closes, atr14, buy_sig, sell_sig, year_end_indices, **sim_kw)
        row = {"Family": family, "Name": name, "Params": params_str, **m}
        results.append(row)
        key = f"{family}|{name}|{params_str}"
        replay_specs[key] = (buy_sig, sell_sig, sim_kw)

    # -------------------------------------------------------------------------
    # Family 1: F1_FisherVol (Weekly/Monthly Fisher + Volume + SMA + Stops)
    # -------------------------------------------------------------------------
    for sma_w in (20, 30):
        w_sma = sl(soxl_f[f"W_SMA{sma_w}"])
        for allow_bot in (False, True):
            b_bot = (soxl_wf < -1.5) & ((soxl_dwf > 0) | (soxl_wvr > 1.3)) if allow_bot else np.zeros(n_bt, dtype=bool)
            b_trend = (soxl_wc > w_sma) & (soxl_wf > soxl_wfs)
            buy_sig = is_w_end & (b_bot | b_trend)

            for mf_sell in (2.0, 2.2, 99.0):
                for s3_buf in (0.95, 1.00):
                    s_top = ((soxl_mf > 2.0) & (soxl_dwf < 0) & (soxl_wc < w_sma * 1.05)) | (
                        (soxl_mf > mf_sell) & (soxl_dwf < -0.25)
                    )
                    s_trend = (soxl_wc < w_sma * s3_buf) & (soxl_wf < soxl_wfs) & (soxl_wf < -0.3)
                    sell_sig = is_w_end & (s_top | s_trend)

                    for tsl, fsl in ((None, None), (0.20, None), (0.25, None), (0.30, None), (None, 0.15), (None, 0.20)):
                        for mh in (0, 10, 20):
                            for cd in (0, 15, 30):
                                pstr = f"W_SMA{sma_w},bot={allow_bot},mf={mf_sell},buf={s3_buf},tsl={tsl},fsl={fsl},mh={mh},cd={cd}"
                                record("F1_FisherVol", "FisherVol_Trend", pstr, buy_sig, sell_sig, {
                                    "trail_sl": tsl, "fixed_sl": fsl, "min_hold_days": mh, "cooldown_days": cd
                                })

    # -------------------------------------------------------------------------
    # Family 2: F2_SMH_Driver (1x SMH Daily & Weekly Trend Driving SOXL)
    # -------------------------------------------------------------------------
    for ma_type in ("SMA", "EMA"):
        for n in (50, 80, 100, 120, 150, 200):
            ma_arr = sl(smh_f[f"D_{ma_type}{n}"])
            for buf in (0.0, 0.01, 0.02):
                b_base = smh_dc > ma_arr * (1.0 + buf)
                s_base = smh_dc < ma_arr * (1.0 - buf)
                for vol_cap in (None, 95, 115):
                    if vol_cap is None:
                        b_sig, s_sig = b_base, s_base
                    else:
                        b_sig = b_base & (soxl_hv20 <= vol_cap * 0.92)
                        s_sig = s_base | (soxl_hv20 > vol_cap)
                    for tsl in (None, 0.20, 0.25, 0.30):
                        for mh, cd in ((0, 0), (5, 10), (10, 10), (10, 20), (20, 20)):
                            pstr = f"SMH_D_{ma_type}{n},buf={buf},vcap={vol_cap},tsl={tsl},mh={mh},cd={cd}"
                            record("F2_SMH_Driver", "SMH_DailyMA", pstr, b_sig, s_sig, {
                                "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd
                            })

    for w_n in (15, 20, 30, 40):
        smh_ws = sl(smh_f[f"W_SMA{w_n}"])
        for use_fish in (False, True):
            if use_fish:
                b_sig = is_w_end & (smh_wc > smh_ws) & (smh_wf > smh_wfs)
                s_sig = is_w_end & ((smh_wc < smh_ws) | ((smh_wf < smh_wfs) & (smh_wf < 0.0)))
            else:
                b_sig = is_w_end & (smh_wc > smh_ws)
                s_sig = is_w_end & (smh_wc < smh_ws)
            for tsl in (None, 0.20, 0.25, 0.30):
                for mh in (0, 10, 20):
                    for cd in (0, 10, 20):
                        pstr = f"SMH_W_SMA{w_n},fish={use_fish},tsl={tsl},mh={mh},cd={cd}"
                        record("F2_SMH_Driver", "SMH_WeeklyMA", pstr, b_sig, s_sig, {
                            "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd
                        })

    # -------------------------------------------------------------------------
    # Family 3: F3_DualMA_VolBrake (Dual EMA Crossover + Volatility Brake)
    # -------------------------------------------------------------------------
    for src_name, src_f in (("SOXL", soxl_f), ("SMH", smh_f)):
        for fast_n, slow_n in ((10, 50), (15, 80), (20, 100), (20, 150), (30, 150), (50, 200)):
            fast_arr = sl(src_f[f"D_EMA{fast_n}"])
            slow_arr = sl(src_f[f"D_EMA{slow_n}"])
            b_cross = fast_arr > slow_arr
            s_cross = fast_arr < slow_arr
            for vcap in (None, 85, 100, 115, 130):
                if vcap is None:
                    b_sig, s_sig = b_cross, s_cross
                else:
                    b_sig = b_cross & (soxl_hv20 <= vcap * 0.92)
                    s_sig = s_cross | (soxl_hv20 > vcap)
                for tsl in (None, 0.20, 0.25):
                    for mh in (0, 10):
                        for cd in (0, 10, 20):
                            pstr = f"{src_name}_EMA({fast_n},{slow_n}),vcap={vcap},tsl={tsl},mh={mh},cd={cd}"
                            record("F3_DualMA_VolBrake", "DualEMA_VolBrake", pstr, b_sig, s_sig, {
                                "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd
                            })

    # -------------------------------------------------------------------------
    # Family 4: F4_Fisher_ATR (Fisher Momentum + ATR Chandelier Trailing Stop)
    # -------------------------------------------------------------------------
    for src_name, src_f, src_dc, src_wc, src_df, src_dfs, src_wf, src_wfs in (
        ("SOXL", soxl_f, closes, soxl_wc, soxl_df, soxl_dfs, soxl_wf, soxl_wfs),
        ("SMH", smh_f, smh_dc, smh_wc, smh_df, smh_dfs, smh_wf, smh_wfs),
    ):
        for tf in ("W", "D"):
            for gate_tf, gate_n in (("D", 100), ("D", 150), ("D", 200), ("W", 20), ("W", 30)):
                if gate_tf == "D":
                    gate_arr = sl(src_f[f"D_SMA{gate_n}"])
                    gate_up = src_dc > gate_arr
                    gate_dn = src_dc < gate_arr * 0.97
                else:
                    gate_arr = sl(src_f[f"W_SMA{gate_n}"])
                    gate_up = src_wc > gate_arr
                    gate_dn = src_wc < gate_arr * 0.97

                if tf == "W":
                    b_sig = is_w_end & gate_up & (src_wf > src_wfs)
                    s_sig = is_w_end & (gate_dn | ((src_wf < src_wfs) & (src_wf < -0.2)))
                else:
                    b_sig = gate_up & (src_df > src_dfs)
                    s_sig = gate_dn | ((src_df < src_dfs) & (src_df < -0.5))

                for atr_m in (2.5, 3.0, 3.5, 4.0, 5.0):
                    for mh in (0, 5, 10):
                        for cd in (0, 10, 20, 30):
                            pstr = f"{src_name}_{tf}_Fish+Gate({gate_tf}{gate_n}),atr={atr_m},mh={mh},cd={cd}"
                            record("F4_Fisher_ATR", "Fisher_ATR_Stop", pstr, b_sig, s_sig, {
                                "atr_sl_mult": atr_m, "min_hold_days": mh, "cooldown_days": cd
                            })

    # -------------------------------------------------------------------------
    # Family 5: F5_Macro_SMH_QQQ (SMH + QQQ Dual-Benchmark Regime + Dip Add-on)
    # -------------------------------------------------------------------------
    dip_buy = is_w_end & (soxl_wf < -1.8) & (soxl_dwf > 0)
    for ma_type in ("SMA", "EMA"):
        for smh_n in (80, 100, 150, 200):
            smh_ma = sl(smh_f[f"D_{ma_type}{smh_n}"])
            for qqq_n in (100, 150, 200):
                qqq_ma = sl(qqq_f[f"D_{ma_type}{qqq_n}"])
                for variant in ("AND", "SMH_primary_QQQ_guard"):
                    if variant == "AND":
                        b_base = (smh_dc > smh_ma) & (qqq_dc > qqq_ma)
                        s_base = (smh_dc < smh_ma * 0.99) | (qqq_dc < qqq_ma * 0.99)
                    else:
                        b_base = (smh_dc > smh_ma) & (qqq_dc > qqq_ma)
                        s_base = (smh_dc < smh_ma * 0.98) | (qqq_dc < qqq_ma * 0.97)
                    for dip in (False, True):
                        b_sig = (b_base | dip_buy) if dip else b_base
                        s_sig = s_base
                        for tsl in (None, 0.20, 0.25, 0.30):
                            for mh, cd in ((0, 0), (5, 5), (5, 10), (10, 10), (10, 20)):
                                pstr = f"{ma_type}(SMH{smh_n},QQQ{qqq_n}),{variant},dip={dip},tsl={tsl},mh={mh},cd={cd}"
                                record("F5_Macro_SMH_QQQ", "Macro_DualBench", pstr, b_sig, s_sig, {
                                    "trail_sl": tsl, "min_hold_days": mh, "cooldown_days": cd
                                })

    elapsed = time.perf_counter() - t0
    df_res = pd.DataFrame(results).sort_values(["Calmar", "CAGR"], ascending=[False, False]).reset_index(drop=True)

    out_csv = ROOT / "tmp" / "scan_results.csv"
    df_res.to_csv(out_csv, index=False)

    print(f"\nTotal combinations scanned: {len(df_res):,} in {elapsed:.2f}s -> saved {out_csv}")

    # Baseline check: C1
    c1_rows = df_res[(df_res["Family"] == "F1_FisherVol") & (df_res["Params"] == "W_SMA30,bot=True,mf=2.2,buf=0.95,tsl=None,fsl=None,mh=0,cd=0")]
    if not c1_rows.empty:
        print("\n=== Previous Strategy C1 (Causal Reference) ===")
        print(c1_rows[["Family", "Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate"]].to_string(index=False))

    cols = ["Family", "Params", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Sortino", "Trades", "WinRate", "Exposure", "WorstYear"]

    # Top 15 Overall by Calmar
    top15 = df_res[(df_res["Trades"] >= 5) & (df_res["CAGR"] >= 30.0)].head(15)
    print("\n=== Top 15 Overall by Calmar Ratio (Trades >= 5, CAGR >= 30%) ===")
    print(top15[cols].to_string(index=False))

    # Pareto Frontier by Max Drawdown Tier
    print("\n=== Pareto Frontier by Max Drawdown Tier (Top 3 Highest CAGR per Tier, Trades >= 5) ===")
    valid_df = df_res[df_res["Trades"] >= 5]
    for tier_label, dd_floor in [("MaxDD > -30%", -30.0), ("MaxDD > -40%", -40.0), ("MaxDD > -50%", -50.0), ("MaxDD > -60%", -60.0)]:
        sub = valid_df[valid_df["MaxDD"] > dd_floor].sort_values(["CAGR", "Calmar"], ascending=[False, False]).head(3)
        print(f"\n-- {tier_label} --")
        if sub.empty:
            print("  (No strategies in this tier)")
        else:
            print(sub[cols].to_string(index=False))

    # Best Strategy per Family
    print("\n=== Best Strategy per Family (by Calmar, Trades >= 5) ===")
    best_fam = valid_df.groupby("Family", as_index=False).first().sort_values("Calmar", ascending=False)
    print(best_fam[cols].to_string(index=False))

    # Detailed Breakdown: #1 Balanced (MaxDD > -50%, highest Calmar) and #1 Aggressive (highest CAGR with Trades >= 5)
    bal_candidates = valid_df[valid_df["MaxDD"] > -50.0]
    if bal_candidates.empty:
        bal_candidates = valid_df
    top_bal = bal_candidates.iloc[0]
    top_agg = valid_df.sort_values(["CAGR", "Calmar"], ascending=[False, False]).iloc[0]

    for label, row in [("#1 Balanced Strategy (MaxDD > -50%, Highest Calmar)", top_bal), ("#1 Aggressive Strategy (Highest CAGR)", top_agg)]:
        key = f"{row['Family']}|{row['Name']}|{row['Params']}"
        b_sig, s_sig, sim_kw = replay_specs[key]
        detail = simulate_signals(opens, closes, atr14, b_sig, s_sig, year_end_indices, record_details=True, **sim_kw)
        print(f"\n=== {label} ===")
        print(f"Family: {row['Family']} | Params: {row['Params']}")
        print(f"Final: ${detail['Final']:,.2f} | CAGR: {detail['CAGR']}% | MaxDD: {detail['MaxDD']}% | Calmar: {detail['Calmar']} | Sharpe: {detail['Sharpe']} | Sortino: {detail['Sortino']} | Trades: {detail['Trades']} | WinRate: {detail['WinRate']}% | Exposure: {detail['Exposure']}%")
        yr_str = " | ".join(f"{yr}: {val:+.1f}%" for yr, val in detail["yearly_rets"].items())
        print(f"Yearly Returns: {yr_str}")


if __name__ == "__main__":
    main()
