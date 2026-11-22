"""Simulate realistic synthetic SMH + SOXL paths (v2: Dollar-Volume->Share-Volume, 2-index divergence, cyclical regimes) and stress-test strategies across 50 unseen worlds."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp"))

from improve_tp3h_and_robustness import build_entry_mask, run_backtest_param
from indicators import ema, fisher_transform, historical_volatility, sma
from scan_exit_conditioned_reentry import simulate_exit_conditioned
from scan_soxl_strategies import INITIAL_CAPITAL, compute_asset_features, load_adjusted
from study_major_bottoms import build_weekly_dataset
from study_major_exits_and_backtest import run_buy_and_hold
from test_cycle_v2_and_hybrid import run_hybrid_backtest
from verify_bottom_volume import add_volume_features

TMP_DIR = ROOT / "tmp"
BT_START = "2015-01-01"
BT_END = "2026-11-19"
N_WORLDS = 50
SEED_START = 100


def build_study_data_from_dfs(
    target_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    ticker: str = "SYN_SOXL",
    bench_ticker: str = "SYN_SMH",
) -> dict:
    """In-memory equivalent of prepare_backtest_data + extend_causal_data + prepare_extended_data."""
    daily_df = target_df.copy()
    bdf = bench_df.copy()

    w, daily_df = build_weekly_dataset(daily_df, bdf)
    w = add_volume_features(w, daily_df, bdf)

    w_bench, _ = build_weekly_dataset(bdf, bdf)
    w["Bench_vs_30W%"] = w_bench["Close_vs_30W%"].reindex(w.index)
    w["Bench_M_Fish_prev"] = w_bench["M_Fish_prev"].reindex(w.index)

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
    w["M_Fish_Sig_prev"] = m_on_w["M_Fish_Sig_prev"]
    w["M_Fish_prev_diff"] = m_on_w["M_Fish_prev_diff"]

    min_low_200_8w = (1.0 + w["Low_vs_200W%"] / 100.0).rolling(8, min_periods=1).min()
    min_wf_4w = w["W_Fisher"].rolling(4, min_periods=1).min()
    w["pre_c7_vol"] = (min_low_200_8w <= 1.10) & (min_wf_4w < -1.5) & (w["max4w_Vol_R"] >= 1.6)
    w["W_EMA5"] = ema(w["Close"], 5)
    w["prev_W_High"] = w["High"].shift(1)
    w["pre_8w_low"] = w["Low"].rolling(8, min_periods=1).min()

    w["E1_C7Vol_EMA10"] = w["pre_c7_vol"] & (w["Close"] > w["W_EMA10"])
    w["E2_PrevHigh"] = w["pre_c7_vol"] & (w["Close"] > w["prev_W_High"])
    w["ALT_Buy_S1"] = (w["W_Fisher"] < -1.8) & ((w["dW_Fish"] > 0) | (w["W_Vol_R"] > 1.3))

    bench_aligned = bdf.reindex(daily_df.index, method="ffill")
    bench_sma200 = sma(bench_aligned["Close"], 200)
    daily_df["Bench_below_SMA200_99"] = (bench_aligned["Close"] < bench_sma200 * 0.99).fillna(False)
    daily_df["M_Fish_prev_ffill"] = m_ind["M_Fish_prev"].reindex(daily_df.index, method="ffill")

    daily_df["HV20"] = historical_volatility(daily_df["Close"], period=20, annualization_factor=252)
    daily_df["Bench_Close"] = bench_aligned["Close"]
    daily_df["Bench_SMA200"] = bench_sma200
    daily_df["Bench_EMA150"] = ema(bench_aligned["Close"], 150)
    daily_df["is_week_end"] = daily_df.index.isin(w.index)

    hi = w["High"].to_numpy(dtype=np.float64)
    lo = w["Low"].to_numpy(dtype=np.float64)
    nw = len(w)
    wks_52h = np.full(nw, np.nan, dtype=np.float64)
    dd52_8wl = np.full(nw, np.nan, dtype=np.float64)
    for i in range(nw):
        s52 = max(0, i - 51)
        p52 = s52 + int(np.argmax(hi[s52 : i + 1]))
        wks_52h[i] = float(i - p52)
        min8l = float(np.min(lo[max(0, i - 7) : i + 1]))
        dd52_8wl[i] = (min8l / hi[p52] - 1.0) * 100.0

    w["wks_52H"] = wks_52h
    w["dd52_8wL%"] = dd52_8wl
    w["min8w_Low_200W%"] = w["Low_vs_200W%"].rolling(8, min_periods=1).min()
    w["Close_vs_EMA10%"] = (w["Close"] / w["W_EMA10"] - 1.0) * 100.0
    w["W_Fish_max6w"] = w["W_Fisher"].rolling(6, min_periods=1).max()

    daily_pos_map = {dt: pos for pos, dt in enumerate(daily_df.index)}
    week_end_daily_pos = {daily_pos_map[w_dt]: w_pos for w_pos, w_dt in enumerate(w.index)}

    return {
        "ticker": ticker,
        "bench_ticker": bench_ticker,
        "daily": daily_df,
        "weekly": w,
        "week_end_daily_pos": week_end_daily_pos,
    }


def run_strategy1_from_dfs(
    target_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict:
    """In-memory equivalent of Strategy 1 (Paired_SMA200_hv95)."""
    full_idx = target_df.index
    tf = compute_asset_features(target_df, full_idx)
    bf = compute_asset_features(bench_df, full_idx)

    mask = (full_idx >= pd.Timestamp(start_date)) & (full_idx <= pd.Timestamp(end_date))
    bt_idx = full_idx[mask]

    def sl(obj) -> np.ndarray:
        return obj.to_numpy()[mask]

    opens = sl(tf["daily"]["Open"]).astype(np.float64)
    closes = sl(tf["daily"]["Close"]).astype(np.float64)
    is_w_end = sl(tf["is_week_end"]).astype(bool)
    t_hv20 = sl(tf["D_HV20"]).astype(np.float64)
    t_wf = sl(tf["W_Fisher"]).astype(np.float64)
    t_dwf = sl(tf["dW_Fisher"]).astype(np.float64)
    t_wvr = sl(tf["W_Vol_R"]).astype(np.float64)
    b_dc = sl(bf["daily"]["Close"]).astype(np.float64)
    b_sma200 = sl(bf["D_SMA200"]).astype(np.float64)

    vcap = 115.0
    vol_exit = t_hv20 > vcap
    alt_buy = is_w_end & (t_wf < -1.8) & ((t_dwf > 0) | (t_wvr > 1.3))
    tb_raw = b_dc > b_sma200 * 1.01
    tb = tb_raw & (t_hv20 <= vcap * 0.92)
    mb = b_dc < b_sma200 * 0.99

    yrs = bt_idx.year.to_numpy()
    yei = [(int(y), int(np.where(yrs == y)[0][-1])) for y in sorted(set(yrs))]

    r = simulate_exit_conditioned(
        opens,
        closes,
        tb,
        vol_exit,
        mb,
        alt_buy,
        yei,
        bt_idx,
        trend_tsl=None,
        alt_fsl=0.20,
        alt_tp=0.25,
        alt_max_days=40,
        default_acd=10,
        after_tsl_spec=("default", tb, alt_buy, 0),
        after_vol_spec=("hv_drop_95", tb_raw & (t_hv20 <= 95.0), alt_buy, 0),
        after_tb_spec=("default", tb, alt_buy, 0),
        after_fsl_spec=("default", tb, alt_buy, -1),
        after_tp_spec=("default", tb, alt_buy, -1, -1, None),
        record_details=True,
    )
    return {
        "System": "S1_Paired_SMA200_hv95",
        "Final": r["Final"],
        "CAGR": r["CAGR"],
        "MaxDD": r["MaxDD"],
        "Calmar": r["Calmar"],
        "Sharpe": r["Sharpe"],
        "Trades": r["Trades"],
        "WinRate": r["WinRate"],
        "trade_logs": r["trade_logs"],
    }


def generate_synthetic_pair(dates: pd.DatetimeIndex, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate correlated (SYN_SOXL, SYN_SMH) via semi-Markov cyclical GJR-GARCH,
    two-index constituent divergence, and DollarVolume->ShareVolume bottom capitulation."""
    rng = np.random.default_rng(seed)
    n = len(dates)

    # Regimes for common 1x semiconductor factor: 0=BULL, 1=BEAR_CHOP, 2=CRASH
    mu_reg = np.array([0.00192, -0.00135, -0.00380], dtype=np.float64)
    ann_v_reg = np.array([0.258, 0.410, 0.770], dtype=np.float64)
    v_reg = (ann_v_reg**2) / 252.0

    # GJR-GARCH(1,1) with Student-t innovations (unit variance)
    df_t = 7.0
    t_scale = np.sqrt((df_t - 2.0) / df_t)
    alpha_g = 0.06
    gamma_g = 0.08
    beta_g = 0.82
    persist = alpha_g + 0.5 * gamma_g + beta_g  # 0.92

    r_semi = np.empty(n, dtype=np.float64)
    sig_semi = np.empty(n, dtype=np.float64)

    state = 0
    days_in_state = 0
    c_semi = 24.0
    c_semi_pk = 24.0
    h = v_reg[0]
    prev_z = 0.0
    prev_neg = 0.0

    for t in range(n):
        dd_semi = c_semi / c_semi_pk - 1.0
        if t > 0:
            if state == 0:  # BULL: ~1.5-2.5y cycles
                p_bear = 0.0011 + 0.0036 * min(1.0, max(0.0, (days_in_state - 150) / 340.0))
                p_crash = 0.00035
                probs = [1.0 - p_bear - p_crash, p_bear, p_crash]
            elif state == 1:  # BEAR_CHOP: drawdown-aware recovery centered around -30%~-48% 1x SMH
                if dd_semi <= -0.27:
                    p_bull = 0.022 + 0.020 * min(1.0, (-dd_semi - 0.27) / 0.14)
                    p_crash = 0.0006
                else:
                    p_bull = 0.0065 + 0.004 * min(1.0, days_in_state / 110.0)
                    p_crash = 0.0022
                probs = [p_bull, max(0.0, 1.0 - p_bull - p_crash), p_crash]
            else:  # CRASH: brief capitulation panic followed by V-bottom or basing
                if dd_semi <= -0.30:
                    p_bull, p_bear = 0.042, 0.014
                else:
                    p_bull, p_bear = 0.024, 0.024
                probs = [p_bull, p_bear, 1.0 - p_bull - p_bear]
            nxt = int(rng.choice(3, p=probs))
            if nxt != state:
                state = nxt
                days_in_state = 0
                if state == 0:
                    # Reset local bull baseline cleanly once bear resolved? No, keep actual c_semi_pk so ATH tracks true peak!
                    pass
            else:
                days_in_state += 1

        omega = (1.0 - persist) * v_reg[state]
        h = omega + (alpha_g + gamma_g * prev_neg) * (prev_z**2) * h + beta_g * h
        sigma = float(np.sqrt(h))
        z = float(rng.standard_t(df_t) * t_scale)
        ret = float(np.clip(mu_reg[state] + sigma * z, -0.18, 0.22))
        r_semi[t] = ret
        sig_semi[t] = sigma
        c_semi = max(0.5, c_semi * (1.0 + ret))
        if c_semi > c_semi_pk:
            c_semi_pk = c_semi
        prev_z = (ret - mu_reg[state]) / max(sigma, 1e-6)
        prev_neg = 1.0 if ret < 0.0 else 0.0

    # Constituent divergence: SMH vs distinct 1x semiconductor basket underlying SOXL (plus ETF tracking error)
    eps_smh = rng.normal(0.0, 0.0014, size=n)
    r_smh = np.clip(r_semi + eps_smh, -0.18, 0.22)

    # Heteroskedastic divergence between SOXL's tracked index and SMH
    sox_diff_scale = 0.0020 + 0.055 * np.abs(r_semi)
    eps_sox = rng.normal(0.0, 1.0, size=n) * sox_diff_scale
    r_sox_1x = r_semi + eps_sox

    # Slight negative MA(1) tracking microstructure noise (matches empirical residual acf1 ~ -0.10)
    raw_etf = rng.normal(0.0, 0.0055, size=n)
    eps_etf = np.empty(n, dtype=np.float64)
    eps_etf[0] = raw_etf[0]
    eps_etf[1:] = raw_etf[1:] - 0.11 * raw_etf[:-1]

    r_soxl = np.clip(3.0672 * r_sox_1x - 0.1125 / 252.0 + eps_etf, -0.45, 0.55)

    c_smh = np.empty(n, dtype=np.float64)
    c_soxl = np.empty(n, dtype=np.float64)
    c_smh[0] = 24.0
    c_soxl[0] = 1.0
    for t in range(1, n):
        c_smh[t] = max(0.5, c_smh[t - 1] * (1.0 + r_smh[t]))
        c_soxl[t] = max(0.05, c_soxl[t - 1] * (1.0 + r_soxl[t]))

    # Correlated overnight gaps
    gap_smh = rng.normal(0.0, 1.0, size=n) * (0.63 * sig_semi)
    gap_soxl = 2.95 * gap_smh + rng.normal(0.0, 0.004, size=n)

    def _build_ohlc_and_dolvol(
        c: np.ndarray,
        r: np.ndarray,
        sig: np.ndarray,
        gap: np.ndarray,
        wick_k: float,
        base_vol: float,
        dol_slow_elast: float,
        dol_fast_elast: float,
        lv_sig_k: float,
        lv_panic_k: float,
        lv_noise: float,
    ) -> pd.DataFrame:
        o = np.empty(n, dtype=np.float64)
        o[0] = c[0] / (1.0 + 0.3 * r[0])
        o[1:] = np.clip(c[:-1] * (1.0 + gap[1:]), 0.01, None)

        up_w = rng.exponential(wick_k * sig)
        dn_w = rng.exponential(1.05 * wick_k * sig)
        h_px = np.maximum(o, c) * (1.0 + up_w)
        l_px = np.maximum(np.minimum(o, c) / (1.0 + dn_w), 0.005)

        # Decompose price horizon: 20d short move vs 100d (20w) cycle anchor
        # Daily ShareVol elasticity = dol_fast_elast - 1.0 (~ -0.52 => mild daily corr ~ -0.10)
        # Multi-week bottom ShareVol elasticity = dol_slow_elast - 1.0 (~ -0.95 => strong bottom surge)
        c_ser = pd.Series(c)
        ma_20d = c_ser.rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)
        ma_slow = c_ser.rolling(100, min_periods=1).mean().to_numpy(dtype=np.float64)
        lv_dol = np.empty(n, dtype=np.float64)
        cur = 0.0
        med_sig = float(np.median(sig))
        for t in range(n):
            v_shock = lv_sig_k * np.log(max(sig[t] / med_sig, 0.4))
            panic = lv_panic_k * max(0.0, -r[t] / max(2.2 * sig[t], 1e-4) - 0.35)
            cur = 0.65 * cur + v_shock + panic + float(rng.normal(0.0, lv_noise))
            lv_dol[t] = cur
        lv_dol -= float(np.mean(lv_dol))

        sec_trend = np.linspace(0.0, 1.4, n)
        rel_slow = np.clip(ma_20d / np.maximum(ma_slow, 1e-4), 0.10, 5.0)
        rel_fast = np.clip(c / np.maximum(ma_20d, 1e-4), 0.25, 4.0)
        dol_vol = base_vol * ma_slow * np.power(rel_slow, dol_slow_elast) * np.power(rel_fast, dol_fast_elast) * np.exp(sec_trend + lv_dol)
        share_vol = np.round(dol_vol / np.maximum(c, 0.01)).astype(np.int64)
        return pd.DataFrame(
            {"Open": o, "High": h_px, "Low": l_px, "Close": c, "Volume": np.maximum(share_vol, 1000)},
            index=dates,
        )

    sig_soxl = 3.0672 * sig_semi
    df_smh = _build_ohlc_and_dolvol(
        c_smh, r_smh, sig_semi, gap_smh, wick_k=0.14, base_vol=4_000_000.0,
        dol_slow_elast=0.45, dol_fast_elast=0.60, lv_sig_k=0.28, lv_panic_k=0.16, lv_noise=0.26,
    )
    df_soxl = _build_ohlc_and_dolvol(
        c_soxl, r_soxl, sig_soxl, gap_soxl, wick_k=0.15, base_vol=15_000_000.0,
        dol_slow_elast=0.05, dol_fast_elast=0.48, lv_sig_k=0.28, lv_panic_k=0.14, lv_noise=0.23,
    )
    return df_soxl, df_smh


def compute_stylized_facts(df_soxl: pd.DataFrame, df_smh: pd.DataFrame, sdata: dict) -> dict[str, float]:
    """Measure daily, weekly, and bottom-volume capitulation moments."""
    out: dict[str, float] = {}
    for pfx, df in [("SMH", df_smh), ("SOXL", df_soxl)]:
        o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
        ret = c.pct_change().dropna()
        hv20 = ret.rolling(20).std() * np.sqrt(252) * 100.0
        gap = (o / c.shift(1) - 1.0).dropna()
        hl = (h / l - 1.0).dropna()
        dd = (c / c.cummax() - 1.0) * 100.0
        wv = v.resample("W-FRI").sum()
        wvr = (wv / wv.rolling(20).mean()).dropna()

        out[f"{pfx}_AnnRet%"] = float(((c.iloc[-1] / c.iloc[0]) ** (252.0 / len(df)) - 1.0) * 100.0)
        out[f"{pfx}_AnnVol%"] = float(ret.std() * np.sqrt(252) * 100.0)
        out[f"{pfx}_Skew"] = float(ret.skew())
        out[f"{pfx}_Kurt"] = float(ret.kurt())
        out[f"{pfx}_MaxDD%"] = float(dd.min())
        out[f"{pfx}_HV20_p50"] = float(hv20.median())
        out[f"{pfx}_HV20_p90"] = float(hv20.quantile(0.90))
        out[f"{pfx}_GapStd%"] = float(gap.std() * 100.0)
        out[f"{pfx}_HL_p50%"] = float(hl.median() * 100.0)
        out[f"{pfx}_WVolR_p90"] = float(wvr.quantile(0.90))
        out[f"{pfx}_WVolR_p95"] = float(wvr.quantile(0.95))
        out[f"{pfx}_Corr_r_dlogV"] = float(ret.corr(np.log(v.astype(float)).diff()))

    r_smh = df_smh["Close"].pct_change().dropna()
    r_soxl = df_soxl["Close"].pct_change().dropna()
    beta = float(np.cov(r_soxl, r_smh)[0, 1] / np.var(r_smh))
    out["Beta_3x"] = beta
    out["Corr_SMH_SOXL"] = float(r_soxl.corr(r_smh))

    # Bottom share-volume surge diagnostics (2015-2026)
    w = sdata["weekly"]
    w_bt = w[w.index >= BT_START]
    rel_p = np.log(w["Close"] / w["Close"].rolling(20).mean())
    rel_v = np.log(w["W_Vol_R"])
    jj = pd.concat([rel_p, rel_v], axis=1).dropna()
    out["SOXL_Corr_P_VolR"] = float(jj.iloc[:, 0].corr(jj.iloc[:, 1])) if len(jj) > 10 else 0.0

    cond_bot = (w_bt["Low_vs_200W%"].rolling(8, min_periods=1).min() <= 10.0) & (
        w_bt["W_Fisher"].rolling(4, min_periods=1).min() < -1.5
    )
    bot_sub = w_bt[cond_bot]
    out["SOXL_P_Vol16_at_Bot%"] = (
        float((bot_sub["max4w_Vol_R"] >= 1.6).mean() * 100.0) if len(bot_sub) > 0 else float("nan")
    )
    out["SOXL_PreC7_Weeks"] = float(w_bt["pre_c7_vol"].fillna(False).sum())
    return out


def save_yahoo_csv(df: pd.DataFrame, path: Path) -> None:
    out = df[["Open", "High", "Low", "Close"]].copy()
    out["Adj Close"] = out["Close"]
    out["Volume"] = df["Volume"].astype(np.int64)
    out.index.name = "Date"
    out.to_csv(path, float_format="%.6f")


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 24)

    real_smh = load_adjusted("SMH")
    real_soxl = load_adjusted("SOXL")
    common_idx = real_smh.index.intersection(real_soxl.index)
    real_smh_c = real_smh.loc[common_idx]
    real_soxl_c = real_soxl.loc[common_idx]

    # 1. Exact parity check of in-memory adapters including Strategy 4
    real_data = build_study_data_from_dfs(real_soxl_c, real_smh_c, "SOXL", "SMH")
    f1_real = build_entry_mask(real_data, 30, -80.0)
    p_tp3h = run_backtest_param(real_data, f1_real, "TP3h_Base", BT_START, BT_END, filter_tag="F1")
    p_s1 = run_strategy1_from_dfs(real_soxl_c, real_smh_c, BT_START, BT_END)
    p_s4 = run_hybrid_backtest(
        real_data,
        "S4_H5_Promote_S1_AdaptHV",
        BT_START,
        BT_END,
        bot_entry_col="E2_PrevHigh",
        use_sl_w25=True,
        trend_mode="SMA200",
        hv_mode="ADAPT_115_140",
        hv_reentry=95.0,
        enable_alt=True,
    )
    assert abs(p_s1["Final"] - 29633196.29) < 1.0, p_s1
    assert abs(p_tp3h["Final"] - 90489657.17) < 1.0 and p_tp3h["Trades"] == 5, p_tp3h
    assert abs(p_s4["Final"] - 37913860.48) < 1.0, p_s4
    print(
        f"In-memory Adapter Parity PASSED: S1=${p_s1['Final']:,.2f}, "
        f"S4=${p_s4['Final']:,.2f}, F1+TP3h=${p_tp3h['Final']:,.2f}\n"
    )

    # 2. Generate 50 synthetic worlds & measure fidelity + strategy robustness
    real_facts = compute_stylized_facts(real_soxl_c, real_smh_c, real_data)
    syn_facts_list: list[dict[str, float]] = []
    worlds: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, float], dict]] = []
    strat_records: list[dict] = []

    strat_names = [
        "Buy&Hold",
        "S1_SMA200_hv95",
        "S4_H5_Promote_S1_AdaptHV",
        "F0+B0_X1a_WTSL35",
        "F0+TP3h_Base",
        "F1+TP3h_Base",
        "F1+U1_MDiff30",
    ]

    for w_i in range(N_WORLDS):
        seed = SEED_START + w_i
        syn_soxl, syn_smh = generate_synthetic_pair(common_idx, seed)
        sdata = build_study_data_from_dfs(syn_soxl, syn_smh, "SYN_SOXL", "SYN_SMH")
        facts = compute_stylized_facts(syn_soxl, syn_smh, sdata)
        syn_facts_list.append(facts)
        worlds.append((seed, syn_soxl, syn_smh, facts, sdata))

        f0_m = build_entry_mask(sdata, None, None)
        f1_m = build_entry_mask(sdata, 30, -80.0)

        r_bh = run_buy_and_hold(sdata, BT_START, BT_END)
        r_bh["System"] = "Buy&Hold"
        r_s1 = run_strategy1_from_dfs(syn_soxl, syn_smh, BT_START, BT_END)
        r_s4 = run_hybrid_backtest(
            sdata,
            "S4_H5_Promote_S1_AdaptHV",
            BT_START,
            BT_END,
            bot_entry_col="E2_PrevHigh",
            use_sl_w25=True,
            trend_mode="SMA200",
            hv_mode="ADAPT_115_140",
            hv_reentry=95.0,
            enable_alt=True,
        )
        r_f0_x1a = run_backtest_param(sdata, f0_m, "B0_X1a_WTSL35", BT_START, BT_END, filter_tag="F0")
        r_f0_tp = run_backtest_param(sdata, f0_m, "TP3h_Base", BT_START, BT_END, filter_tag="F0")
        r_f1_tp = run_backtest_param(sdata, f1_m, "TP3h_Base", BT_START, BT_END, filter_tag="F1")
        r_f1_u1 = run_backtest_param(sdata, f1_m, "U1_MDiff30", BT_START, BT_END, filter_tag="F1")

        bh_final = r_bh["Final"]
        s1_final = r_s1["Final"]
        for sname, res in zip(strat_names, [r_bh, r_s1, r_s4, r_f0_x1a, r_f0_tp, r_f1_tp, r_f1_u1]):
            strat_records.append(
                {
                    "seed": seed,
                    "System": sname,
                    "Final": res["Final"],
                    "CAGR": res["CAGR"],
                    "MaxDD": res["MaxDD"],
                    "Calmar": res["Calmar"],
                    "Sharpe": res["Sharpe"],
                    "Trades": res["Trades"],
                    "WinRate": res["WinRate"],
                    "Beat_BH": float(res["Final"] > bh_final),
                    "Beat_S1": float(res["Final"] > s1_final),
                }
            )

    # PART 1: Stylized facts table
    df_facts = pd.DataFrame(syn_facts_list)
    print("=" * 114)
    print(f"PART 1: STATISTICAL FIDELITY AUDIT v2 (REAL SMH/SOXL vs {N_WORLDS} SYNTHETIC WORLDS)")
    print("=" * 114)
    fact_rows = []
    for k, rval in real_facts.items():
        s = df_facts[k].dropna()
        p10, p50, p90 = float(s.quantile(0.10)), float(s.median()), float(s.quantile(0.90))
        in_band = "YES" if (p10 <= rval <= p90) else "EDGE"
        fact_rows.append(
            {
                "Metric": k,
                "Real": round(rval, 3),
                "Syn_P10": round(p10, 3),
                "Syn_Median": round(p50, 3),
                "Syn_P90": round(p90, 3),
                "In_P10_P90": in_band,
            }
        )
    print(pd.DataFrame(fact_rows).to_string(index=False))

    # PART 2: Export closest representative virtual pair (matching return, vol, maxDD, & PreC7 signal density)
    best_idx = 0
    best_dist = float("inf")
    for idx_w, (_, _, _, f, _) in enumerate(worlds):
        d_ret = (f["SOXL_AnnRet%"] - real_facts["SOXL_AnnRet%"]) / 15.0
        d_vol = (f["SOXL_AnnVol%"] - real_facts["SOXL_AnnVol%"]) / 10.0
        d_dd = (f["SOXL_MaxDD%"] - real_facts["SOXL_MaxDD%"]) / 8.0
        d_pc = (f["SOXL_PreC7_Weeks"] - real_facts["SOXL_PreC7_Weeks"]) / 20.0
        dist = d_ret * d_ret + d_vol * d_vol + d_dd * d_dd + d_pc * d_pc
        if dist < best_dist:
            best_dist = dist
            best_idx = idx_w

    rep_seed, rep_soxl, rep_smh, rep_facts, rep_data = worlds[best_idx]
    save_yahoo_csv(rep_soxl, TMP_DIR / "SYN_SOXL.csv")
    save_yahoo_csv(rep_smh, TMP_DIR / "SYN_SMH.csv")

    print("\n" + "=" * 114)
    print(f"PART 2: REPRESENTATIVE VIRTUAL STOCK v2 EXPORTED (seed={rep_seed}, saved to tmp/SYN_SOXL.csv & tmp/SYN_SMH.csv)")
    print("=" * 114)
    print(
        f"  SYN_SMH : AnnRet={rep_facts['SMH_AnnRet%']:.2f}%, AnnVol={rep_facts['SMH_AnnVol%']:.2f}%, MaxDD={rep_facts['SMH_MaxDD%']:.2f}%, HV20_med={rep_facts['SMH_HV20_p50']:.1f}%\n"
        f"  SYN_SOXL: AnnRet={rep_facts['SOXL_AnnRet%']:.2f}%, AnnVol={rep_facts['SOXL_AnnVol%']:.2f}%, MaxDD={rep_facts['SOXL_MaxDD%']:.2f}%, "
        f"HV20_med={rep_facts['SOXL_HV20_p50']:.1f}%, Beta={rep_facts['Beta_3x']:.3f}, Corr={rep_facts['Corr_SMH_SOXL']:.4f}, "
        f"BotVol16={rep_facts['SOXL_P_Vol16_at_Bot%']:.1f}%, PreC7={int(rep_facts['SOXL_PreC7_Weeks'])}w"
    )

    rep_f0 = build_entry_mask(rep_data, None, None)
    rep_f1 = build_entry_mask(rep_data, 30, -80.0)
    rep_rows = [
        {**run_buy_and_hold(rep_data, BT_START, BT_END), "System": "Buy&Hold (SYN_SOXL)"},
        run_strategy1_from_dfs(rep_soxl, rep_smh, BT_START, BT_END),
        run_hybrid_backtest(
            rep_data,
            "S4_H5_Promote_S1_AdaptHV",
            BT_START,
            BT_END,
            bot_entry_col="E2_PrevHigh",
            use_sl_w25=True,
            trend_mode="SMA200",
            hv_mode="ADAPT_115_140",
            hv_reentry=95.0,
            enable_alt=True,
        ),
        run_backtest_param(rep_data, rep_f0, "B0_X1a_WTSL35", BT_START, BT_END, filter_tag="F0"),
        run_backtest_param(rep_data, rep_f0, "TP3h_Base", BT_START, BT_END, filter_tag="F0"),
        run_backtest_param(rep_data, rep_f1, "TP3h_Base", BT_START, BT_END, filter_tag="F1"),
        run_backtest_param(rep_data, rep_f1, "U1_MDiff30", BT_START, BT_END, filter_tag="F1"),
    ]
    cols_show = ["System", "Final", "CAGR", "MaxDD", "Calmar", "Sharpe", "Trades", "WinRate"]
    print("\nRepresentative World v2 Backtest (2015-01-01 ~ 2026-11-19):")
    print(pd.DataFrame([{k: r[k] for k in cols_show} for r in rep_rows]).to_string(index=False))
    print("\nRepresentative World [F1+TP3h_Base] Trade Log:")
    print(pd.DataFrame(rep_rows[5]["trade_logs"]).to_string(index=False))

    # PART 3: 50-World Out-of-Sample Strategy Stress Test
    print("\n" + "=" * 114)
    print(f"PART 3: {N_WORLDS}-WORLD UNSEEN MONTE CARLO STRATEGY STRESS TEST v2 (2015-01-01 ~ 2026-11-19)")
    print("=" * 114)
    df_st = pd.DataFrame(strat_records)
    agg_rows = []
    for sname in strat_names:
        sub = df_st[df_st["System"] == sname]
        agg_rows.append(
            {
                "System": sname,
                "Beat_BH%": round(float(sub["Beat_BH"].mean() * 100.0), 1),
                "Beat_S1%": round(float(sub["Beat_S1"].mean() * 100.0), 1),
                "P25_Final": round(float(sub["Final"].quantile(0.25)), 0),
                "Med_Final": round(float(sub["Final"].median()), 0),
                "P75_Final": round(float(sub["Final"].quantile(0.75)), 0),
                "Med_CAGR%": round(float(sub["CAGR"].median()), 2),
                "Med_MaxDD%": round(float(sub["MaxDD"].median()), 2),
                "Worst_MaxDD%": round(float(sub["MaxDD"].min()), 2),
                "Med_Calmar": round(float(sub["Calmar"].median()), 3),
                "Med_Sharpe": round(float(sub["Sharpe"].median()), 3),
                "Med_Trades": round(float(sub["Trades"].median()), 1),
                "Med_WinRate%": round(float(sub["WinRate"].median()), 1),
            }
        )
    print(pd.DataFrame(agg_rows).to_string(index=False))
    print(f"\nCompleted {N_WORLDS} unseen worlds in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
