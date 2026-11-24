"""Test momentum deceleration / rollover signals on MSTR/BTC ratio for pinpoint top exits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import ema, fisher_transform, rsi, sma
from tmp.study_mstr_strategies import _ds_du, compute_stats, prepare_data
from tools.build_soxl_zones import FEE_RATE, INITIAL_CAPITAL


def run_momentum_test(df: pd.DataFrame, mode: str, start_date: str = "2020-01-02") -> tuple[dict, list[dict]]:
    d = df.set_index("Date").copy()
    d["mstr_btc"] = d["Close"] / d["b_close"]
    d["r60"] = d["mstr_btc"] / d["mstr_btc"].rolling(60).mean()
    d["r_ema5"] = ema(d["mstr_btc"], 5)
    d["r_ema10"] = ema(d["mstr_btc"], 10)
    d["r_ema20"] = ema(d["mstr_btc"], 20)
    d["r_roc10"] = d["mstr_btc"].pct_change(10)
    d["r_roc20"] = d["mstr_btc"].pct_change(20)
    d["r_roc10_ema5"] = ema(d["r_roc10"], 5)
    d["r_rsi14"] = rsi(d["mstr_btc"], 14)
    rf = fisher_transform(d["mstr_btc"], d["mstr_btc"], period=10)
    d["r_fish"] = rf.iloc[:, 0]
    d["r_fish_sig"] = rf.iloc[:, 1]
    d["b_roc10"] = d["b_close"].pct_change(10)
    d["m_roc10"] = d["Close"].pct_change(10)

    idx = d.index
    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = len(idx) - 1
    n_bt = i_end - i_start + 1

    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    b_hvr = d["b_hvr"].to_numpy(dtype=np.float64)
    b_sp35 = d["b_hvr_spike35"].to_numpy(dtype=bool)
    m_gac_m = d["m_gac_mid"].to_numpy(dtype=np.float64)
    r60 = d["r60"].to_numpy(dtype=np.float64)
    r_ema5 = d["r_ema5"].to_numpy(dtype=np.float64)
    r_ema10 = d["r_ema10"].to_numpy(dtype=np.float64)
    r_ema20 = d["r_ema20"].to_numpy(dtype=np.float64)
    r_roc10 = d["r_roc10"].to_numpy(dtype=np.float64)
    r_roc10_ema5 = d["r_roc10_ema5"].to_numpy(dtype=np.float64)
    r_rsi14 = d["r_rsi14"].to_numpy(dtype=np.float64)
    r_fish = d["r_fish"].to_numpy(dtype=np.float64)
    r_fish_sig = d["r_fish_sig"].to_numpy(dtype=np.float64)
    mstr_btc = d["mstr_btc"].to_numpy(dtype=np.float64)
    b_roc10 = d["b_roc10"].to_numpy(dtype=np.float64)

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
    cool = False
    armed_high = False

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
            armed_high = False
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
            armed_high = False
            pending = ""

        if shares > 0.0:
            days_inv += 1

        equity[k] = cash + shares * c_px

        if pos < i_end:
            ds, du = _ds_du(idx[pos])
            in_buy = (du <= 540) and (ds > 560)
            in_sell = (ds >= 480) and (du > 540)
            bhvr = b_hvr[pos]
            bsp = bool(b_sp35[pos])
            rv = r60[pos]
            gm = m_gac_m[pos]

            if shares == 0.0:
                if cool:
                    if ds < 480 and not np.isnan(rv) and rv <= 1.10 and not np.isnan(gm) and c_px > gm:
                        cool = False
                        pending, pending_reason = "BUY", "mNAV_Cool"
                else:
                    if in_buy and ((bsp and not np.isnan(bhvr) and bhvr < 1.0) or du <= 490):
                        pending, pending_reason = "BUY", "Halv_Buy"
                    elif ds < 480:
                        pending, pending_reason = "BUY", "Halv_Bull"
            else:
                if ds >= 150 and not np.isnan(rv):
                    if mode == "BASE_R60_170":
                        if rv >= 1.70:
                            cool = True
                            pending, pending_reason = "SELL", f"R60>={rv:.2f}"
                    elif mode == "ARM145_EMA5_CROSS_EMA10":
                        if rv >= 1.45:
                            armed_high = True
                        if armed_high and r_ema5[pos] < r_ema10[pos]:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm145+EMA5<10(r60={rv:.2f})"
                    elif mode == "ARM140_MB_CROSS_EMA10":
                        if rv >= 1.40:
                            armed_high = True
                        if armed_high and mstr_btc[pos] < r_ema10[pos]:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm140+MB<EMA10(r60={rv:.2f})"
                    elif mode == "ARM150_MB_CROSS_EMA10":
                        if rv >= 1.50:
                            armed_high = True
                        if armed_high and mstr_btc[pos] < r_ema10[pos]:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm150+MB<EMA10(r60={rv:.2f})"
                    elif mode == "ARM145_ROC10_DECEL":
                        # Armed when r60>=1.45, sell when 10d ROC of MSTR/BTC falls below its 5d EMA & turns < 5%
                        if rv >= 1.45:
                            armed_high = True
                        if armed_high and r_roc10[pos] < r_roc10_ema5[pos] and r_roc10[pos] < 0.05:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm145+ROC10_Decel(r60={rv:.2f})"
                    elif mode == "ARM140_ROC10_DECEL_BTCUP":
                        # Specifically when BTC is still rising (b_roc10 > 0) but MSTR/BTC ROC10 decelerates
                        if rv >= 1.40:
                            armed_high = True
                        if armed_high and b_roc10[pos] > 0 and r_roc10[pos] < r_roc10_ema5[pos] and r_roc10[pos] < 0.02:
                            cool = True
                            pending, pending_reason = "SELL", f"BTC_Up+MB_Decel(r60={rv:.2f})"
                    elif mode == "ARM140_FISH_XDOWN":
                        if rv >= 1.40:
                            armed_high = True
                        if armed_high and r_fish[pos] > 1.0 and r_fish[pos] < r_fish_sig[pos]:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm140+MB_FishX(r60={rv:.2f})"
                    elif mode == "ARM145_RSI70_ROLL":
                        if rv >= 1.45 and r_rsi14[pos] >= 70.0:
                            armed_high = True
                        if armed_high and r_rsi14[pos] < 65.0:
                            cool = True
                            pending, pending_reason = "SELL", f"Arm145+RSI<65(r60={rv:.2f})"

                if pending == "" and ((in_sell and not np.isnan(bhvr) and bhvr >= 1.30) or (ds >= 540 and du > 540)):
                    cool = False
                    pending, pending_reason = "SELL", "Halv_SellZone"

    if shares > 0.0:
        ret_pct = (shares * closes[i_end] / trade_cash_start - 1.0) * 100.0
        trades.append({
            "entry_date": entry_dt, "entry_px": round(entry_px, 2),
            "exit_date": idx[i_end].strftime("%Y-%m-%d") + "(OPEN)", "exit_px": round(closes[i_end], 2),
            "ret_pct": round(ret_pct, 1), "reason": "OPEN",
        })
    return compute_stats(equity, trades, days_inv, idx[i_start:i_end + 1]), trades


def main() -> None:
    df = prepare_data()
    modes = [
        "BASE_R60_170",
        "ARM145_EMA5_CROSS_EMA10",
        "ARM140_MB_CROSS_EMA10",
        "ARM150_MB_CROSS_EMA10",
        "ARM145_ROC10_DECEL",
        "ARM140_ROC10_DECEL_BTCUP",
        "ARM140_FISH_XDOWN",
        "ARM145_RSI70_ROLL",
    ]
    print("=== MSTR/BTC Momentum Deceleration vs Threshold (2020-2026) ===")
    res = []
    for m in modes:
        st, tr = run_momentum_test(df, m)
        res.append((st["final"], m, st, tr))
    res.sort(reverse=True)
    for _, m, st, _ in res:
        print(
            f"{m:26s} | Final=${st['final']:12,.2f} | CAGR={st['cagr']:6.2f}% | "
            f"MaxDD={st['max_dd']:6.2f}% | Calmar={st['calmar']:5.3f} | Sharpe={st['sharpe']:4.2f} | "
            f"Trades={st['trades']:2d} | WR={st['win_rate']:5.1f}%"
        )
    print("\n--- Top 3 Momentum Trades ---")
    for _, m, _, tr in res[:3]:
        print(f"\n[{m}]")
        for t in tr:
            print(f"  {t['entry_date']} @ ${t['entry_px']} -> {t['exit_date']} @ ${t['exit_px']} ({t['ret_pct']:+.1f}%) [{t['reason']}]")


if __name__ == "__main__":
    main()
