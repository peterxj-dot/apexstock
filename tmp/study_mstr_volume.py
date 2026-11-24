"""Verify MSTR volume spikes around historical tops and test volume-augmented exits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tmp.study_mstr_strategies import _ds_du, compute_stats, prepare_data
from tools.build_soxl_zones import FEE_RATE, INITIAL_CAPITAL


def run_vol_test(df: pd.DataFrame, mode: str, start_date: str = "2020-01-02") -> tuple[dict, list[dict]]:
    d = df.set_index("Date").copy()
    d["mstr_btc"] = d["Close"] / d["b_close"]
    d["r60"] = d["mstr_btc"] / d["mstr_btc"].rolling(60).mean()
    d["vol_ma20"] = d["Volume"].rolling(20).mean()
    d["vol_r20"] = d["Volume"] / d["vol_ma20"]
    d["vol_r20_max3"] = d["vol_r20"].rolling(3).max()

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
    vol_r20 = d["vol_r20"].to_numpy(dtype=np.float64)
    vol_r20_max3 = d["vol_r20_max3"].to_numpy(dtype=np.float64)

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
            vr = vol_r20[pos]
            vr3 = vol_r20_max3[pos]
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
                if ds >= 150 and not np.isnan(rv) and not np.isnan(vr):
                    if mode == "BASE_R60_170":
                        if rv >= 1.70:
                            cool = True
                            pending, pending_reason = "SELL", f"R60>={rv:.2f}"
                    elif mode == "R60_170_AND_VOL15":
                        if rv >= 1.70 and vr3 >= 1.50:
                            cool = True
                            pending, pending_reason = "SELL", f"R60>={rv:.2f}&Vol={vr3:.1f}x"
                    elif mode == "R60_155_AND_VOL20":
                        if rv >= 1.55 and vr >= 2.0:
                            cool = True
                            pending, pending_reason = "SELL", f"R60={rv:.2f}&Vol={vr:.1f}x"
                    elif mode == "R60_150_AND_VOL22":
                        if rv >= 1.50 and vr >= 2.2:
                            cool = True
                            pending, pending_reason = "SELL", f"R60={rv:.2f}&Vol={vr:.1f}x"
                    elif mode == "R60_170_OR_R150_VOL22":
                        if rv >= 1.70 or (rv >= 1.50 and vr >= 2.2):
                            cool = True
                            pending, pending_reason = "SELL", f"R60={rv:.2f}|Vol={vr:.1f}x"
                    elif mode == "R60_170_OR_R145_VOL25":
                        if rv >= 1.70 or (rv >= 1.45 and vr >= 2.5):
                            cool = True
                            pending, pending_reason = "SELL", f"R60={rv:.2f}|Vol={vr:.1f}x"
                    elif mode == "PURE_VOL_CLIMAX_25":
                        if vr >= 2.5:
                            cool = True
                            pending, pending_reason = "SELL", f"Vol={vr:.1f}x"
                    elif mode == "PURE_VOL_CLIMAX_30":
                        if vr >= 3.0:
                            cool = True
                            pending, pending_reason = "SELL", f"Vol={vr:.1f}x"

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
    df = prepare_data().set_index("Date")
    df["mstr_btc"] = df["Close"] / df["b_close"]
    df["r60"] = df["mstr_btc"] / df["mstr_btc"].rolling(60).mean()
    df["vol_r20"] = df["Volume"] / df["Volume"].rolling(20).mean()

    print("=== Volume Ratios Around MSTR Historical Major Tops ===")
    for label, start, end in [
        ("2021-02 Top", "2021-02-01", "2021-02-19"),
        ("2024-03 Top", "2024-03-11", "2024-04-01"),
        ("2024-11 Top", "2024-11-18", "2024-11-26"),
        ("2025-07 Top", "2025-07-14", "2025-07-22"),
    ]:
        sub = df.loc[start:end, ["Close", "Volume", "vol_r20", "r60"]]
        pk_dt = sub["Close"].idxmax()
        mx_vol_dt = sub["vol_r20"].idxmax()
        print(
            f"{label:12s} | PricePeak={pk_dt.strftime('%Y-%m-%d')} @ ${sub.loc[pk_dt, 'Close']:.2f} (VolR={sub.loc[pk_dt, 'vol_r20']:.2f}x, R60={sub.loc[pk_dt, 'r60']:.2f}x) "
            f"| MaxVol={mx_vol_dt.strftime('%Y-%m-%d')} (VolR={sub.loc[mx_vol_dt, 'vol_r20']:.2f}x)"
        )

    df_reset = df.reset_index()
    modes = [
        "BASE_R60_170",
        "R60_170_AND_VOL15",
        "R60_155_AND_VOL20",
        "R60_150_AND_VOL22",
        "R60_170_OR_R150_VOL22",
        "R60_170_OR_R145_VOL25",
        "PURE_VOL_CLIMAX_25",
        "PURE_VOL_CLIMAX_30",
    ]
    print("\n=== Volume-Augmented Exit Backtest (2020-2026) ===")
    res = []
    for m in modes:
        st, tr = run_vol_test(df_reset, m)
        res.append((st["final"], m, st, tr))
    res.sort(reverse=True)
    for _, m, st, _ in res:
        print(
            f"{m:24s} | Final=${st['final']:12,.2f} | CAGR={st['cagr']:6.2f}% | "
            f"MaxDD={st['max_dd']:6.2f}% | Calmar={st['calmar']:5.3f} | Sharpe={st['sharpe']:4.2f} | "
            f"Trades={st['trades']:2d} | WR={st['win_rate']:5.1f}%"
        )

    print("\n--- Trades for Top 2 Volume Variants ---")
    for _, m, _, tr in res[:3]:
        if m == "BASE_R60_170":
            continue
        print(f"\n[{m}]")
        for t in tr:
            print(f"  {t['entry_date']} @ ${t['entry_px']} -> {t['exit_date']} @ ${t['exit_px']} ({t['ret_pct']:+.1f}%) [{t['reason']}]")


if __name__ == "__main__":
    main()
