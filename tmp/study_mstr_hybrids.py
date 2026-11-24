"""Test hybrid BTC-Halving + MSTR-volatility / mNAV-proxy exits on MSTR (2020-2026)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tmp.study_mstr_strategies import _ds_du, compute_stats, prepare_data
from tools.build_soxl_zones import FEE_RATE, INITIAL_CAPITAL


def run_hybrid(df: pd.DataFrame, variant: str, start_date: str = "2020-01-02") -> tuple[dict, list[dict]]:
    d = df.set_index("Date").copy()
    # MSTR/BTC relative strength (mNAV proxy) & its 60d/120d ratio
    d["mstr_btc"] = d["Close"] / d["b_close"]
    d["mstr_btc_ma60"] = d["mstr_btc"].rolling(60).mean()
    d["mstr_btc_ratio60"] = d["mstr_btc"] / d["mstr_btc_ma60"]

    idx = d.index
    i_start = int(idx.searchsorted(pd.Timestamp(start_date), side="left"))
    i_end = len(idx) - 1
    n_bt = i_end - i_start + 1

    opens = d["Open"].to_numpy(dtype=np.float64)
    closes = d["Close"].to_numpy(dtype=np.float64)
    b_close = d["b_close"].to_numpy(dtype=np.float64)
    b_sma150 = d["b_sma150"].to_numpy(dtype=np.float64)
    b_hvr = d["b_hvr"].to_numpy(dtype=np.float64)
    b_sp35 = d["b_hvr_spike35"].to_numpy(dtype=bool)
    m_hv20 = d["m_hv20"].to_numpy(dtype=np.float64)
    m_hvr = d["m_hvr"].to_numpy(dtype=np.float64)
    m_gac_m = d["m_gac_mid"].to_numpy(dtype=np.float64)
    m_gac_u = d["m_gac_upper"].to_numpy(dtype=np.float64)
    m_sma50 = d["m_sma50"].to_numpy(dtype=np.float64)
    m_sma200 = d["m_sma200"].to_numpy(dtype=np.float64)
    m_btc_r60 = d["mstr_btc_ratio60"].to_numpy(dtype=np.float64)

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
    vol_cool_wait = False
    cooldown = 0

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
            entry_px = 0.0
            pending = ""

        if shares > 0.0:
            days_inv += 1
        elif cooldown > 0:
            cooldown -= 1

        equity[k] = cash + shares * c_px

        if pos < i_end:
            ds, du = _ds_du(idx[pos])
            in_buy = (du <= 540) and (ds > 560)
            in_sell = (ds >= 480) and (du > 540)
            in_bull = (du <= 540) or (ds <= 540)
            hv = m_hv20[pos]
            holding = shares > 0.0

            if variant == "BTC_Halv_HV_Combo":
                if not holding:
                    if in_buy and ((b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0) or du <= 490):
                        pending, pending_reason = "BUY", "BTC_HV_Cool/H-490"
                    elif ds < 480:
                        pending, pending_reason = "BUY", "Halv_Bull"
                else:
                    if (in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30) or (ds >= 540 and du > 540):
                        pending, pending_reason = "SELL", "SellZone_HVR/H+540"

            elif variant == "Halv_MSTR_HV130_Cool90":
                # Halving bottom entry + MSTR HV20>130% brake (re-enter when HV20<=90% & >SMA50) + Halving sell zone
                if not holding:
                    if cooldown == 0:
                        if vol_cool_wait:
                            if in_bull and ds < 480 and hv <= 90.0 and not np.isnan(m_sma50[pos]) and c_px > m_sma50[pos]:
                                vol_cool_wait = False
                                pending, pending_reason = "BUY", "MSTR_HVCool<=90"
                        else:
                            if in_buy and ((b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0) or du <= 490):
                                pending, pending_reason = "BUY", "BTC_HV_Cool/H-490"
                            elif ds < 480:
                                pending, pending_reason = "BUY", "Halv_Bull"
                else:
                    if ds >= 150 and hv > 130.0:
                        vol_cool_wait = True
                        cooldown = 5
                        pending, pending_reason = "SELL", "MSTR_HV>130"
                    elif (in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30) or (ds >= 540 and du > 540):
                        vol_cool_wait = False
                        pending, pending_reason = "SELL", "SellZone_HVR/H+540"

            elif variant == "Halv_MSTR_HV125_GACReentry":
                # Halving bottom entry + MSTR HV20>125% brake, re-enter only when MSTR > GAC_Upper inside bull
                if not holding:
                    if cooldown == 0:
                        if vol_cool_wait:
                            if ds < 480 and hv <= 105.0 and not np.isnan(m_gac_u[pos]) and c_px > m_gac_u[pos]:
                                vol_cool_wait = False
                                pending, pending_reason = "BUY", "MSTR_GAC_Reentry"
                        else:
                            if in_buy and ((b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0) or du <= 490):
                                pending, pending_reason = "BUY", "BTC_HV_Cool/H-490"
                            elif ds < 480:
                                pending, pending_reason = "BUY", "Halv_Bull"
                else:
                    if ds >= 150 and hv > 125.0:
                        vol_cool_wait = True
                        cooldown = 5
                        pending, pending_reason = "SELL", "MSTR_HV>125"
                    elif (in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30) or (ds >= 540 and du > 540):
                        vol_cool_wait = False
                        pending, pending_reason = "SELL", "SellZone_HVR/H+540"

            elif variant == "Halv_Phase_MSTR_GAC":
                # Accum [H-520~+100] hold; Mid/Late [H+100~+540] use MSTR GAC(10,2,1.5); Bear flat
                if not holding:
                    if (du <= 520 and ds > 540) or (0 <= ds <= 100):
                        pending, pending_reason = "BUY", "Accum"
                    elif 100 < ds <= 540 and not np.isnan(m_gac_u[pos]) and c_px > m_gac_u[pos]:
                        pending, pending_reason = "BUY", "MSTR_GAC_Up"
                else:
                    if ds > 540 and du > 520:
                        pending, pending_reason = "SELL", "Bear"
                    elif 100 < ds <= 540 and not np.isnan(m_gac_m[pos]) and c_px < m_gac_m[pos]:
                        pending, pending_reason = "SELL", "MSTR_GAC_Mid"

            elif variant == "Halv_mNAV_Spike180":
                # Take profit / trim when MSTR/BTC surges >1.8x 60d avg, re-enter when <1.15x inside bull
                if not holding:
                    if vol_cool_wait:
                        if ds < 480 and not np.isnan(m_btc_r60[pos]) and m_btc_r60[pos] <= 1.10 and c_px > m_gac_m[pos]:
                            vol_cool_wait = False
                            pending, pending_reason = "BUY", "mNAV_Cool"
                    else:
                        if in_buy and ((b_sp35[pos] and not np.isnan(b_hvr[pos]) and b_hvr[pos] < 1.0) or du <= 490):
                            pending, pending_reason = "BUY", "BTC_HV_Cool/H-490"
                        elif ds < 480:
                            pending, pending_reason = "BUY", "Halv_Bull"
                else:
                    if ds >= 150 and not np.isnan(m_btc_r60[pos]) and m_btc_r60[pos] >= 1.75:
                        vol_cool_wait = True
                        pending, pending_reason = "SELL", "mNAV_Spike>=1.75x"
                    elif (in_sell and not np.isnan(b_hvr[pos]) and b_hvr[pos] >= 1.30) or (ds >= 540 and du > 540):
                        vol_cool_wait = False
                        pending, pending_reason = "SELL", "SellZone_HVR/H+540"

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
    variants = [
        "BTC_Halv_HV_Combo",
        "Halv_MSTR_HV130_Cool90",
        "Halv_MSTR_HV125_GACReentry",
        "Halv_Phase_MSTR_GAC",
        "Halv_mNAV_Spike180",
    ]
    print("=== MSTR Hybrid Strategies (2020-01-02 ~ 2026-11-23) ===")
    for v in variants:
        st, tr = run_hybrid(df, v)
        print(
            f"{v:28s} | Final=${st['final']:12,.2f} | CAGR={st['cagr']:6.2f}% | "
            f"MaxDD={st['max_dd']:6.2f}% | Calmar={st['calmar']:5.3f} | Sharpe={st['sharpe']:4.2f} | "
            f"Trades={st['trades']:2d} | WR={st['win_rate']:5.1f}% | Exp={st['exposure']:5.1f}% | WorstYr={st['worst_yr']:6.1f}%"
        )
    print("\n--- Trades for Halv_MSTR_HV125_GACReentry & Halv_Phase_MSTR_GAC ---")
    for v in ["Halv_MSTR_HV125_GACReentry", "Halv_Phase_MSTR_GAC"]:
        _, tr = run_hybrid(df, v)
        print(f"\n[{v}]")
        for t in tr:
            print(f"  {t['entry_date']} @ ${t['entry_px']} -> {t['exit_date']} @ ${t['exit_px']} ({t['ret_pct']:+.1f}%) [{t['reason']}]")


if __name__ == "__main__":
    main()
