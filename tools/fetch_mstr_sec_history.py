"""Fetch MSTR (CIK 0001050446) historical BTC purchases, holdings, and ATM share issuances from SEC EDGAR."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "MSTR_sec_btc_history.csv"
OUT_SHARES_CSV = ROOT / "data" / "MSTR_sec_shares.csv"
HEADERS = {"User-Agent": "ApexStock peterxj@gmail.com"}
CIK = "0001050446"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    # Strip HTML tags & collapse whitespace
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = txt.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def fetch_xbrl_shares() -> pd.DataFrame:
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json")
    rows = []
    for ns, tag in [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesIssued"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ]:
        units_map = facts.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {})
        for unit_name, items in units_map.items():
            if "share" not in unit_name.lower():
                continue
            for it in items:
                end_dt = it.get("end")
                filed = it.get("filed")
                val = it.get("val")
                form = it.get("form", "")
                if end_dt and val and end_dt >= "2020-01-01" and form in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                    adj_val = val * 10 if (end_dt < "2024-08-01" and val < 50_000_000) else val
                    rows.append({
                        "as_of_date": end_dt,
                        "filed_date": filed,
                        "form": form,
                        "raw_shares": int(val),
                        "split_adj_shares": int(adj_val),
                        "tag": tag,
                    })
    if not rows:
        return pd.DataFrame(columns=["as_of_date", "filed_date", "form", "raw_shares", "split_adj_shares", "tag"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["filed_date", "as_of_date", "tag"], keep="first")
    df = df.sort_values(["filed_date", "as_of_date"]).reset_index(drop=True)
    return df


def parse_8k_btc_and_atm(txt: str) -> dict:
    out: dict = {
        "btc_purchased": None,
        "btc_total": None,
        "btc_avg_price": None,
        "mstr_shares_sold": None,
        "mstr_net_proceeds_m": None,
    }

    # 1) New-style 2025-2026 table: "BTC Purchased ... Aggregate BTC Holdings ... 457 $ 29.6 $ 64,898 851,305 $ 64.23 $ 75,446"
    m_new = re.search(
        r"Average Purchase Price\s*(?:\(\d+\)\s*)+([\d,]+)\s*\$\s*[\d,.]+\s*\$\s*([\d,]+)\s+([\d,]+)\s*\$\s*[\d,.]+\s*\$\s*([\d,]+)",
        txt,
    )
    if m_new:
        out["btc_purchased"] = int(m_new.group(1).replace(",", ""))
        out["btc_total"] = int(m_new.group(3).replace(",", ""))
        out["btc_avg_price"] = int(m_new.group(4).replace(",", ""))
    else:
        # 2) Narrative style: "acquired approximately 12,000 bitcoins ... held an aggregate of approximately 205,000 bitcoins"
        m_buy = re.search(r"(?:acquired|purchased)\s+(?:approximately\s+)?([\d,]+)\s+bitcoins?", txt, re.I)
        if m_buy:
            out["btc_purchased"] = int(m_buy.group(1).replace(",", ""))
        m_tot = re.search(
            r"(?:held|holds)\s+(?:an\s+aggregate\s+of\s+)?(?:approximately\s+)?([\d,]+)\s+bitcoins?",
            txt,
            re.I,
        )
        if m_tot:
            out["btc_total"] = int(m_tot.group(1).replace(",", ""))
        m_avg = re.search(r"average\s+purchase\s+price\s+of\s+(?:approximately\s+)?\$\s*([\d,]+)", txt, re.I)
        if m_avg:
            out["btc_avg_price"] = int(m_avg.group(1).replace(",", ""))

    # ATM MSTR Stock sold
    m_atm = re.search(
        r"MSTR\s+Stock\s+([\d,]+)\s+\$\s*(?:-|[\d,.]+)\s+\$\s*([\d,.]+)",
        txt,
    )
    if m_atm:
        out["mstr_shares_sold"] = int(m_atm.group(1).replace(",", ""))
        out["mstr_net_proceeds_m"] = float(m_atm.group(2).replace(",", ""))
    else:
        m_atm2 = re.search(r"issued\s+and\s+sold\s+(?:an\s+aggregate\s+of\s+)?([\d,]+)\s+shares\s+of\s+class\s+A\s+common\s+stock.*?net\s+proceeds.*?\$?([\d,.]+)\s*(million|billion)?", txt, re.I)
        if m_atm2:
            sh = int(m_atm2.group(1).replace(",", ""))
            amt = float(m_atm2.group(2).replace(",", ""))
            if m_atm2.group(3) and m_atm2.group(3).lower() == "billion":
                amt *= 1000.0
            out["mstr_shares_sold"] = sh
            out["mstr_net_proceeds_m"] = round(amt, 2)

    return out


def fetch_8k_history() -> pd.DataFrame:
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{CIK}.json")
    recent = sub["filings"]["recent"]
    all_batches = [recent]
    for f in sub["filings"].get("files", []):
        if f.get("filingTo", "9999") >= "2020-08-01":
            all_batches.append(_get_json(f"https://data.sec.gov/submissions/{f['name']}"))

    targets = []
    for batch in all_batches:
        forms = batch["form"]
        dates = batch["filingDate"]
        accs = batch["accessionNumber"]
        docs = batch["primaryDocument"]
        for form, fdt, acc, doc in zip(forms, dates, accs, docs):
            if form == "8-K" and fdt >= "2020-08-01":
                targets.append((fdt, acc, doc))

    print(f"Scanning {len(targets)} MSTR 8-K filings since 2020-08-01...")
    records = []
    for i, (fdt, acc, doc) in enumerate(targets):
        acc_nodash = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/1050446/{acc_nodash}/{doc}"
        try:
            txt = _get_text(url)
        except Exception as exc:
            print(f"  [WARN] {fdt} {acc}: {exc}")
            continue
        if ("bitcoin" in txt.lower()) or ("btc" in txt.lower()) or ("mstr stock" in txt.lower()):
            parsed = parse_8k_btc_and_atm(txt)
            if any(v is not None for v in parsed.values()):
                records.append({
                    "filing_date": fdt,
                    "accession_number": acc,
                    **parsed,
                })
        time.sleep(0.12)
    df = pd.DataFrame(records).sort_values("filing_date").reset_index(drop=True)
    return df


def main() -> None:
    df_shares = fetch_xbrl_shares()
    df_shares.to_csv(OUT_SHARES_CSV, index=False)
    print(f"Saved {len(df_shares)} share count records -> {OUT_SHARES_CSV}")

    df_8k = fetch_8k_history()
    df_8k.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df_8k)} BTC/ATM 8-K records -> {OUT_CSV}")
    print(df_8k.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
