"""30-minute breaking news & SEC filing sentinel for Nasdaq Top 20 + SPCX + MSTR.

Fetches incremental items from:
1) Yahoo Finance per-ticker news (yf.Ticker.news)
2) Google News RSS batches (including SpaceX/MSTR focus & public X mentions)
3) SEC EDGAR EFTS full-text search for recent 8-K / 6-K / S-3 / SC 13D / 424B5 filings

Deduplicates strictly via logs/news_sentinel/seen_items.json and outputs only unseen
candidates with priority hints so the scheduled agent can apply high-impact filtering
and send Telegram alerts only for truly material events.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.build_soxl_zones import send_tg_notification
from tools.daily_report_helper import _md_to_telegram_text

WATCHLIST_PATH = ROOT / "tools" / "news_sentinel_watchlist.json"
LOG_DIR = ROOT / "logs" / "news_sentinel"
SEEN_PATH = LOG_DIR / "seen_items.json"
LATEST_ALERT_PATH = LOG_DIR / "latest_alert.md"

USER_AGENT_BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
USER_AGENT_SEC = "ApexStockNewsSentinel/1.0 (contact: apexstock-local@example.com)"

HIGH_IMPACT_RE = re.compile(
    r"\b("
    r"earnings|guidance|revenue\s+miss|revenue\s+beat|profit\s+warning|outlook|pre-?announce|"
    r"acquir|acquisition|merger|buyout|takeover|spin-?off|divest|"
    r"antitrust|doj|ftc|sec\s+probe|subpoena|lawsuit|countersu|sues?\b|indict|sanction|tariff|export\s+(control|ban|rule|restrict)|match\s+act|ban\b|"
    r"ceo\s+(steps\s+down|resign|ousted|fired|depart)|cfo\s+(steps\s+down|resign|depart)|"
    r"8-k|s-3|424b|offering|convertible|dilution|buyback|repurchase|dividend\s+hike|stock\s+split|"
    r"lock-?up|lockup|starship|starlink\s+ipo|fcc\s+(approv|authoriz)|authoriz\w*\s+.*constellation|launch\s+fail|explosion|"
    r"bitcoin\s+purchas|buys?\s+\d[\d,\.]*\s*(btc|bitcoin)|atm\s+offering|mnav|"
    r"halted|shipment\s+halt|trading\s+halt|recall|data\s+breach|hack|cyberattack|layoff|restructur|"
    r"downgrade|upgrade|price\s+target|major\s+contract|pentagon|defense\s+contract|military\s+command|"
    r"blackwell|rubin|high-na|euv|foundry|18a|hbm4|robotaxi|fsd\s+approv"
    r")\b",
    re.IGNORECASE,
)

NOISE_RE = re.compile(
    r"("
    r"^\d+\s+(stocks|reasons|things|etfs)\s+to\s+(buy|watch|know|avoid)|"
    r"if\s+you\s+invested\s+\$|"
    r"should\s+you\s+(buy|sell|hold)|"
    r"better\s+buy:|"
    r"motley\s+fool|"
    r"zacks\s+rank|"
    r"dividend\s+stock\s+to\s+buy|"
    r"millionaire-maker|"
    r"prediction:\s+this\s+stock|"
    r"here'?s\s+why\s+.+\s+(is\s+up|is\s+down|fell|rose|gained|dropped)\s+today"
    r")",
    re.IGNORECASE,
)


def _norm_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\s*[-|–—]\s*[^-|–—]{2,40}$", "", t)  # strip trailing publisher suffix
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t).strip()
    return t


def _item_id(url: str, title: str) -> str:
    clean_url = url.split("?")[0].strip().lower() if url else ""
    base = _norm_title(title) or clean_url
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _parse_dt(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _fetch_yf_ticker_news(company: dict, cutoff_utc: datetime) -> list[dict]:
    ticker = company["ticker"]
    out: list[dict] = []
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return out

    for entry in raw_news[:12]:
        content = entry.get("content") if isinstance(entry, dict) and isinstance(entry.get("content"), dict) else entry
        if not isinstance(content, dict):
            continue
        title = str(content.get("title") or "").strip()
        if not title:
            continue
        pub_dt = _parse_dt(content.get("pubDate") or content.get("displayTime") or entry.get("providerPublishTime"))
        if pub_dt and pub_dt < cutoff_utc:
            continue
        url = ""
        canon = content.get("canonicalUrl")
        click = content.get("clickThroughUrl")
        if isinstance(canon, dict) and canon.get("url"):
            url = str(canon["url"])
        elif isinstance(click, dict) and click.get("url"):
            url = str(click["url"])
        elif entry.get("link"):
            url = str(entry["link"])
        provider = content.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(entry.get("publisher") or "")
        summary = str(content.get("summary") or "").strip()[:260]
        out.append({
            "source": "yfinance",
            "tickers": [ticker],
            "title": title,
            "publisher": publisher,
            "url": url,
            "published_utc": pub_dt.strftime("%Y-%m-%d %H:%M UTC") if pub_dt else "",
            "published_ts": pub_dt.timestamp() if pub_dt else 0.0,
            "summary": summary,
        })
    return out


def _fetch_google_news_batch(batch: dict, companies: list[dict], cutoff_utc: datetime) -> list[dict]:
    q = urllib.parse.quote(batch["query"])
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT_BROWSER})
    out: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
    except Exception:
        return out

    for item in root.findall(".//item")[:20]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        publisher = source_el.text.strip() if source_el is not None and source_el.text else ""
        pub_dt = _parse_dt(pub_raw)
        if not title or (pub_dt and pub_dt < cutoff_utc):
            continue

        matched_tickers = _match_tickers(title, companies)
        if not matched_tickers:
            continue
        out.append({
            "source": f"google_rss:{batch['label']}",
            "tickers": matched_tickers,
            "title": title,
            "publisher": publisher,
            "url": link,
            "published_utc": pub_dt.strftime("%Y-%m-%d %H:%M UTC") if pub_dt else "",
            "published_ts": pub_dt.timestamp() if pub_dt else 0.0,
            "summary": "",
        })
    return out


def _fetch_sec_filings(companies: list[dict], now_utc: datetime) -> list[dict]:
    """Fetch recent material SEC filings (8-K, 6-K, S-3, 424B5, SC 13D) via EFTS."""
    start_date = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = now_utc.strftime("%Y-%m-%d")
    tickers = [c["ticker"] for c in companies]
    # Query EFTS for material forms filed in last 24h across our tickers
    q = " OR ".join(f'"{t}"' for t in tickers)
    params = urllib.parse.urlencode({
        "q": q,
        "forms": "8-K,6-K,S-3,424B5,SC 13D,SC 13D/A",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
        "from": "0",
        "size": "40",
    })
    url = f"https://efts.sec.gov/LATEST/search-index?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT_SEC, "Accept": "application/json"})
    out: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return out

    ticker_set = set(tickers)
    name_map = {c["name"].lower(): c["ticker"] for c in companies}
    hits = data.get("hits", {}).get("hits", [])
    for h in hits:
        src = h.get("_source", {})
        display_names = src.get("display_names") or []
        matched: list[str] = []
        for dn in display_names:
            dn_str = str(dn)
            for m in re.findall(r"\(([A-Z]{1,5})\)", dn_str):
                if m in ticker_set and m not in matched:
                    matched.append(m)
            low = dn_str.lower()
            for nm, tk in name_map.items():
                if nm in low and tk not in matched:
                    matched.append(tk)
        if not matched:
            continue

        form = str(src.get("form") or src.get("root_form") or "FILING")
        file_date = str(src.get("file_date") or "")
        items = src.get("items") or []
        adsh = str(src.get("adsh") or h.get("_id") or "").split(":")[0]
        ciks = src.get("ciks") or []
        cik_num = str(int(ciks[0])) if ciks else ""
        adsh_nodash = adsh.replace("-", "")
        sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{adsh_nodash}/{adsh}-index.htm" if cik_num and adsh else "https://www.sec.gov/edgar/searchedgar/companysearch"
        entity = str(display_names[0]) if display_names else matched[0]
        items_str = f" Items {','.join(items)}" if items else ""
        title = f"[SEC {form}] {entity}{items_str} ({file_date})"
        pub_dt = _parse_dt(f"{file_date}T00:00:00Z") if file_date else now_utc
        out.append({
            "source": "sec_edgar",
            "tickers": matched,
            "title": title,
            "publisher": "SEC EDGAR",
            "url": sec_url,
            "published_utc": f"{file_date} (SEC)",
            "published_ts": pub_dt.timestamp() if pub_dt else now_utc.timestamp(),
            "summary": f"Form {form}{items_str}; adsh={adsh}",
        })
    return out


def _match_tickers(text: str, companies: list[dict]) -> list[str]:
    matched: list[str] = []
    for c in companies:
        tk = c["ticker"]
        for kw in c.get("keywords", [tk]):
            if len(kw) <= 4 and kw.isupper():
                if re.search(rf"(?<!\.)\b{re.escape(kw)}\b", text):
                    matched.append(tk)
                    break
            else:
                if kw.lower() in text.lower():
                    matched.append(tk)
                    break
    return matched


def _classify_priority(item: dict) -> str:
    if item["source"] == "sec_edgar":
        return "HIGH_SEC_FILING"
    blob = f"{item['title']} {item.get('summary', '')}"
    if NOISE_RE.search(item["title"]):
        return "LOW_NOISE"
    if HIGH_IMPACT_RE.search(blob):
        return "HIGH_KEYWORD"
    if any(t in ("SPCX", "MSTR") for t in item["tickers"]):
        return "WATCH_SPCX_MSTR"
    return "NORMAL"


def _load_seen() -> dict[str, float]:
    if not SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_seen(seen: dict[str, float], now_ts: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Keep rolling 14-day dedup cache
    min_ts = now_ts - 14 * 86400
    pruned = {k: v for k, v in seen.items() if v >= min_ts}
    SEEN_PATH.write_text(json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_fetch(hours: int = 12, dry_run: bool = False) -> None:
    cfg = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    companies = cfg["companies"]
    batches = cfg.get("google_news_batches", [])

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    now_utc = now_jst.astimezone(timezone.utc)
    now_ts = now_utc.timestamp()
    cutoff_utc = now_utc - timedelta(hours=hours)

    is_bootstrap = not SEEN_PATH.exists()
    seen = _load_seen()

    raw_items: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as ex:
        futs = [ex.submit(_fetch_yf_ticker_news, c, cutoff_utc) for c in companies]
        futs.extend(ex.submit(_fetch_google_news_batch, b, companies, cutoff_utc) for b in batches)
        futs.append(ex.submit(_fetch_sec_filings, companies, now_utc))
        for f in concurrent.futures.as_completed(futs):
            try:
                raw_items.extend(f.result())
            except Exception:
                pass

    # Merge duplicates across sources within this fetch
    merged: dict[str, dict] = {}
    for it in raw_items:
        iid = _item_id(it.get("url", ""), it.get("title", ""))
        if iid in merged:
            existing = merged[iid]
            for t in it["tickers"]:
                if t not in existing["tickers"]:
                    existing["tickers"].append(t)
            if not existing.get("summary") and it.get("summary"):
                existing["summary"] = it["summary"]
        else:
            it["id"] = iid
            merged[iid] = it

    new_items: list[dict] = []
    filtered_noise_count = 0
    already_seen_count = 0

    for iid, it in merged.items():
        if iid in seen:
            already_seen_count += 1
            continue
        prio = _classify_priority(it)
        it["priority_hint"] = prio
        seen[iid] = now_ts
        if prio == "LOW_NOISE":
            filtered_noise_count += 1
            continue
        new_items.append(it)

    if not dry_run:
        _save_seen(seen, now_ts)

    prio_order = {"HIGH_SEC_FILING": 0, "HIGH_KEYWORD": 1, "WATCH_SPCX_MSTR": 2, "NORMAL": 3}
    new_items.sort(key=lambda x: (prio_order.get(x["priority_hint"], 9), -(x.get("published_ts") or 0.0)))

    # Keep payload concise for low-token 30-min runs (all HIGH items + up to 25 remaining)
    high_items = [x for x in new_items if x["priority_hint"] in ("HIGH_SEC_FILING", "HIGH_KEYWORD", "WATCH_SPCX_MSTR")]
    normal_items = [x for x in new_items if x["priority_hint"] == "NORMAL"]
    selected = high_items[:50] + normal_items[:20]

    # Strip internal timestamp float
    for x in selected:
        x.pop("published_ts", None)

    status = "NO_NEW_ITEMS" if not selected else ("BOOTSTRAP_SEEDED" if is_bootstrap else "NEW_CANDIDATES")
    payload = {
        "status": status,
        "is_bootstrap": is_bootstrap,
        "generated_at_jst": now_jst.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "date_jst": now_jst.strftime("%Y-%m-%d"),
        "watchlist_count": len(companies),
        "raw_fetched": len(raw_items),
        "unique_fetched": len(merged),
        "already_seen_skipped": already_seen_count,
        "low_noise_skipped": filtered_noise_count,
        "new_high_or_watch_count": len(high_items),
        "new_normal_count": len(normal_items),
        "candidates": selected,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_notify(md_path: Path | None = None) -> None:
    target = md_path or LATEST_ALERT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Alert file not found: {target}")
    raw = target.read_text(encoding="utf-8").strip()
    if not raw:
        print("[Telegram] Empty alert file; skipped.")
        return

    msg = _md_to_telegram_text(raw)
    max_len = 3600
    chunks: list[str] = []
    cur = ""
    for para in msg.split("\n\n"):
        cand = f"{cur}\n\n{para}" if cur else para
        if len(cand) <= max_len:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            if len(para) <= max_len:
                cur = para
            else:
                for i in range(0, len(para), max_len):
                    chunks.append(para[i : i + max_len])
                cur = ""
    if cur:
        chunks.append(cur)

    for idx, chunk in enumerate(chunks, 1):
        prefix = f"({idx}/{len(chunks)})\n" if len(chunks) > 1 else ""
        ok = send_tg_notification(prefix + chunk)
        if not ok:
            raise RuntimeError(f"Failed sending Telegram chunk {idx}/{len(chunks)}")
    print(f"[Telegram] Sent {len(chunks)} chunk(s) from {target}")


def cmd_cleanup(delay_sec: int = 3) -> None:
    """Schedule self-deletion of the current Antigravity conversation after turn completion."""
    ls_addr = os.environ.get("ANTIGRAVITY_LS_ADDRESS", "").strip()
    csrf = os.environ.get("ANTIGRAVITY_CSRF_TOKEN", "").strip()
    conv_id = os.environ.get("ANTIGRAVITY_CONVERSATION_ID", "").strip()
    if not ls_addr or not csrf or not conv_id:
        print("[Cleanup] Missing ANTIGRAVITY_LS_ADDRESS / CSRF / CONVERSATION_ID; skipped.")
        return

    code = (
        "import json, sys, time, urllib.request; "
        "time.sleep(float(sys.argv[4])); "
        "req = urllib.request.Request("
        "f'http://{sys.argv[1]}/product.v1.ConversationService/DeleteConversation', "
        "data=json.dumps({'conversationId': sys.argv[3]}).encode('utf-8'), "
        "headers={'Content-Type': 'application/json', 'Connect-Protocol-Version': '1', 'x-antigravity-csrf-token': sys.argv[2]}, "
        "method='POST'); "
        "urllib.request.urlopen(req, timeout=10).read()"
    )
    subprocess.Popen(
        [sys.executable, "-c", code, ls_addr, csrf, conv_id, str(max(1, delay_sec))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[Cleanup] Scheduled self-delete for {conv_id} in {delay_sec}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="30-min breaking news sentinel for Nasdaq Top 20 + SPCX + MSTR.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch", help="Fetch incremental unseen news and SEC filings as JSON.")
    p_fetch.add_argument("--hours", type=int, default=12, help="Lookback window in hours (default: 12).")
    p_fetch.add_argument("--dry-run", action="store_true", help="Do not update seen_items.json.")
    p_not = sub.add_parser("notify", help="Send latest_alert.md to Telegram.")
    p_not.add_argument("path", nargs="?", type=Path, default=None, help="Optional Markdown alert path.")
    p_cln = sub.add_parser("cleanup", help="Self-delete current automation conversation after turn finishes.")
    p_cln.add_argument("--delay", type=int, default=3, help="Delay in seconds before deleting (default: 3).")
    args = parser.parse_args()
    if args.cmd == "fetch":
        cmd_fetch(hours=args.hours, dry_run=args.dry_run)
    elif args.cmd == "notify":
        cmd_notify(args.path)
    elif args.cmd == "cleanup":
        cmd_cleanup(delay_sec=args.delay)


if __name__ == "__main__":
    main()
