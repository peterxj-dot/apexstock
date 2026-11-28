#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/jessemacree/apexstock"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/daily_btc_update.log"
STAMP_FILE="$LOG_DIR/last_daily_btc_run_utc_date.txt"

mkdir -p "$LOG_DIR"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

# Check UTC time regardless of Mac local timezone.
# BTC daily bars close at 00:00 UTC; run once per UTC day after 00:15 UTC (09:15 JST).
CHECK_OUT="$("$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone

now_utc = datetime.now(timezone.utc)
in_window = (now_utc.hour > 0) or (now_utc.hour == 0 and now_utc.minute >= 15)
utc_date = now_utc.strftime("%Y-%m-%d")
print(f"{1 if in_window else 0}|{utc_date}|{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
PY
)"

IFS='|' read -r IN_WINDOW UTC_DATE UTC_STR <<< "$CHECK_OUT"
LAST_DATE=""
if [[ -f "$STAMP_FILE" ]]; then
  LAST_DATE="$(tr -d '[:space:]' < "$STAMP_FILE")"
fi

if [[ "$FORCE" -eq 0 ]]; then
  if [[ "$IN_WINDOW" -ne 1 ]]; then
    exit 0
  fi
  if [[ "$LAST_DATE" == "$UTC_DATE" ]]; then
    exit 0
  fi
fi

{
  echo "============================================================"
  echo "[START] Local: $(date '+%Y-%m-%d %H:%M:%S %Z') | UTC: $UTC_STR | Day: $UTC_DATE"
  cd "$ROOT"
  "$PYTHON_BIN" "$ROOT/tools/download_data.py" --tickers "BTC-USD,MSTR,QQQ,TQQQ,SMH,SOXL,TECL,VGT,^VIX,^VIX3M,^VXN,^MOVE,HYG,LQD"
  for fred_id in WALCL WTREGEN RRPONTSYD M2SL DFII10; do
    curl -4 -sL --max-time 20 "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${fred_id}&cosd=2013-01-01" -o "$ROOT/data/FRED_${fred_id}.csv" || true
  done
  "$PYTHON_BIN" "$ROOT/tools/build_btc_zones.py" --notify-tg
  "$PYTHON_BIN" "$ROOT/tools/build_mstr_zones.py" --notify-tg
  "$PYTHON_BIN" "$ROOT/tools/build_soxl_zones.py"
  "$PYTHON_BIN" "$ROOT/tools/build_tqqq_zones.py"
  "$PYTHON_BIN" "$ROOT/tools/build_tecl_zones.py"
  "$PYTHON_BIN" "$ROOT/tools/build_vgt_margin_compare.py"
  "$PYTHON_BIN" "$ROOT/tools/build_chart.py"
  "$PYTHON_BIN" "$ROOT/tools/build_risk_radar.py"
  "$PYTHON_BIN" "$ROOT/tools/build_panic_bottom_radar.py"
  "$PYTHON_BIN" "$ROOT/tools/build_chip_structure.py"
  "$PYTHON_BIN" "$ROOT/tools/build_dashboard.py"
  echo "$UTC_DATE" > "$STAMP_FILE"
  echo "[DONE] Marked $UTC_DATE complete at $(date '+%Y-%m-%d %H:%M:%S %Z')"
} >> "$LOG_FILE" 2>&1
