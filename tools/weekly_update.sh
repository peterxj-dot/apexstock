#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/jessemacree/apexstock"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/weekly_update.log"
STAMP_FILE="$LOG_DIR/last_weekly_run_et_week.txt"

mkdir -p "$LOG_DIR"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

# Check US Eastern Time (America/New_York) regardless of Mac local timezone.
# Window: Friday >= 19:00 ET (3h after regular close), Saturday, or Sunday.
CHECK_OUT="$("$PYTHON_BIN" - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo

now_et = datetime.now(ZoneInfo("America/New_York"))
wd = now_et.weekday()  # Mon=0 .. Fri=4, Sat=5, Sun=6
in_window = (wd == 4 and now_et.hour >= 19) or (wd in (5, 6))
market_week = now_et.strftime("%G-W%V")
print(f"{1 if in_window else 0}|{market_week}|{now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
PY
)"

IFS='|' read -r IN_WINDOW MARKET_WEEK ET_STR <<< "$CHECK_OUT"
LAST_WEEK=""
if [[ -f "$STAMP_FILE" ]]; then
  LAST_WEEK="$(tr -d '[:space:]' < "$STAMP_FILE")"
fi

if [[ "$FORCE" -eq 0 ]]; then
  if [[ "$IN_WINDOW" -ne 1 ]]; then
    exit 0
  fi
  if [[ "$LAST_WEEK" == "$MARKET_WEEK" ]]; then
    exit 0
  fi
fi

{
  echo "============================================================"
  echo "[START] Local: $(date '+%Y-%m-%d %H:%M:%S %Z') | ET: $ET_STR | Week: $MARKET_WEEK"
  cd "$ROOT"
  "$PYTHON_BIN" "$ROOT/tools/download_data.py"
  "$PYTHON_BIN" "$ROOT/tools/build_soxl_zones.py" --notify-tg
  "$PYTHON_BIN" "$ROOT/tools/build_tqqq_zones.py"
  echo "$MARKET_WEEK" > "$STAMP_FILE"
  echo "[DONE] Marked $MARKET_WEEK complete at $(date '+%Y-%m-%d %H:%M:%S %Z')"
} >> "$LOG_FILE" 2>&1
