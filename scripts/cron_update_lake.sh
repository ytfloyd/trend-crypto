#!/usr/bin/env bash
# Daily refresh of the crypto data lake.
#
# Steps:
#   1. Bring bars_1d_clean / bars_1h_clean / bars_4h_clean to current
#      via scripts/update_clean_tables_fast.py (fast — minutes).
#   2. Bring candles_1m to current via scripts/collect_coinbase.py update
#      (incremental from last_ts; daily run = ~3-5 minutes).
#
# Designed to be run from cron once per day. Acquires a flock so concurrent
# invocations are skipped (no overlapping API hammering). Logs go to
# logs/cron_update_lake_YYYYMMDD.log under the repo root.
#
# Crontab entry installed by this script (see install_cron.sh):
#
#   30 3 * * *  /Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto/scripts/cron_update_lake.sh
#
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Env setup (cron has a minimal PATH; bring in what we need)
# ──────────────────────────────────────────────────────────────────────────────
export PATH="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin"
export HOME="${HOME:-/Users/russellfloyd}"
export TZ="America/Los_Angeles"

REPO="/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto"
DB="/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/market.duckdb"
CDP_KEY_FILE="/Users/russellfloyd/.secrets/cdp_api_key.json"

LOG_DIR="${REPO}/logs"
LOCK_DIR="${LOG_DIR}/cron_update_lake.lock.d"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cron_update_lake_${TS}.log"

mkdir -p "${LOG_DIR}"

# ──────────────────────────────────────────────────────────────────────────────
# Logging helper (everything goes to LOG_FILE and stderr)
# ──────────────────────────────────────────────────────────────────────────────
log() {
    printf '%s [cron_update_lake] %s\n' \
        "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$*" \
        | tee -a "${LOG_FILE}" >&2
}

_have_lock=0
cleanup() {
    rc=$?
    if [[ "${_have_lock}" -eq 1 ]]; then
        rmdir "${LOCK_DIR}" 2>/dev/null || true
    fi
    if [[ $rc -ne 0 ]]; then
        log "FAILED with exit $rc — see ${LOG_FILE}"
    else
        log "OK"
    fi
    exit $rc
}
trap cleanup EXIT

# ──────────────────────────────────────────────────────────────────────────────
# Single-instance lock (mkdir is atomic on macOS / Linux — no flock needed)
# ──────────────────────────────────────────────────────────────────────────────
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    # Stale lock recovery: if dir exists but holder PID is dead, take it.
    holder="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
    if [[ -n "${holder}" ]] && ! kill -0 "${holder}" 2>/dev/null; then
        log "stale lock held by dead pid ${holder} — reclaiming"
        rm -rf "${LOCK_DIR}"
        mkdir "${LOCK_DIR}"
    else
        log "another instance is already running (pid=${holder:-?}) — skipping"
        exit 0
    fi
fi
echo "$$" > "${LOCK_DIR}/pid"
_have_lock=1

# ──────────────────────────────────────────────────────────────────────────────
# Load CDP credentials from the live key file (env vars are stale)
# ──────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${CDP_KEY_FILE}" ]]; then
    log "ERROR: CDP key file not found at ${CDP_KEY_FILE}"
    exit 2
fi

COINBASE_API_KEY="$(/opt/anaconda3/bin/python -c "import json,sys; print(json.load(open('${CDP_KEY_FILE}'))['name'])")"
COINBASE_API_SECRET="$(/opt/anaconda3/bin/python -c "import json,sys; print(json.load(open('${CDP_KEY_FILE}'))['privateKey'])")"
export COINBASE_API_KEY COINBASE_API_SECRET

cd "${REPO}"

log "starting refresh — db=${DB}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: fast clean-table refresh (daily + hourly → also rebuilds 4h)
#   Backtests read these directly, so they must be current first.
# ──────────────────────────────────────────────────────────────────────────────
log "phase 1/2: fast clean-table refresh (1d + 1h, rebuilds 4h)"
/opt/anaconda3/bin/python scripts/update_clean_tables_fast.py \
    --db "${DB}" \
    --quotes USD,USDC \
    --granularities 1d,1h \
    --workers 8 \
    --max-rps 12 \
    >> "${LOG_FILE}" 2>&1
log "phase 1 done"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: incremental candles_1m update (consistency layer for raw 1m).
#   Runs only the delta from each symbol's last_ts → now, so daily incremental
#   is fast (~5 min for ~850 symbols).
# ──────────────────────────────────────────────────────────────────────────────
log "phase 2/2: candles_1m incremental update (skipping >7d-stale symbols)"
/opt/anaconda3/bin/python scripts/collect_coinbase.py \
    --db "${DB}" \
    --max-rps 12 \
    update --workers 8 --max-stale-days 7 \
    >> "${LOG_FILE}" 2>&1
log "phase 2 done"

# ──────────────────────────────────────────────────────────────────────────────
# Status report
# ──────────────────────────────────────────────────────────────────────────────
/opt/anaconda3/bin/python <<'PY' >> "${LOG_FILE}" 2>&1
import duckdb, datetime as dt
c = duckdb.connect('/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/market.duckdb', read_only=True)
print('=== POST-REFRESH STATUS ===')
for t in ('candles_1m', 'bars_1d_clean', 'bars_1h_clean', 'bars_4h_clean'):
    r = c.execute(f'SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(ts) FROM {t}').fetchone()
    print(f'  {t:18s}  {r[0]:>13,d} rows  {r[1]:>4d} syms  max_ts={r[2]}')
n = c.execute("""SELECT COUNT(*) FROM (
    SELECT symbol, MAX(ts) AS lt FROM bars_1d_clean GROUP BY symbol
) WHERE lt < NOW() - INTERVAL '2 days' """).fetchone()
print(f'  stale (>2d) symbols in bars_1d_clean: {n[0]}')
c.close()
PY

# ──────────────────────────────────────────────────────────────────────────────
# Log rotation: keep only last 30 daily logs
# ──────────────────────────────────────────────────────────────────────────────
find "${LOG_DIR}" -maxdepth 1 -type f -name 'cron_update_lake_*.log' \
    | sort -r | tail -n +31 | xargs -I{} rm -f {} 2>/dev/null || true
