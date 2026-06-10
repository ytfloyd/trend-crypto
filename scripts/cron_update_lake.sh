#!/bin/bash
# Nightly Coinbase crypto data-lake refresh (invoked by launchd job
# com.nrt.crypto-lake-refresh, daily 03:30 local).
#
#   phase 1/2: candles_1m incremental update (all symbols already in the lake)
#   phase 2/2: rebuild clean tables (bars_1h_clean, bars_4h_clean, bars_1d_clean)
#
# Market data is pulled from Coinbase's UNAUTHENTICATED public endpoints, so no
# API credentials are required (the collector prefers get_public_candles).
#
# Concurrency-safe via an atomic mkdir lock under logs/cron_update_lake.lock.d.
# Per-run logs are written to logs/cron_update_lake_<ts>.log.

set -o pipefail

REPO="/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto"
LAKE="/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
WORKERS="${CRYPTO_LAKE_WORKERS:-4}"
MAXRPS="${CRYPTO_LAKE_MAX_RPS:-8}"

# Prefer anaconda python if present (matches launchd PATH), else system python3.
if [ -x /opt/anaconda3/bin/python3 ]; then
  PY=/opt/anaconda3/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY=python3
fi

cd "$REPO" || exit 1
export NUMEXPR_MAX_THREADS=8
# Market data needs no auth; drop any (possibly revoked) CDP creds so the REST
# client never attempts signed calls that would 401.
unset COINBASE_API_KEY COINBASE_API_SECRET

mkdir -p logs
LOCK_DIR="logs/cron_update_lake.lock.d"
LOCK_PID="$LOCK_DIR/pid"
RUN_LOG="logs/cron_update_lake_$(date +%Y%m%d_%H%M%S).log"

log() { echo "$(date "+%Y-%m-%dT%H:%M:%S%z") [cron_update_lake] $*" | tee -a "$RUN_LOG"; }

# ── acquire lock (atomic mkdir); reclaim if held by a dead pid ───────────────
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [ -f "$LOCK_PID" ]; then
    held=$(cat "$LOCK_PID" 2>/dev/null)
    if [ -n "$held" ] && kill -0 "$held" 2>/dev/null; then
      log "another refresh is running (pid $held) — exiting"
      exit 0
    fi
    log "stale lock held by dead pid ${held:-?} — reclaiming"
  else
    log "stale lock with no pid file — reclaiming"
  fi
fi
# Ensure the lock dir exists (we either created it or are reclaiming it).
mkdir -p "$LOCK_DIR"
echo "$$" > "$LOCK_PID"
trap 'rm -rf "$LOCK_DIR"' EXIT

log "starting refresh — db=$LAKE workers=$WORKERS max_rps=$MAXRPS py=$PY"

log "phase 1/2: candles_1m incremental update"
"$PY" scripts/collect_coinbase.py --db "$LAKE" --max-rps "$MAXRPS" update --workers "$WORKERS" >>"$RUN_LOG" 2>&1
rc1=$?
log "phase 1 done (rc=$rc1)"

log "phase 2/2: rebuild clean tables (1h/4h/1d)"
"$PY" scripts/collect_coinbase.py --db "$LAKE" refresh >>"$RUN_LOG" 2>&1
rc2=$?
log "phase 2 done (rc=$rc2)"

if [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]; then
  log "OK"
  exit 0
else
  log "FINISHED WITH ERRORS (rc1=$rc1 rc2=$rc2)"
  exit 1
fi
