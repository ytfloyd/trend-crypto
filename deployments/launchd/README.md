# Launchd jobs (macOS)

This directory holds copies of the LaunchAgent plists that schedule recurring
jobs on the local macOS machine. The actual files **must live under**
`~/Library/LaunchAgents/` for `launchd` to load them — copies here are
checked in for reproducibility / version control only.

## `com.nrt.crypto-lake-refresh.plist`

Runs `scripts/cron_update_lake.sh` every day at **03:30 local time**, which:

1. Brings `bars_1d_clean`, `bars_1h_clean`, `bars_4h_clean` to current via
   `scripts/update_clean_tables_fast.py` (~12 s when already current).
2. Runs incremental `candles_1m` update via `scripts/collect_coinbase.py
   update --max-stale-days 7` so delisted/dead products are skipped (~80 s).
3. Logs to `logs/cron_update_lake_YYYYMMDD_HHMMSS.log` and rotates to keep
   the latest 30 daily logs.

### Install

```bash
cp deployments/launchd/com.nrt.crypto-lake-refresh.plist \
   ~/Library/LaunchAgents/

launchctl unload ~/Library/LaunchAgents/com.nrt.crypto-lake-refresh.plist 2>/dev/null
launchctl load -w  ~/Library/LaunchAgents/com.nrt.crypto-lake-refresh.plist

# Verify it's registered (state should be "0" or a positive PID):
launchctl list | grep crypto-lake-refresh
```

### Trigger immediately (for testing)

```bash
launchctl kickstart -k gui/$(id -u)/com.nrt.crypto-lake-refresh
```

### Tail today's log

```bash
ls -tr logs/cron_update_lake_*.log | tail -1 | xargs tail -f
```

### Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.nrt.crypto-lake-refresh.plist
rm ~/Library/LaunchAgents/com.nrt.crypto-lake-refresh.plist
```

### Notes

- Credentials come from `~/.secrets/cdp_api_key.json` (the same file
  `live_turtle` uses), not the `COINBASE_API_KEY` / `COINBASE_API_SECRET`
  env vars (which can drift stale).
- The job uses an `mkdir`-based lock at `logs/cron_update_lake.lock.d/` so
  overlapping runs (e.g. machine wakes from sleep right at 03:30 while a
  manual run is still going) silently skip.
- `RunAtLoad` is `false`, so loading the plist does **not** trigger an
  immediate run — first run is at the next scheduled time.
- macOS coalesces missed jobs: if the laptop is asleep at 03:30, the job
  fires once on wake.
