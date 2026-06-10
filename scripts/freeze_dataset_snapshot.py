#!/usr/bin/env python
"""Freeze an immutable, point-in-time snapshot of the market datasets for
reproducible backtesting.

The snapshot is a self-contained directory of read-only DuckDB files + futures
parquet that mirrors the live layout, so it is a drop-in for
``LakeDataProvider(data_root=<snapshot_dir>)``.

Cutoff semantics
----------------
A single instant ``cutoff_utc`` is derived from ``--cutoff-date`` interpreted as
"end of that day" in ``--tz`` (i.e. the following local midnight).  All bars with
``ts < cutoff_utc`` are kept, with ONE exception: crypto *daily* bars are
UTC-day-aligned and stamped at the day's 00:00 UTC start, so a naive instant cut
would include a still-forming day.  We therefore keep crypto daily bars only up
to the last fully-completed UTC day at/under the cutoff (``ts < floor_utc_day``),
preventing a partial bar from leaking into backtests.

Datasets
--------
* crypto  : coinbase_crypto_ohlcv_lake.duckdb — default: clean 1d/4h/1h (+ optional 1m).
            ``--full``: every table in the crypto lake.
* etf     : etf_market.duckdb -> bars_1d.
* stocks  : stocks_market.duckdb — default: bars_1d only; ``--full``: all tables
            (option_chains, vol_surface_snaps, gamma_screener_daily, option_ticks).
* futures : continuous-futures parquet artifacts (CL / NG / SI / ES / VX / NQ / RTY).
* futures_lake (``--full`` only): futures_market.duckdb -> bars_1m + ingest_state,
            all symbols / expiries, filtered to cutoff.
* indices (``--full`` only): indices_market.duckdb -> bars_1d for Cboe indices
            (VIXEQ, DSPX, VIX, …).

Each output file is chmod 0444 (read-only) and a MANIFEST.json / MANIFEST.md is
written with the cutoff, per-table row counts, min/max ts, source path, and a
sha256 of every frozen file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

DATA_ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/data")
REPO_ROOT = Path("/Users/russellfloyd/Dropbox/NRT/nrt_dev/trend_crypto")

CRYPTO_DB = DATA_ROOT / "coinbase_crypto_ohlcv_lake.duckdb"
ETF_DB = DATA_ROOT / "etf_market.duckdb"
STOCKS_DB = DATA_ROOT / "stocks_market.duckdb"

CRYPTO_INTRADAY_TABLES = ["bars_4h_clean", "bars_1h_clean"]
CRYPTO_DAILY_TABLE = "bars_1d_clean"
FUTURES_DB = DATA_ROOT / "futures_market.duckdb"
INDICES_DB = DATA_ROOT / "indices_market.duckdb"

STOCKS_FULL_TABLES: dict[str, str] = {
    "bars_1d": "ts < TIMESTAMP '{cut_naive}'",
    "gamma_screener_daily": "as_of_date <= DATE '{cut_date}'",
    "option_chains": "last_updated < TIMESTAMP WITH TIME ZONE '{cut_tz}'",
    "vol_surface_snaps": "snap_ts < TIMESTAMP WITH TIME ZONE '{cut_tz}'",
    "option_ticks": "TRUE",
}

FUTURES_PARQUET = {
    # CL: front-month unadjusted stitch (2022+) — institutional valid-only series
    # remains at cl_institutional_continuous/ for production rolls.
    "CL": REPO_ROOT / "artifacts/research/cl_continuous/bars_1m.parquet",
    "NG": REPO_ROOT / "artifacts/research/ng_institutional_continuous/bars_1m.parquet",
    "SI": REPO_ROOT / "artifacts/research/si_quicklook/si_front_month_1m.parquet",
    "ES": REPO_ROOT / "artifacts/research/es_continuous/bars_1m.parquet",
    "VX": REPO_ROOT / "artifacts/research/vx_continuous/bars_1m.parquet",
    "NQ": REPO_ROOT / "artifacts/research/nq_continuous/bars_1m.parquet",
    "RTY": REPO_ROOT / "artifacts/research/rty_continuous/bars_1m.parquet",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_readonly(path: Path) -> None:
    os.chmod(path, 0o444)


def _cutoff_sql_literals(cutoff_utc: dt.datetime, daily_floor_utc: dt.datetime) -> dict[str, str]:
    cut = cutoff_utc.strftime("%Y-%m-%d %H:%M:%S%z")
    cut = cut[:-2] + ":" + cut[-2:]
    dfloor = daily_floor_utc.strftime("%Y-%m-%d %H:%M:%S%z")
    dfloor = dfloor[:-2] + ":" + dfloor[-2:]
    cut_naive = cutoff_utc.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "cut_tz": cut,
        "daily_floor_tz": dfloor,
        "cut_naive": cut_naive,
        "cut_date": cutoff_utc.astimezone(dt.timezone.utc).date().isoformat(),
    }


def _table_columns(con: duckdb.DuckDBPyConnection, table: str, *, schema: str | None = None) -> set[str]:
    if schema:
        return {row[0] for row in con.execute(f'DESCRIBE {schema}."{table}"').fetchall()}
    return {row[0] for row in con.execute(f'DESCRIBE "{table}"').fetchall()}


def freeze_crypto(out_db: Path, cutoff_utc: dt.datetime, daily_floor_utc: dt.datetime,
                  include_1m: bool, *, full: bool = False) -> list[dict]:
    lit = _cutoff_sql_literals(cutoff_utc, daily_floor_utc)
    con = duckdb.connect(str(out_db))
    con.execute(f"ATTACH '{CRYPTO_DB}' AS src (READ_ONLY)")
    rows: list[dict] = []

    if full:
        tables = [r[0] for r in con.execute("SHOW TABLES FROM src").fetchall()]
        for t in sorted(tables):
            cols = _table_columns(con, t, schema="src")
            if "ts" in cols and "1d" in t:
                where = f"ts < TIMESTAMP WITH TIME ZONE '{lit['daily_floor_tz']}'"
            elif "ts" in cols:
                where = f"ts < TIMESTAMP WITH TIME ZONE '{lit['cut_tz']}'"
            elif "time" in cols:
                where = f"time < TIMESTAMP WITH TIME ZONE '{lit['cut_tz']}'"
            else:
                where = "TRUE"
            con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}" WHERE {where}')
    else:
        con.execute(
            f'CREATE TABLE "{CRYPTO_DAILY_TABLE}" AS SELECT * FROM src."{CRYPTO_DAILY_TABLE}" '
            f"WHERE ts < TIMESTAMP WITH TIME ZONE '{lit['daily_floor_tz']}'"
        )
        tables = [CRYPTO_DAILY_TABLE] + list(CRYPTO_INTRADAY_TABLES)
        for t in CRYPTO_INTRADAY_TABLES:
            con.execute(
                f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}" '
                f"WHERE ts < TIMESTAMP WITH TIME ZONE '{lit['cut_tz']}'"
            )
        if include_1m:
            con.execute(
                f"CREATE TABLE candles_1m AS SELECT * FROM src.candles_1m "
                f"WHERE ts < TIMESTAMP WITH TIME ZONE '{lit['cut_tz']}'"
            )
            tables.append("candles_1m")

    for t in tables:
        cols = _table_columns(con, t)
        ts_col = "ts" if "ts" in cols else ("time" if "time" in cols else None)
        sym_col = "symbol" if "symbol" in cols else ("product_id" if "product_id" in cols else None)
        if ts_col and sym_col:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN("{ts_col}"), MAX("{ts_col}"), '
                f'COUNT(DISTINCT "{sym_col}") FROM "{t}"'
            ).fetchone()
        elif ts_col:
            n, mn, mx = con.execute(
                f'SELECT COUNT(*), MIN("{ts_col}"), MAX("{ts_col}") FROM "{t}"'
            ).fetchone()
            ns = None
        else:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            mn = mx = ns = None
        rows.append({
            "table": t, "rows": n, "symbols": ns,
            "min_ts": str(mn) if mn is not None else None,
            "max_ts": str(mx) if mx is not None else None,
        })
    con.execute("DETACH src")
    con.close()
    return rows


def freeze_stocks(out_db: Path, cutoff_utc: dt.datetime, daily_floor_utc: dt.datetime,
                  *, full: bool = False) -> list[dict]:
    lit = _cutoff_sql_literals(cutoff_utc, daily_floor_utc)
    con = duckdb.connect(str(out_db))
    con.execute(f"ATTACH '{STOCKS_DB}' AS src (READ_ONLY)")
    rows: list[dict] = []
    tables = STOCKS_FULL_TABLES if full else {"bars_1d": STOCKS_FULL_TABLES["bars_1d"]}
    for table, where_tpl in tables.items():
        where = where_tpl.format(**lit)
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM src."{table}" WHERE {where}')
        cols = _table_columns(con, table)
        if "ts" in cols and "underlying" in cols:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT underlying) FROM "{table}"'
            ).fetchone()
        elif "ts" in cols and "symbol" in cols:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT symbol) FROM "{table}"'
            ).fetchone()
        elif "ts" in cols:
            n, mn, mx = con.execute(
                f'SELECT COUNT(*), MIN(ts), MAX(ts) FROM "{table}"'
            ).fetchone()
            ns = None
        elif "as_of_date" in cols:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN(as_of_date), MAX(as_of_date), COUNT(DISTINCT symbol) FROM "{table}"'
            ).fetchone()
        elif "snap_ts" in cols:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN(snap_ts), MAX(snap_ts), COUNT(DISTINCT underlying) FROM "{table}"'
            ).fetchone()
        elif "underlying" in cols:
            n, mn, mx, ns = con.execute(
                f'SELECT COUNT(*), MIN(last_updated), MAX(last_updated), COUNT(DISTINCT underlying) FROM "{table}"'
            ).fetchone()
        else:
            n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            mn = mx = ns = None
        rows.append({"table": table, "rows": n, "symbols": ns,
                     "min_ts": str(mn), "max_ts": str(mx)})
    con.execute("DETACH src")
    con.close()
    return rows


def freeze_futures_lake(out_db: Path, src_db: Path, cutoff_utc: dt.datetime) -> list[dict]:
    lit = _cutoff_sql_literals(cutoff_utc, cutoff_utc)
    con = duckdb.connect(str(out_db))
    con.execute(f"ATTACH '{src_db}' AS src (READ_ONLY)")
    rows: list[dict] = []
    con.execute(
        f"CREATE TABLE bars_1m AS SELECT * FROM src.bars_1m "
        f"WHERE ts < TIMESTAMP WITH TIME ZONE '{lit['cut_tz']}'"
    )
    con.execute("CREATE TABLE ingest_state AS SELECT * FROM src.ingest_state")
    for table in ("bars_1m", "ingest_state"):
        if table == "bars_1m":
            n, mn, mx, ns, ne = con.execute(
                "SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT symbol), "
                "COUNT(DISTINCT expiry) FROM bars_1m"
            ).fetchone()
            rows.append({
                "table": table, "rows": n, "symbols": ns, "expiries": ne,
                "min_ts": str(mn), "max_ts": str(mx),
            })
        else:
            n, ns = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM ingest_state"
            ).fetchone()
            rows.append({"table": table, "rows": n, "symbols": ns,
                         "min_ts": None, "max_ts": None})
    con.execute("DETACH src")
    con.close()
    return rows


def freeze_indices(out_db: Path, src_db: Path, cutoff_date: str) -> list[dict]:
    """Freeze Cboe index daily levels through ``cutoff_date`` (inclusive).

    Index bars are calendar-day stamps at naive midnight; compare on date, not
    the UTC instant cutoff used for intraday lakes.
    """
    con = duckdb.connect(str(out_db))
    con.execute(f"ATTACH '{src_db}' AS src (READ_ONLY)")
    con.execute(
        "CREATE TABLE bars_1d AS SELECT * FROM src.bars_1d "
        f"WHERE CAST(ts AS DATE) <= DATE '{cutoff_date}'"
    )
    n, mn, mx, ns = con.execute(
        "SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT symbol) FROM bars_1d"
    ).fetchone()
    con.execute("DETACH src")
    con.close()
    return [{"table": "bars_1d", "rows": n, "symbols": ns, "min_ts": str(mn), "max_ts": str(mx)}]


def freeze_daily_db(src_db: Path, out_db: Path, table: str, cutoff_utc: dt.datetime) -> list[dict]:
    # ETF/stocks daily bars are tz-naive (US daily). Compare against the naive
    # UTC wall time of the cutoff.
    cut_naive = cutoff_utc.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    con = duckdb.connect(str(out_db))
    con.execute(f"ATTACH '{src_db}' AS src (READ_ONLY)")
    con.execute(
        f'CREATE TABLE "{table}" AS SELECT * FROM src."{table}" '
        f"WHERE ts < TIMESTAMP '{cut_naive}'"
    )
    n, mn, mx, ns = con.execute(
        f'SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT symbol) FROM "{table}"'
    ).fetchone()
    con.execute("DETACH src")
    con.close()
    return [{"table": table, "rows": n, "symbols": ns, "min_ts": str(mn), "max_ts": str(mx)}]


def freeze_futures(out_dir: Path, cutoff_utc: dt.datetime) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym, src in FUTURES_PARQUET.items():
        if not src.exists():
            rows.append({"symbol": sym, "source": str(src), "status": "MISSING"})
            continue
        df = pd.read_parquet(src)
        before = len(df)
        df = df[df["ts"] < cutoff_utc].copy()
        dest = out_dir / src.name.replace(".parquet", f"_{sym}.parquet") if src.name == "bars_1m.parquet" else out_dir / src.name
        # keep stable, unambiguous names
        dest = out_dir / f"{sym}_{src.name}"
        if dest.exists():
            os.chmod(dest, 0o644); dest.unlink()
        df.to_parquet(dest, index=False)
        make_readonly(dest)
        rows.append({"symbol": sym, "source": str(src), "file": dest.name,
                     "rows": len(df), "rows_dropped": before - len(df),
                     "min_ts": str(df["ts"].min()), "max_ts": str(df["ts"].max()),
                     "sha256": sha256(dest)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff-date", default="2026-06-03", help="Last day to include (YYYY-MM-DD).")
    ap.add_argument("--tz", default="America/Los_Angeles", help="Timezone for end-of-day cutoff.")
    # Default to a synced (non-Dropbox-ignored) location so the immutable
    # snapshots are backed up, unlike the ignored data/ lake dir.
    ap.add_argument("--out-dir", default="/Users/russellfloyd/Dropbox/NRT/nrt_dev/frozen_snapshots",
                    help="Parent dir; a <cutoff-date> subdir is created.")
    ap.add_argument("--include", default=None,
                    help="Comma-separated datasets (default: crypto,etf,stocks,futures[,futures_lake]).")
    ap.add_argument("--include-1m", action="store_true",
                    help="Freeze crypto candles_1m (~12 GB filtered). Default on with --full.")
    ap.add_argument("--full", action="store_true",
                    help="Freeze all crypto tables, all stocks tables, and the full futures lake.")
    ap.add_argument("--futures-lake-path", type=Path, default=None,
                    help="Source futures_market.duckdb (default data/futures_market.duckdb).")
    args = ap.parse_args()

    if args.include is None:
        args.include = (
            "crypto,etf,stocks,futures,futures_lake,indices"
            if args.full
            else "crypto,etf,stocks,futures"
        )
    if args.full and not args.include_1m:
        args.include_1m = True

    tz = ZoneInfo(args.tz)
    day = dt.date.fromisoformat(args.cutoff_date)
    # End of day = following local midnight.
    cutoff_local = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0), tzinfo=tz)
    cutoff_utc = cutoff_local.astimezone(dt.timezone.utc)
    # Last fully-completed UTC day boundary at/under the cutoff (for daily bars).
    daily_floor_utc = cutoff_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    if daily_floor_utc > cutoff_utc:
        daily_floor_utc -= dt.timedelta(days=1)

    include = {x.strip() for x in args.include.split(",") if x.strip()}
    out = Path(args.out_dir) / args.cutoff_date
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest.setdefault("datasets", {})
        manifest.setdefault("files", {})
    else:
        manifest = {
            "cutoff_date_label": args.cutoff_date,
            "cutoff_tz": args.tz,
            "cutoff_local": cutoff_local.isoformat(),
            "cutoff_utc": cutoff_utc.isoformat(),
            "crypto_daily_floor_utc": daily_floor_utc.isoformat(),
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "datasets": {},
            "files": {},
        }

    if "crypto" in include:
        db = out / "coinbase_crypto_ohlcv_lake.duckdb"
        if db.exists():
            os.chmod(db, 0o644); db.unlink()
        print(f"freezing crypto -> {db}")
        manifest["datasets"]["crypto"] = {
            "source": str(CRYPTO_DB),
            "tables": freeze_crypto(
                db, cutoff_utc, daily_floor_utc, args.include_1m, full=args.full
            ),
            "full": args.full,
        }
        make_readonly(db)
        manifest["files"][db.name] = {"sha256": sha256(db), "bytes": db.stat().st_size}

    if "etf" in include:
        db = out / "etf_market.duckdb"
        if db.exists():
            os.chmod(db, 0o644); db.unlink()
        print(f"freezing etf -> {db}")
        manifest["datasets"]["etf"] = {"source": str(ETF_DB),
                                       "tables": freeze_daily_db(ETF_DB, db, "bars_1d", cutoff_utc)}
        make_readonly(db)
        manifest["files"][db.name] = {"sha256": sha256(db), "bytes": db.stat().st_size}

    if "stocks" in include:
        db = out / "stocks_market.duckdb"
        if db.exists():
            os.chmod(db, 0o644); db.unlink()
        print(f"freezing stocks -> {db}")
        manifest["datasets"]["stocks"] = {
            "source": str(STOCKS_DB),
            "tables": freeze_stocks(db, cutoff_utc, daily_floor_utc, full=args.full),
            "full": args.full,
        }
        make_readonly(db)
        manifest["files"][db.name] = {"sha256": sha256(db), "bytes": db.stat().st_size}

    if "futures" in include:
        fdir = out / "futures_continuous"
        print(f"freezing futures continuous parquets -> {fdir}")
        manifest["datasets"]["futures_continuous"] = {"parquet": freeze_futures(fdir, cutoff_utc)}

    if "futures_lake" in include:
        src_futures = args.futures_lake_path or FUTURES_DB
        db = out / "futures_market.duckdb"
        if db.exists():
            os.chmod(db, 0o644); db.unlink()
        print(f"freezing futures lake ({src_futures}) -> {db}")
        manifest["datasets"]["futures_lake"] = {
            "source": str(src_futures),
            "tables": freeze_futures_lake(db, src_futures, cutoff_utc),
        }
        make_readonly(db)
        manifest["files"][db.name] = {"sha256": sha256(db), "bytes": db.stat().st_size}

    if "indices" in include:
        db = out / "indices_market.duckdb"
        if not INDICES_DB.exists():
            raise FileNotFoundError(
                f"indices lake missing at {INDICES_DB}; run "
                "python -m scripts.research.cboe_indices.ingest first"
            )
        if db.exists():
            os.chmod(db, 0o644); db.unlink()
        print(f"freezing indices -> {db}")
        manifest["datasets"]["indices"] = {
            "source": str(INDICES_DB),
            "tables": freeze_indices(db, INDICES_DB, args.cutoff_date),
            "note": "Cboe vol + correlation index daily levels (~40 symbols)",
        }
        make_readonly(db)
        manifest["files"][db.name] = {"sha256": sha256(db), "bytes": db.stat().st_size}

    # Write manifest (json + markdown).
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
    _write_md(out / "MANIFEST.md", manifest)
    print(f"\nSnapshot written to: {out}")
    print(f"Manifest: {out / 'MANIFEST.json'}")


def _write_md(path: Path, m: dict) -> None:
    lines = [
        f"# Frozen dataset snapshot — {m['cutoff_date_label']}",
        "",
        f"- **Cutoff (label):** end of {m['cutoff_date_label']} ({m['cutoff_tz']})",
        f"- **Cutoff (local):** {m['cutoff_local']}",
        f"- **Cutoff (UTC):** {m['cutoff_utc']}",
        f"- **Crypto daily floor (UTC):** {m['crypto_daily_floor_utc']} (forming day excluded)",
        f"- **Generated (UTC):** {m['generated_at_utc']}",
        "",
        "## Datasets",
    ]
    for name, d in m["datasets"].items():
        lines.append(f"\n### {name}")
        if "source" in d:
            lines.append(f"- source: `{d['source']}`")
        if d.get("note"):
            lines.append(f"- note: {d['note']}")
        for t in d.get("tables", []):
            sym = t.get("symbols")
            sym_s = f"{sym} symbols" if sym is not None else "n/a symbols"
            extra = f", {t['expiries']} expiries" if t.get("expiries") else ""
            ts_rng = f"ts {t['min_ts']} → {t['max_ts']}" if t.get("min_ts") else "no ts"
            lines.append(f"  - `{t['table']}`: {t['rows']:,} rows, {sym_s}{extra}, {ts_rng}")
        for p in d.get("parquet", []):
            if p.get("status") == "MISSING":
                lines.append(f"  - {p['symbol']}: MISSING source")
            else:
                lines.append(f"  - {p['symbol']} → `{p['file']}`: {p['rows']:,} rows "
                             f"(dropped {p['rows_dropped']}), ts {p['min_ts']} → {p['max_ts']}")
    lines += ["", "## Usage", "",
              "```python",
              "from convexity_pipeline.adapters.data_provider import LakeDataProvider",
              f"prov = LakeDataProvider(data_root=\"{path.parent}\",",
              "                        futures_parquet={"]
    for sym, p in FUTURES_PARQUET.items():
        fname = f"{sym}_{p.name}"
        lines.append(f"                            '{sym}': '{path.parent}/futures_continuous/{fname}',")
    lines += ["                        })",
              "df = prov.get_bars('BTC-USD', '1d')",
              "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
