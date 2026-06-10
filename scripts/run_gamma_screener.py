#!/usr/bin/env python3
"""End-of-day underpriced-gamma equity screener.

Flow:
    1. Resolve the ticker universe (sp100 / sp500 / custom).
    2. (Optionally) snapshot each ticker's option surface via IB into
       ``vol_surface_snaps`` in the stocks DuckDB.
    3. Compute per-symbol features (constant-maturity IV, RV, skew, term).
    4. Fetch upcoming earnings dates (Finnhub, optional).
    5. Score cross-sectionally and rank (scoped to symbols snapped today).
    6. Write to ``gamma_screener_daily`` DuckDB table and a CSV.
    7. Print the top-N to the console.

Examples:
    # Full run (snapshot + score) on S&P 100
    python scripts/run_gamma_screener.py --universe sp100

    # Skip snapshotting, just re-score from already-captured surfaces
    python scripts/run_gamma_screener.py --universe sp100 --skip-ingest

    # Dry run — no IB, no DB writes, no CSV; validates imports and config
    python scripts/run_gamma_screener.py --dry-run

    # Small test
    python scripts/run_gamma_screener.py --tickers AAPL,MSFT,NVDA,SPY

Operational cadence — SP500 chunked population
----------------------------------------------
A full SP500 pass (488 names) takes ~20h serial on one IB client, which
does not fit inside a single TWS session. Split it across ~4 post-close
evenings, one chunk per night (kick off at ~16:05 ET, finishes by ~21:00):

    python scripts/run_gamma_screener.py --universe sp500 --chunk 1/4
    python scripts/run_gamma_screener.py --universe sp500 --chunk 2/4
    python scripts/run_gamma_screener.py --universe sp500 --chunk 3/4
    python scripts/run_gamma_screener.py --universe sp500 --chunk 4/4

Each chunk ranks cross-sectionally within itself (per-chunk scope),
lands one row per symbol in ``gamma_screener_daily``, and appends to
the durable ``vol_surface_snaps`` history.

If TWS drops mid-run, just re-run the exact same command. The ingest
loop skips tickers already snapped today (UTC) and picks up where it
left off. Use ``--force-resnap`` to override the resume and re-snap
everything.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.logging import setup_logging, get_logger  # noqa: E402
from data.options.schema import OptionsSchema  # noqa: E402
from screeners.gamma.config import GammaScreenerConfig  # noqa: E402
from screeners.gamma.earnings import fetch_earnings_in_window  # noqa: E402
from screeners.gamma.ingest import snapshot_universe  # noqa: E402
from screeners.gamma.schema import GammaScreenerSchema  # noqa: E402
from screeners.gamma.signals import compute_features  # noqa: E402
from screeners.gamma.score import rank_universe, ScoredRow  # noqa: E402
from screeners.gamma.universe import get_universe  # noqa: E402

logger = get_logger("run_gamma_screener")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Underpriced-gamma equity screener")
    p.add_argument("--universe", default="sp100", choices=["sp100", "sp500"])
    p.add_argument("--tickers", default=None, help="Comma-separated override (beats --universe)")
    p.add_argument("--chunk", default=None,
                   help="Slice the resolved universe into N chunks and run the Mth. "
                        "Format 'M/N', 1-indexed (e.g. '1/4'). Ignored with --tickers.")
    p.add_argument("--db", default=None, help="Override DB path (default from config)")
    p.add_argument("--port", type=int, default=None, help="Override IB port")
    p.add_argument("--client-id", type=int, default=None, help="Override IB client_id")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip IB snapshotting; score from existing vol_surface_snaps")
    p.add_argument("--skip-earnings", action="store_true",
                   help="Skip Finnhub earnings lookup")
    p.add_argument("--force-resnap", action="store_true",
                   help="Re-snap every ticker even if already present for today. "
                        "Default is resume-friendly (skip already-snapped).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit; no IB, no DB writes")
    p.add_argument("--top-n", type=int, default=25, help="Top rows to print/save")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--min-slices", type=int, default=None,
                   help="Override min valid VolSlices per symbol (default from config). "
                        "Lower this (e.g. 1) for post-close smoke tests where quote depth is thin.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _parse_chunk_spec(spec: str, n_tickers: int) -> tuple[int, int, int, int]:
    """Parse 'M/N' into (chunk_idx, chunk_total, start, stop) slice indices.

    Raises ValueError on malformed input. chunk_idx is 1-indexed.
    """
    try:
        m_str, n_str = spec.split("/", 1)
        m = int(m_str.strip())
        n = int(n_str.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"--chunk expects 'M/N' format, got {spec!r}")
    if n <= 0 or m <= 0 or m > n:
        raise ValueError(f"--chunk M/N requires 1 <= M <= N, got {spec!r}")
    stride = math.ceil(n_tickers / n)
    start = (m - 1) * stride
    stop = min(m * stride, n_tickers)
    return m, n, start, stop


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.chunk:
            logger.warning("--chunk is ignored when --tickers is given")
        return tickers

    tickers = list(get_universe(args.universe, PROJECT_ROOT))
    if args.chunk:
        m, n, start, stop = _parse_chunk_spec(args.chunk, len(tickers))
        sliced = tickers[start:stop]
        if sliced:
            logger.info(
                "Chunk %d/%d of %s: %s…%s (%d tickers, index %d:%d of %d)",
                m, n, args.universe, sliced[0], sliced[-1],
                len(sliced), start, stop, len(tickers),
            )
        else:
            logger.warning("Chunk %d/%d of %s is empty", m, n, args.universe)
        return sliced
    return tickers


def _build_config(args: argparse.Namespace) -> GammaScreenerConfig:
    kwargs: dict[str, object] = {"report_top_n": args.top_n}
    if args.db:
        kwargs["stocks_db_path"] = args.db
    if args.port is not None:
        kwargs["ib_port"] = args.port
    if args.client_id is not None:
        kwargs["ib_client_id"] = args.client_id
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.min_slices is not None:
        kwargs["min_slices_required"] = args.min_slices
    return GammaScreenerConfig(**kwargs)


def _ensure_schemas(db_path: str) -> None:
    import duckdb
    conn = duckdb.connect(db_path)
    OptionsSchema(conn).ensure_tables()
    GammaScreenerSchema(conn).ensure_tables()
    conn.close()


def _symbols_snapped_today(
    db_path: str,
    tickers: list[str],
    as_of_date: date,
) -> list[str]:
    """Return the subset of ``tickers`` that have a vol_surface_snaps row today.

    Used to scope scoring to fresh surfaces only — prevents stale snaps
    from prior days bleeding into a chunk's cross-sectional rank when
    ingest partially fails.
    """
    if not tickers:
        return []
    import duckdb
    start_ts = datetime.combine(as_of_date, datetime.min.time(), tzinfo=timezone.utc)
    end_ts = start_ts + timedelta(days=1)
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ",".join(["?"] * len(tickers))
        rows = conn.execute(
            f"""
            SELECT DISTINCT underlying
            FROM vol_surface_snaps
            WHERE underlying IN ({placeholders})
              AND snap_ts >= ?
              AND snap_ts < ?
            """,
            [*tickers, start_ts, end_ts],
        ).fetchall()
    finally:
        conn.close()
    fresh = {r[0] for r in rows}
    return [t for t in tickers if t in fresh]


def _print_top(scored: list[ScoredRow], top_n: int) -> None:
    ranked = [r for r in scored if r.rank_combined is not None]
    ranked.sort(key=lambda r: r.rank_combined or 10**9)
    ranked = ranked[:top_n]
    if not ranked:
        print("\n  No scored candidates.\n")
        return

    def _pct(x: Optional[float]) -> str:
        return f"{x*100:5.1f}%" if x is not None else "   n/a"

    def _f(x: Optional[float]) -> str:
        return f"{x:+.2f}" if x is not None else "  n/a"

    print(f"\n{'─'*96}")
    print(f"  Top {len(ranked)} underpriced-gamma candidates")
    print(f"{'─'*96}")
    print(f"  {'Rank':>4}  {'Symbol':<7}  {'Spot':>8}  "
          f"{'IV30':>6}  {'RV20':>6}  {'IV/RV':>6}  "
          f"{'Skew':>6}  {'T30-90':>7}  {'Earn':>4}  {'Score':>6}")
    for r in ranked:
        print(
            f"  {r.rank_combined:>4}  {r.symbol:<7}  ${r.spot:>7.2f}  "
            f"{_pct(r.iv30)}  {_pct(r.rv_yz20)}  "
            f"{(r.iv30_rv20_ratio or 0):>5.2f}  "
            f"{_pct(r.skew_25d_30)}  {_pct(r.term_30_90)}  "
            f"{'YES' if r.earnings_in_window else ' no':>4}  "
            f"{_f(r.score_combined)}"
        )
    print(f"{'─'*96}\n")


def _write_csv(scored: list[ScoredRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not scored:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].to_dict().keys()))
        writer.writeheader()
        for r in scored:
            writer.writerow(r.to_dict())
    logger.info("Wrote %d rows to %s", len(scored), output_path)


def _persist(scored: list[ScoredRow], cfg: GammaScreenerConfig) -> int:
    import duckdb
    conn = duckdb.connect(cfg.stocks_db_path)
    GammaScreenerSchema(conn).ensure_tables()
    rows = [r.to_dict() for r in scored]
    n = GammaScreenerSchema(conn).upsert_rows(rows)
    conn.close()
    return n


def _score_and_persist(
    tickers: list[str],
    cfg: GammaScreenerConfig,
    as_of: date,
    skip_earnings: bool,
) -> int:
    """Run earnings lookup → features → score → persist → CSV → top-N print.

    Scopes the work to tickers that actually have a ``vol_surface_snaps``
    row for ``as_of``. Called from main() in a ``finally`` block so that
    a mid-ingest crash still surfaces scores for whatever made it in.
    Returns a process exit code (0 on any rows persisted, 1 otherwise).
    """
    fresh = _symbols_snapped_today(cfg.stocks_db_path, tickers, as_of)
    if not fresh:
        logger.error("No symbols with a snap for %s — nothing to score", as_of)
        return 1
    if len(fresh) < len(tickers):
        logger.info(
            "Scoring %d/%d requested tickers (others lack a snap for %s)",
            len(fresh), len(tickers), as_of,
        )

    earnings_hits: set[str] = set()
    if not skip_earnings:
        earnings_hits = fetch_earnings_in_window(
            fresh,
            as_of=as_of,
            lookahead_days=cfg.earnings_lookahead_days,
        )

    logger.info("Computing features…")
    features = compute_features(
        symbols=fresh,
        cfg=cfg,
        as_of_date=as_of,
        earnings_symbols_in_window=earnings_hits,
    )

    if not features:
        logger.error("No features produced — check IB snapshot and bars_1d coverage")
        return 1

    logger.info("Scoring %d symbols…", len(features))
    scored = rank_universe(features, cfg)

    n_persisted = _persist(scored, cfg)
    logger.info("Persisted %d rows to gamma_screener_daily", n_persisted)

    output_dir = Path(cfg.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    csv_path = output_dir / f"gamma_screener_{as_of:%Y%m%d}.csv"
    _write_csv(scored, csv_path)

    _print_top(scored, cfg.report_top_n)
    return 0 if n_persisted > 0 else 1


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    tickers = _resolve_tickers(args)
    cfg = _build_config(args)
    as_of = datetime.now(timezone.utc).date()

    print(f"\n{'═'*60}")
    print("  Underpriced-Gamma Equity Screener")
    print(f"{'═'*60}")
    print(f"  As-of date : {as_of}")
    print(f"  Universe   : {args.universe} ({len(tickers)} tickers)")
    if args.chunk:
        print(f"  Chunk      : {args.chunk}")
    print(f"  DB path    : {cfg.stocks_db_path}")
    print(f"  IB         : {cfg.ib_host}:{cfg.ib_port} (client={cfg.ib_client_id})")
    print(f"  Skip ingest: {args.skip_ingest}")
    print(f"  Force resnap: {args.force_resnap}")
    print(f"  Dry run    : {args.dry_run}")
    print()

    if args.dry_run:
        print("  Dry run — exiting before any IB or DB calls.")
        return 0

    if not tickers:
        logger.error("No tickers resolved — nothing to do.")
        return 1

    _ensure_schemas(cfg.stocks_db_path)

    ingest_interrupted = False
    if not args.skip_ingest:
        logger.info("Starting IB snapshot ingestion…")
        try:
            result = snapshot_universe(
                tickers, cfg,
                force_resnap=args.force_resnap,
                as_of_date=as_of,
            )
            logger.info(
                "Ingest: %d ok, %d failed, %.1f min",
                len(result.succeeded), len(result.failed),
                result.elapsed_secs / 60.0,
            )
            if result.failed:
                for sym in sorted(result.failed):
                    logger.warning("Failed %s: %s", sym, result.failed[sym])
        except KeyboardInterrupt:
            ingest_interrupted = True
            logger.warning(
                "Ingest interrupted by user — proceeding to score whatever "
                "has been persisted to vol_surface_snaps today."
            )
        except Exception as exc:  # noqa: BLE001
            ingest_interrupted = True
            logger.exception(
                "Ingest crashed (%s) — proceeding to score whatever has been "
                "persisted to vol_surface_snaps today.", exc,
            )
    else:
        logger.info("Skipping IB ingestion (--skip-ingest)")

    rc = _score_and_persist(
        tickers=tickers,
        cfg=cfg,
        as_of=as_of,
        skip_earnings=args.skip_earnings,
    )
    if ingest_interrupted and rc == 0:
        logger.warning(
            "Exiting with rc=2 because ingest was interrupted; rerun the same "
            "command to resume remaining tickers."
        )
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
