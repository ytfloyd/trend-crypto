"""Round-trip + upsert tests for ``volbook.bundle``."""
from __future__ import annotations

from pathlib import Path

from volbook.bundle import Bar, OhlcvBundle, OhlcvSeries


def _make_series(bar_size: str = "1 day", n: int = 3) -> OhlcvSeries:
    bars = [
        Bar(
            t=f"2026-04-{i + 1:02d}T00:00:00+00:00",
            o=80.0 + i,
            h=81.0 + i,
            l=79.5 + i,
            c=80.5 + i,
            v=1000.0 + i,
        )
        for i in range(n)
    ]
    return OhlcvSeries(
        symbol="CL",
        expiry="202606",
        exchange="NYMEX",
        bar_size=bar_size,
        duration="1 Y",
        what_to_show="TRADES",
        bars=bars,
        fetched_at="2026-04-24T00:00:00+00:00",
    )


def test_series_key_is_symbol_expiry_barsize() -> None:
    s = _make_series(bar_size="1 hour")
    assert s.key == "CL-202606-1 hour"
    assert s.contract_key == "CL-202606"


def test_bundle_upsert_replaces_matching_key() -> None:
    bundle = OhlcvBundle()
    original = _make_series(bar_size="1 day", n=3)
    bundle.upsert(original)
    assert len(bundle.series) == 1

    replacement = _make_series(bar_size="1 day", n=5)
    bundle.upsert(replacement)
    assert len(bundle.series) == 1, "upsert must replace in-place, not append"
    assert len(bundle.series[0].bars) == 5


def test_bundle_upsert_preserves_other_series() -> None:
    bundle = OhlcvBundle()
    bundle.upsert(_make_series(bar_size="1 day"))
    bundle.upsert(_make_series(bar_size="1 hour"))
    assert sorted(bundle.keys()) == ["CL-202606-1 day", "CL-202606-1 hour"]

    # Replacing one should leave the other untouched.
    bundle.upsert(_make_series(bar_size="1 day", n=10))
    daily = next(s for s in bundle.series if s.bar_size == "1 day")
    hourly = next(s for s in bundle.series if s.bar_size == "1 hour")
    assert len(daily.bars) == 10
    assert len(hourly.bars) == 3


def test_bundle_json_round_trip(tmp_path: Path) -> None:
    bundle = OhlcvBundle()
    bundle.upsert(_make_series(bar_size="1 day"))
    bundle.upsert(_make_series(bar_size="1 hour"))

    path = tmp_path / "bundle.json"
    bundle.save(path)
    assert path.exists()

    loaded = OhlcvBundle.load(path)
    assert loaded.keys() == bundle.keys()
    assert loaded.generated_at  # stamped on save
    for original, reloaded in zip(bundle.series, loaded.series):
        assert original.symbol == reloaded.symbol
        assert original.expiry == reloaded.expiry
        assert original.bar_size == reloaded.bar_size
        assert len(original.bars) == len(reloaded.bars)
        assert original.bars[0] == reloaded.bars[0]


def test_bundle_load_missing_file_returns_empty(tmp_path: Path) -> None:
    empty = OhlcvBundle.load(tmp_path / "does_not_exist.json")
    assert empty.series == []
