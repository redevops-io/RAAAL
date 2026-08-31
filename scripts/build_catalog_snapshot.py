"""Build a priced snapshot for the load-test catalog from real market data.

    data/catalog_instruments.yaml  ->  Yahoo  ->  parquet + manifest

The catalog names assets the way people do — "quality ETF", "60/40",
"T-bills". `catalog_instruments.yaml` resolves each to the closest real,
tradeable ticker, and this script prices exactly those tickers. Nothing here
invents a symbol or a series.

**The manifest says `licensed`, not `synthetic`, and that is the point.**
Yahoo is a vendor source with its own terms, so a snapshot built here is not
admissible under `PILOT_DATA_POLICY=SYNTHETIC_ONLY` — `pilot_policy.py` will
refuse it, by design, until all six licensing questions carry recorded
answers. Marking it `synthetic` to get past that gate would be defeating the
control rather than satisfying it, and would put unlicensed prices in front of
pilot users under a notice telling them the data was invented.

So this snapshot is for internal development and catalog validation. Serving it
is a separate, deliberate decision with a named policy version behind it.

    python scripts/build_catalog_snapshot.py --out data/snapshots
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
#: The committed universe of record — every ticker priced into the snapshot, and
#: therefore what a plan may name (the pilot's priceable set is the snapshot's own
#: columns). Committed so the daily refresh has a source that exists in a fresh
#: checkout; adding an instrument is an edit here.
UNIVERSE = REPO / "data" / "instruments" / "pilot_universe.yaml"
#: The load-test catalog. Gitignored, so present only on a machine that has it;
#: unioned in when it is, absent without complaint. `pilot_universe.yaml` is the
#: authority either way.
MAPPING = REPO / "data" / "catalog_instruments.yaml"

#: The ticker whose trading days define the calendar. Every other series is
#: reindexed onto it. A liquid, continuously-listed US equity ETF is the right
#: choice; anything that trades on weekends is the wrong one.
SESSION_REFERENCE = "SPY"


def tickers_from(mapping: dict) -> list[str]:
    """Every ticker the load-test catalog needs, from the mapping and the aliases.

    Derived, not restated. A second hand-written list is the copy that stops
    matching the first, and the one that drifts is always the one nobody reads.
    """
    named = {t for entry in mapping["mappings"].values() for t in entry["tickers"]}
    aliased = set(mapping.get("aliases", {}).values())
    return sorted(named | aliased)


def universe_tickers() -> list[str]:
    """The tickers to price: the committed universe, unioned with the load-test
    catalog when that (gitignored) file happens to be present. The committed file
    is the authority; the catalog only ever adds, never subtracts."""
    tickers: set[str] = set()
    if UNIVERSE.exists():
        doc = yaml.safe_load(UNIVERSE.read_text()) or {}
        tickers |= set(doc.get("funds_and_macro", []))
        tickers |= set(doc.get("equities", []))
    if MAPPING.exists():
        tickers |= set(tickers_from(yaml.safe_load(MAPPING.read_text())))
    if not tickers:
        raise SystemExit(
            f"no universe: neither {UNIVERSE} nor {MAPPING} yielded any ticker")
    return sorted(tickers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "snapshots"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--snapshot-id", default=None)
    arguments = parser.parse_args()

    from src.market_data import ingest

    tickers = universe_tickers()
    print(f"{len(tickers)} tickers "
          f"(committed {UNIVERSE.name}{' + ' + MAPPING.name if MAPPING.exists() else ''})")

    # One ingestion path. The retries, the calendar and the corporate-action
    # capture all live in `src.market_data.ingest`, so this script cannot
    # quietly grow a second policy for any of them.
    panel = ingest.fetch(tickers, start=arguments.start,
                         session_reference=SESSION_REFERENCE)
    close = panel.prices
    inception = {c: close[c].first_valid_index() for c in close.columns}

    empty = [c for c in close.columns if close[c].notna().sum() == 0]
    if empty:
        print(f"no data returned for: {empty}", file=sys.stderr)
        return 1
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        print(f"absent from the response entirely: {missing}", file=sys.stderr)
        return 1

    # A move no split accounts for is either a real session or a stitched
    # series. Reported, never fatal: BTC-USD and ^VIX have genuine ones.
    for ticker, moves in ingest.unexplained(panel).items():
        print(f"  unexplained move  {ticker}: "
              f"{', '.join(f'{d.date()} {v:+.1%}' for d, v in moves.items())}")

    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    snapshot_id = arguments.snapshot_id or f"prices-catalog-{today:%Y%m%d}"
    parquet = out / f"{snapshot_id}.parquet"

    # Compare against the newest snapshot already here before writing over the
    # question. A split legitimately rewrites every earlier price; anything
    # else is the vendor revising history, and a stored run cited the old
    # numbers.
    # Market series only. The first version of this glob also matched the
    # sibling `.total-return.parquet` and reconciled the market series against
    # the total-return one — 104,006 changed values, and a refusal to write.
    # The guard behaved correctly on a comparison that was meaningless.
    existing = sorted(one for one in out.glob("prices-catalog-*.parquet")
                      if not one.name.endswith(".total-return.parquet"))
    if existing:
        import pandas
        previous = pandas.read_parquet(existing[-1])
        # The previous manifest's split table, so an action already applied
        # there cannot be used to excuse a fresh rewrite.
        previous_manifest = existing[-1].with_suffix(".yaml")
        previous_splits = {}
        if previous_manifest.exists():
            stored = yaml.safe_load(previous_manifest.read_text()) or {}
            previous_splits = (stored.get("corporate_actions") or {}).get("splits") or {}
        result = ingest.reconcile(previous, panel, previous_splits=previous_splits)
        print(f"  reconciled against {existing[-1].name}: "
              f"{sum(result.changed.values())} values changed, "
              f"{sum(result.explained.values())} explained by an action")
        if not result.clean:
            for ticker, dates in result.unexplained_changes.items():
                print(f"  UNEXPLAINED REWRITE  {ticker}: {len(dates)} values, "
                      f"{dates.min().date()} -> {dates.max().date()}",
                      file=sys.stderr)
            print("  Refusing to overwrite. A stored run cited the old values.",
                  file=sys.stderr)
            return 2

    close.to_parquet(parquet)
    # The total-return series beside the market one. Both are needed and using
    # either for the other's question is wrong in a way that looks reasonable.
    if panel.total_return is not None:
        panel.total_return.to_parquet(out / f"{snapshot_id}.total-return.parquet")

    # The claim `calendar/nyse@1` has to be true of the file, not of the
    # intention. The first build asserted it while carrying 1210 weekends.
    weekends = int((close.index.dayofweek >= 5).sum())
    if weekends:
        print(f"{weekends} weekend rows survived the reindex", file=sys.stderr)
        return 1

    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "market-data/prices",
        "snapshot_id": snapshot_id,
        "kind": "licensed",
        "uri": str(parquet.relative_to(REPO)),
        "generator": "scripts/build_catalog_snapshot.py",
        "file_sha256": digest,
        "schema_version": 1,
        "calendar": "calendar/nyse@1",
        "sessions": int(len(close)),
        "assets": int(close.shape[1]),
        "data_as_of": today.isoformat(),
        "coverage": {"start": close.index.min().date().isoformat(),
                     "end": close.index.max().date().isoformat()},
        "provider": "yahoo",
        "fetched_at": panel.fetched_at.isoformat(timespec="seconds"),
        "adjustment": panel.adjustment,
        "series": {
            "market": f"{snapshot_id}.parquet",
            "total_return": f"{snapshot_id}.total-return.parquet",
            "note": ("market = split-adjusted Close, values a holding. "
                     "total_return = Adj Close, measures a strategy. "
                     "As-traded prices are market x the split factor after "
                     "the date; see src/market_data/ingest.as_traded_price."),
        },
        # The vendor's own corporate-action tables, kept beside the prices.
        # They are what makes a later fetch's differences explainable rather
        # than merely different.
        "corporate_actions": {
            "splits": {t: {d.date().isoformat(): float(v) for d, v in s.items()}
                       for t, s in sorted(panel.splits.items())},
            "dividend_dates": {t: len(d) for t, d in sorted(panel.dividends.items())},
        },
        "license_class": "vendor-terms",
        "redistributable": False,
        "license_review_status": "OPEN",
        # Every question still open. The loader reads this rather than trusting
        # the operator who built the file.
        "egress_policy": {
            "public_export": "DENY",
            "case_bundle": "DENY",
            "model_provider_upload": "DENY",
            "internal_benchmark": "ALLOW",
        },
        "warning": ("Vendor prices under Yahoo's terms. Not admissible under "
                    "PILOT_DATA_POLICY=SYNTHETIC_ONLY; serving this requires "
                    "the six licensing questions answered and a named policy "
                    "version."),
        "symbols": sorted(close.columns),
        # Recorded, not hidden. A backtest that starts before a ticker listed
        # is a backtest of an instrument that did not exist, and the engine
        # needs to be able to refuse it rather than price a gap.
        "inception": {c: (d.date().isoformat() if d is not None else None)
                      for c, d in sorted(inception.items())},
    }
    manifest_path = out / f"{snapshot_id}.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    print(f"{parquet}  {close.shape[0]} sessions x {close.shape[1]} assets")
    print(f"{manifest_path}")
    print(f"coverage {manifest['coverage']['start']} -> {manifest['coverage']['end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
