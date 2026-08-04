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
MAPPING = REPO / "data" / "catalog_instruments.yaml"

#: The ticker whose trading days define the calendar. Every other series is
#: reindexed onto it. A liquid, continuously-listed US equity ETF is the right
#: choice; anything that trades on weekends is the wrong one.
SESSION_REFERENCE = "SPY"


def tickers_from(mapping: dict) -> list[str]:
    """Every ticker the catalog needs, from the mapping and the aliases.

    Derived, not restated. A second hand-written list is the copy that stops
    matching the first, and the one that drifts is always the one nobody reads.
    """
    named = {t for entry in mapping["mappings"].values() for t in entry["tickers"]}
    aliased = set(mapping.get("aliases", {}).values())
    return sorted(named | aliased)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "data" / "snapshots"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--snapshot-id", default=None)
    arguments = parser.parse_args()

    import yfinance

    mapping = yaml.safe_load(MAPPING.read_text())
    tickers = tickers_from(mapping)
    print(f"{len(tickers)} tickers from {MAPPING.name}")

    frame = yfinance.download(tickers, start=arguments.start, auto_adjust=True,
                              progress=False, threads=True)
    close = frame["Close"] if "Close" in frame.columns.get_level_values(0) else frame
    close = close.dropna(how="all").sort_index()

    # --- put the frame on a trading calendar --------------------------------
    #
    # `BTC-USD` trades every calendar day. Taking the union of every ticker's
    # index therefore produces weekends, on which every equity is NaN — 1210 of
    # them in the first build of this file. The manifest still said
    # `calendar: nyse` and counted them as sessions.
    #
    # The damage is not missing data, it is arithmetic: a 200-row window on a
    # calendar-day index spans 199 days rather than the 292 that 200 NYSE
    # sessions actually cover. Every moving average, every volatility estimate
    # and every drawdown would be computed over the wrong horizon and would
    # still look entirely reasonable.
    if SESSION_REFERENCE not in close.columns:
        print(f"{SESSION_REFERENCE} is required to define the session calendar",
              file=sys.stderr)
        return 1
    sessions = close.index[close[SESSION_REFERENCE].notna()]
    close = close.reindex(sessions)

    # Forward-fill inside each ticker's own life, never before it. A ticker
    # that listed in 2020 has no 2015 price, and inventing one by back-filling
    # would put a figure where the instrument did not exist.
    inception = {}
    for column in close.columns:
        first = close[column].first_valid_index()
        inception[column] = first
        if first is not None:
            close.loc[first:, column] = close.loc[first:, column].ffill()

    # A ticker that silently returned nothing would become a column of NaN and
    # a plan naming it would price to nothing while looking configured.
    empty = [c for c in close.columns if close[c].notna().sum() == 0]
    if empty:
        print(f"no data returned for: {empty}", file=sys.stderr)
        return 1
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        print(f"absent from the response entirely: {missing}", file=sys.stderr)
        return 1

    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    snapshot_id = arguments.snapshot_id or f"prices-catalog-{today:%Y%m%d}"
    parquet = out / f"{snapshot_id}.parquet"
    close.to_parquet(parquet)

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
