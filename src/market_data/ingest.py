"""One way to fetch prices, and a record of what the vendor said when.

    fetch()      -> whole history, at one instant, with its corporate actions
    unexplained()-> moves no split or dividend accounts for
    reconcile()  -> how a new fetch differs from an old one, and why

**A vendor that adjusts retroactively has no stable history.** Yahoo returns
NVDA at 121 for 2024-06-06 today and returned 1209 for the same date before
the 2024 split — and not only under `auto_adjust=True`: the unadjusted
`Close` is split-adjusted too. There is no raw price to ask for. The same
(ticker, date) is simply a different number depending on when you asked.

Three consequences shape everything here.

**Never append.** A series assembled from two fetches that straddle a split
contains a 10x cliff that is an artifact of the assembly, not of the market.
The existing `src/data_loader.download_prices` avoids this by re-downloading
the whole range and overwriting — but it then has the second problem instead.

**Never re-fetch silently.** Overwriting is worse than a cliff in one respect:
it is invisible. A run stored last month cited prices that this month's fetch
no longer reports, and nothing anywhere says so. For a system whose figures
carry a content digest and an execution-input digest, a source that quietly
changes history is the one thing those digests cannot survive.

**So a fetch is a dated, immutable snapshot**, and a later fetch is compared
against it rather than replacing it. Differences a corporate action explains
are expected; differences it does not are a vendor revision, and the operator
has to see them rather than inherit them.

Guidance on this is settled and external — keep the vendor export, keep the
corporate-action tables beside it, version the processed set with its fetch
date, and check for extreme moves that indicate a missing adjustment. What is
specific to us is that the check must attribute a move to an action rather
than threshold it: a bare threshold flags every VIX and Bitcoin session and is
therefore ignored within a week.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

#: Retries and backoff, carried over from `data_loader._download_ticker`,
#: which had them and which this replaces for the snapshot path.
# Retries/backoff on the vendor pull. Raised for the full S&P 500 + funds
# (544 tickers): a larger universe is a larger surface for a transient Yahoo
# hiccup, so give the batch more attempts before the drop-tolerance guard in
# build_catalog_snapshot decides a name is genuinely missing for the day.
RETRIES = 5
BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

#: A single-session move this large is worth explaining. It is not evidence of
#: anything on its own — see `unexplained`, which requires that no corporate
#: action accounts for it before saying a word.
MOVE_THRESHOLD = 0.35


@dataclass(frozen=True)
class PanelFetch:
    """One whole-history fetch, and what the vendor said about it at the time.

    Two price series, because they answer different questions and using either
    for the other's question is wrong in a way that looks plausible:

        market   Close with auto_adjust=False -- split-adjusted, dividends NOT
                 removed. This is what a share was worth. Value a holding with
                 it.
        total    Adj Close -- splits and dividends. This is what an investment
                 returned. Measure a strategy with it.

    Valuing a position with `total` silently credits reinvested dividends into
    the share price and then credits them again as cash. Measuring performance
    with `market` drops the dividends entirely. Neither error announces itself.
    """

    prices: pd.DataFrame
    splits: Mapping[str, pd.Series]
    dividends: Mapping[str, pd.Series]
    fetched_at: datetime
    total_return: Optional[pd.DataFrame] = None
    source: str = "yahoo"
    adjustment: str = "split_adjusted_close"
    notes: Sequence[str] = field(default_factory=tuple)


def fetch(tickers: Sequence[str], *, start: str, end: Optional[str] = None,
          session_reference: Optional[str] = None,
          settled_only: bool = True) -> PanelFetch:
    """Fetch the whole history for every ticker, once, with its actions.

    `start` is not a resumption point. It is the beginning of the history this
    snapshot covers, and the entire range is requested every time — a fetch
    that begins where the last one stopped is the append this module exists to
    prevent.
    """
    import yfinance

    # auto_adjust=False so both series come back: `Close` is split-adjusted
    # only, `Adj Close` carries dividends as well.
    frame = _download(tickers, start=start, end=end)
    levels = frame.columns.get_level_values(0)
    close = frame["Close"] if "Close" in levels else frame
    total = frame["Adj Close"] if "Adj Close" in levels else None
    close = close.dropna(how="all").sort_index()
    if total is not None:
        total = total.reindex(close.index)

    if session_reference:
        if session_reference not in close.columns:
            raise ValueError(
                f"{session_reference} defines the trading calendar and is absent")
        # Instruments that trade on weekends (BTC-USD) otherwise widen the
        # index to calendar days, on which every equity is null — and a
        # 200-row window then spans 199 days instead of the 292 that 200
        # sessions cover. The calendar is a property of the panel, not of
        # whichever ticker happened to have the longest index.
        close = close.reindex(close.index[close[session_reference].notna()])

    if settled_only:
        # Drop the session in progress. Today's bar moves until the close, so
        # a fetch at 10am and one at 3pm disagree about "today" — 36 tickers
        # differed on exactly that date the first time this was reconciled,
        # and every one of them read as an unexplained rewrite.
        #
        # A snapshot whose last row is provisional is a snapshot that cannot
        # be re-derived, which is the property everything downstream is built
        # on. Settled sessions only, and the manifest can then say the
        # coverage means something.
        today = datetime.now(timezone.utc).date()
        close = close[close.index.date < today]

    for column in close.columns:
        first = close[column].first_valid_index()
        if first is not None:
            # Forward-fill within a ticker's own life. Filling before it is
            # pricing an instrument that did not exist.
            close.loc[first:, column] = close.loc[first:, column].ffill()

    splits, dividends = {}, {}
    for ticker in tickers:
        try:
            handle = yfinance.Ticker(ticker)
            split = handle.splits
            dividend = handle.dividends
        except Exception:                                        # noqa: BLE001
            # A missing action table is not a missing price. Record the gap by
            # its absence rather than failing the fetch; `unexplained` will
            # then be unable to excuse a move for this ticker, which is the
            # safe direction to be wrong in.
            continue
        if split is not None and len(split):
            splits[ticker] = _naive_index(split)
        if dividend is not None and len(dividend):
            dividends[ticker] = _naive_index(dividend)

    if total is not None:
        total = total.reindex(close.index)
        for column in total.columns:
            first = total[column].first_valid_index()
            if first is not None:
                total.loc[first:, column] = total.loc[first:, column].ffill()

    return PanelFetch(
        prices=close,
        total_return=total,
        splits=splits,
        dividends=dividends,
        fetched_at=datetime.now(timezone.utc),
    )


def _download(tickers: Sequence[str], *, start: str, end: Optional[str]):
    import yfinance

    delay = BACKOFF_SECONDS
    for attempt in range(1, RETRIES + 1):
        try:
            frame = yfinance.download(list(tickers), start=start, end=end,
                                      auto_adjust=False, progress=False,
                                      threads=True)
            if frame is None or frame.empty:
                raise RuntimeError("empty response")
            return frame
        except Exception:                                        # noqa: BLE001
            if attempt == RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF_SECONDS)


def _naive_index(series: pd.Series) -> pd.Series:
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out = series.copy()
    out.index = index.normalize()
    return out


def unexplained(panel: PanelFetch, *, threshold: float = MOVE_THRESHOLD
                ) -> Dict[str, pd.Series]:
    """Single-session moves that no split accounts for.

    The naive version of this check thresholds the return and stops. Run
    against a real panel it flags BTC-USD at -37% and ^VIX at -36% — both
    ordinary sessions for those instruments — and a check that fires on
    healthy data every week is one nobody reads by the second week.

    So the move has to be *unexplained*: a split within a session either side
    excuses it, because that is exactly the shape a split leaves behind when
    two adjustment epochs are stitched together.
    """
    findings: Dict[str, pd.Series] = {}
    for ticker in panel.prices.columns:
        series = panel.prices[ticker].dropna()
        if len(series) < 2:
            continue
        returns = series.pct_change().dropna()
        large = returns[returns.abs() >= threshold]
        if large.empty:
            continue
        actions = panel.splits.get(ticker)
        if actions is not None and len(actions):
            dates = set(actions.index.date)
            large = large[[
                not _near(stamp, dates) for stamp in large.index
            ]]
        if not large.empty:
            findings[ticker] = large
    return findings


def _dates(table) -> Sequence[date]:
    """Action dates, tolerant of a table, a mapping or nothing at all."""
    if table is None:
        return ()
    if isinstance(table, pd.Series):
        return [d.date() for d in table.index]
    if isinstance(table, Mapping):
        return [date.fromisoformat(str(k)) if not isinstance(k, date) else k
                for k in table]
    return [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in table]


def _near(stamp, dates, window: int = 1) -> bool:
    day = stamp.date()
    return any(abs((day - one).days) <= window for one in dates)


@dataclass(frozen=True)
class Reconciliation:
    """How a new fetch differs from the snapshot already on disk."""

    changed: Dict[str, int]
    explained: Dict[str, int]
    unexplained_changes: Dict[str, pd.Index]
    added_tickers: Sequence[str]
    removed_tickers: Sequence[str]

    @property
    def clean(self) -> bool:
        return not any(len(v) for v in self.unexplained_changes.values())


#: Relative difference below which two fetches are the same fetch.
#:
#: 1e-6 was the first choice and it sat *below the vendor's own noise floor*:
#: re-fetching minutes apart moved 110 values by up to 1.2e-6, purely from
#: adjustment arithmetic and parquet round-tripping. A comparator that fires on
#: identical data is one whose output nobody can act on.
PRICE_TOLERANCE = 1e-4


def reconcile(previous: pd.DataFrame, panel: PanelFetch, *,
              tolerance: float = PRICE_TOLERANCE,
              previous_splits: Optional[Mapping[str, Sequence]] = None
              ) -> Reconciliation:
    """Attribute every changed historical value to a corporate action, or flag it.

    This is the check the old path did not have. `download_prices`
    re-downloaded the whole range and overwrote the cache, which avoids the
    stitched-together cliff and replaces it with something quieter: the
    numbers a stored run cited are simply different now, and nothing says so.

    A split or dividend on or after a date legitimately rewrites every price
    before it. Anything else is the vendor revising history, and an operator
    should decide what that means rather than discover it in a figure.
    """
    changed: Dict[str, int] = {}
    explained: Dict[str, int] = {}
    unexplained_changes: Dict[str, pd.Index] = {}

    common = [c for c in panel.prices.columns if c in previous.columns]
    for ticker in common:
        old = previous[ticker].dropna()
        new = panel.prices[ticker].dropna()
        shared = old.index.intersection(new.index)
        if not len(shared):
            continue
        difference = (new.loc[shared] - old.loc[shared]).abs()
        scale = old.loc[shared].abs().clip(lower=1e-9)
        moved = shared[(difference / scale) > tolerance]
        if not len(moved):
            continue
        changed[ticker] = len(moved)

        # Only a split that is *new since the previous snapshot* explains a
        # rewrite.
        #
        # The first version excused any change predating the latest corporate
        # action of any kind. Tickers pay quarterly dividends, so that excused
        # essentially all of history — 110 noise changes were reported as
        # "explained by an action", which is a green light that cannot go red.
        # An action already present when the previous snapshot was written has
        # already been applied to it and explains nothing further.
        seen = set(_dates(previous_splits.get(ticker)) if previous_splits else ())
        fresh = [d for d in _dates(panel.splits.get(ticker)) if d not in seen]
        if fresh:
            latest = max(fresh)
            still = moved[moved.date > latest]
        else:
            still = moved
        explained[ticker] = len(moved) - len(still)
        if len(still):
            unexplained_changes[ticker] = still

    return Reconciliation(
        changed=changed,
        explained=explained,
        unexplained_changes=unexplained_changes,
        added_tickers=sorted(set(panel.prices.columns) - set(previous.columns)),
        removed_tickers=sorted(set(previous.columns) - set(panel.prices.columns)),
    )


# --- as-traded, and why a share count is not a number ----------------------
#
# A holding recorded before a split is recorded in shares that no longer
# exist. Somebody who bought 10 NVDA at $1,209 in May 2024 holds 100 today,
# and their statement still says 10 at $1,209. Both are true; neither can be
# read without knowing which side of the split it was written on.
#
# So a quantity and a price each carry an adjustment epoch, and the thing that
# survives the split is neither of them:
#
#     as_traded_shares x as_traded_price  ==  current_shares x current_price
#
# Value is the invariant. Store what the statement said — 10 at $1,209, on
# that date — and convert at read time through a factor derived from the
# corporate-action table. Rewriting the stored quantity to 100 would destroy
# the only record of what the user actually did, to save a multiplication.


def split_factor(splits: Optional[pd.Series], when: date) -> float:
    """How many of today's shares one share held on `when` became.

    The product of every split strictly after that date. A share bought before
    NVDA's 10:1 is ten shares now, so the factor is 10; a share bought after
    it is one share, and the factor is 1.
    """
    if splits is None or not len(splits):
        return 1.0
    factor = 1.0
    for stamp, ratio in splits.items():
        stamp_date = stamp.date() if hasattr(stamp, "date") else stamp
        if stamp_date > when:
            factor *= float(ratio)
    return factor


def as_traded_price(split_adjusted: float, splits: Optional[pd.Series],
                    when: date) -> float:
    """The price actually printed on the day.

    No vendor here reports it — Yahoo's `Close` is split-adjusted even with
    `auto_adjust=False` — but it is recoverable, and a user comparing a figure
    against their own statement needs the number they remember.
    """
    return split_adjusted * split_factor(splits, when)


def current_shares(as_traded_shares: float, splits: Optional[pd.Series],
                   when: date) -> float:
    """What a quantity recorded on `when` is worth in today's share count."""
    return as_traded_shares * split_factor(splits, when)
