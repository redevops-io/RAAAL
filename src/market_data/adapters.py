"""Where observations come from, and what is known about them.

An adapter fetches and describes. It does not decide what a plan holds, does not
price anything, and has no route to the evaluator — everything it produces
becomes a `MarketSnapshot` and travels the same lifecycle as any other:

    adapter -> observations + provenance -> descriptor -> bytes -> verified read

**Adapters never write directly to evaluation.** Structural rather than
promised: this module imports nothing from `evaluation` or `workspace`, and a
test reads the import graph to say so. The reason is the one this project keeps
paying for — a second path to a figure is a path with none of the checks on it,
and it looks exactly like the first until somebody compares them.

**An adapter declares; it does not infer.** Licence class, review status,
adapter version and corporate-action treatment are stated by the adapter that
knows them. Where it does not know, it says `NOT_DECLARED` rather than choosing
a plausible value — a snapshot silent about its licence is one somebody will
assume was cleared.

**Symbols are resolved before fetching, never after.** An adapter handed
"the S&P 500" must refuse rather than fetch something: resolution is a question
about what was meant, and answering it with whatever the provider returns for
that string is the provider deciding the portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .symbols import Instrument, Resolution, resolve_all


class AdapterRefused(ValueError):
    """The adapter will not fetch, and says which input it could not use."""


@dataclass(frozen=True)
class Fetched:
    """Observations, and everything the adapter knows about them."""

    observations: Any
    instruments: Tuple[Instrument, ...]
    adapter_name: str
    adapter_version: str
    license_class: str
    license_review_status: str
    dataset_id: str
    source_uri: str
    data_as_of: str
    calendar: str

    def symbols(self) -> Tuple[str, ...]:
        return tuple(one.symbol for one in self.instruments)


@dataclass(frozen=True)
class LocalParquetAdapter:
    """The adapter this build actually has: a pinned file on disk.

    Named and versioned rather than treated as "no adapter". Two adapters
    reading one vendor can disagree about adjustment, session alignment and
    what a missing day means, so a snapshot that named the vendor and not the
    adapter would attribute a difference to the market.
    """

    name: str = "local-parquet"
    version: str = "1"

    def fetch(self, queries: Sequence[str], *, reinvested: bool = False
              ) -> Fetched:
        """Resolve the names, then fetch. Refuses on anything unresolved.

        The order matters and is the second hard rule: an adapter that fetched
        first and resolved afterwards would be asking a provider what the words
        meant, which is the provider choosing the portfolio.
        """
        from .loader import load_prices

        instruments, failed = resolve_all(queries)
        if failed:
            raise AdapterRefused(
                "these were not resolved to an instrument, and nothing is "
                "fetched for a name nobody has pinned: "
                + "; ".join(one.detail for one in failed))

        untradable = [one for one in instruments if not one.tradable]
        if untradable:
            raise AdapterRefused(
                f"{', '.join(one.symbol for one in untradable)} "
                f"{'is an index' if len(untradable) == 1 else 'are indices'} "
                "— computed from constituents and not purchasable. A backtest "
                "that bought one is a backtest of something nobody can hold")

        snapshot, allow_network = self._serving_snapshot()
        frame = load_prices(snapshot, reinvested=reinvested,
                            allow_network=allow_network)

        wanted = [one.symbol for one in instruments]
        missing = [one for one in wanted if one not in frame.columns]
        if missing:
            raise AdapterRefused(
                f"{', '.join(missing)} resolved to instruments this snapshot "
                "does not carry. That is a coverage gap, not a result")

        return Fetched(
            observations=frame[wanted],
            instruments=instruments,
            adapter_name=self.name,
            adapter_version=self.version,
            license_class=str(getattr(snapshot, "license_class", "")
                              or "NOT_DECLARED"),
            license_review_status=str(
                getattr(snapshot, "license_review_status", "")
                or "NOT_DECLARED"),
            dataset_id=str(getattr(snapshot, "dataset_id", "")
                           or "NOT_DECLARED"),
            source_uri=str(getattr(snapshot, "uri", "") or "NOT_DECLARED"),
            data_as_of=str(getattr(snapshot, "data_as_of", "")
                           or "NOT_DECLARED"),
            calendar=str(getattr(snapshot, "calendar", "") or "NOT_DECLARED"))

    def _serving_snapshot(self) -> Tuple[Any, bool]:
        """The snapshot this deployment's data policy permits, and whether
        loading it may reach the network.

        The adapter used to hardcode the synthetic fixture, so a deployment on
        the approved vendor policy still served invented series while its banner
        promised vendor data. The policy is read here — the one place that turns
        a snapshot into observations — so the figures and the disclosure can
        never disagree about where the numbers came from.

        **Fails closed.** Under the vendor policy with no approved snapshot
        (missing manifest, missing or incomplete licensing record) it refuses
        rather than falling back to synthetic. A silent fallback is the exact
        substitution this project forbids everywhere else: a run whose figures
        come from data the policy did not name.
        """
        from ..deploy.context import current
        from .access import approved_snapshot
        from .loader import synthetic_snapshot
        from .pilot_policy import PilotDataPolicy

        policy = current().market_data.policy
        if policy is PilotDataPolicy.PILOT_VENDOR_APPROVED:
            snapshot = approved_snapshot()
            if snapshot is None:
                raise AdapterRefused(
                    "the deployment's data policy is the approved vendor "
                    "policy, but no approved snapshot resolved — a missing "
                    "manifest, or a licensing record whose answers are not all "
                    "recorded. Refusing rather than serving synthetic figures "
                    "under a policy that promises vendor data.")
            # A vendor snapshot is S3-backed; loading fetches and verifies it.
            return snapshot, True
        return synthetic_snapshot(), False


def snapshot_from(fetched: Fetched, *, reinvested: bool):
    """Turn what an adapter produced into the contract everything else speaks.

    The only way out of this module. An adapter's `Fetched` is provider-shaped
    and goes no further — what travels is a `MarketSnapshot`, addressed by the
    digest of the observations, carrying the adapter that produced them and the
    request that selected them.
    """
    from .snapshot_contract import SourceAdapter, describe

    class _Source:
        """The fields `describe` reads, from what the adapter declared."""

        snapshot_id = None
        dataset_id = fetched.dataset_id
        uri = fetched.source_uri
        calendar = fetched.calendar
        data_as_of = fetched.data_as_of
        license_class = fetched.license_class
        license_review_status = fetched.license_review_status
        content_digest_version = "mdv1"

    source = _Source()
    source.snapshot_id = f"{fetched.adapter_name}@{fetched.adapter_version}"
    return describe(
        source, fetched.observations,
        resolution={"reinvested": bool(reinvested), "version": "mdr1"},
        adapter=SourceAdapter(fetched.adapter_name, fetched.adapter_version))
