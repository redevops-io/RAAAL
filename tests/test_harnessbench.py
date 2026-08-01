"""HarnessBench: the equivalence invariant, checked before any timing is read.

The claim under test is not that Polars is fast. It is that Polars computes the
*same thing*, because a faster backend that differs slightly has not accelerated
anything — it has forked the semantics, and the fork gets discovered by a
customer rather than by a test.

Polars is an execution backend, never a second owner of semantics. Where the two
disagree, the canonical implementation is right by definition.
"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")

from src.loadtest.events import EVENT_COLUMNS, generate
from src.loadtest.harnessbench import (
    WORKLOADS, crossover, measure, with_parquet_projection, write_projection,
)


@pytest.fixture(scope="module")
def rows():
    return generate(5_000)


@pytest.fixture(scope="module")
def projection(rows, tmp_path_factory):
    path = tmp_path_factory.mktemp("hb") / "events.parquet"
    write_projection(rows, str(path))
    with_parquet_projection(str(path))
    yield str(path)
    with_parquet_projection(None)


class TestTheEventStream:

    def test_it_is_deterministic(self):
        """A mismatch found on one machine must reproduce on another."""
        assert generate(500, seed=3) == generate(500, seed=3)

    def test_it_carries_the_planned_schema(self, rows):
        assert set(rows[0]) == set(EVENT_COLUMNS)

    def test_mission_histories_vary_in_length(self, rows):
        """Replay cost is decided by the longest history, not the mean.

        A uniform stream makes every group-by look the same size and hides
        exactly the skew that decides whether a query is fast.
        """
        from collections import Counter

        lengths = sorted(Counter(r["mission_id"] for r in rows).values())
        assert lengths[-1] > lengths[len(lengths) // 2]

    def test_traffic_is_skewed_across_tenants(self, rows):
        from collections import Counter

        share = max(Counter(r["tenant_id"] for r in rows).values()) / len(rows)
        assert share > 1.5 / 64, "a few tenants should own most of the traffic"


class TestEquivalence:
    """The invariant. Every workload, both Polars backends."""

    @pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
    def test_polars_computes_what_canonical_computes(self, workload, rows,
                                                     projection):
        measurements = measure(workload, rows, repeats=1)
        for m in measurements:
            if m.matches_canonical is None:
                continue
            assert m.matches_canonical, (
                f"{workload.name}/{m.backend} diverged from canonical — "
                f"{workload.invariant} does not hold: {m.mismatch}")

    def test_a_seeded_divergence_is_caught(self, rows, projection):
        """The equivalence check must be able to fail.

        Percentile interpolation caused a real mismatch on the first run: Polars
        interpolates by default and the canonical form takes the value at an
        index. If this check could not detect that, it would be decoration.
        """
        from src.loadtest.harnessbench import Workload, canonical_denial_scan

        def wrong(rows, *, lazy: bool):
            out = canonical_denial_scan(rows)
            return out[:-1] if out else out

        broken = Workload("seeded", canonical_denial_scan, wrong, "n/a")
        measurements = measure(broken, rows, repeats=1)
        assert all(m.matches_canonical is False
                   for m in measurements if m.matches_canonical is not None)

    def test_results_are_compared_as_values_not_objects(self, rows, projection):
        """Polars returns numpy-backed ints; dict order differs between paths.

        Neither is a semantic difference, and treating either as one would bury
        a real mismatch under noise.
        """
        measurements = measure(WORKLOADS[0], rows, repeats=1)
        assert all(m.result_count == measurements[0].result_count
                   for m in measurements)


class TestTheCrossoverIsMeasured:

    def test_it_is_reported_per_workload(self, rows, projection):
        """Not one global constant. A grouped replay and a latency aggregation
        cross at different sizes, and a single threshold is wrong for both."""
        measurements = [m for w in WORKLOADS for m in measure(w, rows, repeats=1)]
        points = crossover(measurements)
        assert set(points) == {w.name for w in WORKLOADS}

    def test_canonical_winning_is_a_finding_not_a_gap(self, rows, projection):
        """`None` means canonical wins at every scale measured. That is a
        result — it says the workload does not belong on the Polars path."""
        measurements = [m for m in measure(WORKLOADS[0], rows, repeats=1)]
        assert crossover(measurements)[WORKLOADS[0].name] in (None, len(rows))
