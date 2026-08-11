"""HarnessBench: the same workload three ways, compared on results first.

    canonical      Python objects. The semantics.
    polars eager   the same query, materialized.
    polars lazy    the same query, streamed.

**Speed is the second question.** The first is whether the three agree, because
a faster backend that computes something slightly different has not accelerated
anything — it has forked the semantics, and the fork will be discovered by a
customer. Every workload here declares an equivalence invariant and the runner
checks it before it reports a timing.

Polars is an execution backend, never a second owner of semantics. Where the two
disagree, the canonical implementation is right by definition and the Polars one
is a defect.

The crossover point is measured per workload and stored, not fixed as one global
constant. A grouped replay and a latency aggregation cross at different sizes,
and a single threshold would be wrong for both.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

Rows = Sequence[Mapping[str, Any]]


# --- canonical implementations --------------------------------------------
#
# Plain Python, deliberately. These define what each workload means; they are
# not a strawman to be beaten. Where one is written inefficiently the comparison
# is worthless, so each is the way it would actually be written.

def canonical_latency_summary(rows: Rows) -> List[Dict[str, Any]]:
    """p50/p95/p99 of duration by event type."""
    buckets: Dict[str, List[int]] = {}
    for row in rows:
        buckets.setdefault(row["event_type"], []).append(row["duration_us"])

    out = []
    for event_type, values in buckets.items():
        values.sort()
        n = len(values)
        out.append({
            "event_type": event_type, "events": n,
            "p50_us": values[min(n - 1, int(0.50 * n))],
            "p95_us": values[min(n - 1, int(0.95 * n))],
            "p99_us": values[min(n - 1, int(0.99 * n))],
        })
    return sorted(out, key=lambda r: r["event_type"])


def canonical_mission_replay(rows: Rows) -> List[Dict[str, Any]]:
    """Derived state per mission: last event, length, denials, span.

    The fleet-wide version of what the ledger does for one mission. Ordering is
    by sequence number, because that is the ledger's order and a replay that
    sorts by timestamp would reorder two events written in the same microsecond.
    """
    by_mission: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_mission.setdefault(row["mission_id"], []).append(row)

    out = []
    for mission_id, history in by_mission.items():
        history = sorted(history, key=lambda r: r["sequence_number"])
        denials = sum(1 for r in history if r["decision"].startswith("DENIED"))
        out.append({
            "mission_id": mission_id,
            "events": len(history),
            "last_sequence": history[-1]["sequence_number"],
            "last_event_type": history[-1]["event_type"],
            "denials": denials,
            "span_s": history[-1]["occurred_at"] - history[0]["occurred_at"],
        })
    return sorted(out, key=lambda r: r["mission_id"])


def canonical_denial_scan(rows: Rows) -> List[Dict[str, Any]]:
    """Which rules deny, per tenant. A validation sweep."""
    counts: Dict[tuple, int] = {}
    for row in rows:
        if not row["decision"].startswith("DENIED"):
            continue
        key = (row["tenant_id"], row["rule_id"], row["decision"])
        counts[key] = counts.get(key, 0) + 1
    return sorted(
        ({"tenant_id": t, "rule_id": r, "decision": d, "denials": n}
         for (t, r, d), n in counts.items()),
        key=lambda r: (r["tenant_id"], r["rule_id"], r["decision"]))


def canonical_secret_then_egress(rows: Rows, *, window_s: int = 600
                                 ) -> List[Dict[str, Any]]:
    """An egress within `window_s` of a secret read, in the same session.

    A sequence rule, and the shape Discovery uses: not "did X happen" but "did X
    happen after Y, close enough to matter".
    """
    reads: Dict[str, List[int]] = {}
    egress: List[Mapping[str, Any]] = []
    for row in rows:
        if row["event_type"] == "secret.read":
            reads.setdefault(row["session_id"], []).append(row["occurred_at"])
        elif row["event_type"] == "network.egress":
            egress.append(row)

    for times in reads.values():
        times.sort()

    import bisect

    out = []
    for row in egress:
        times = reads.get(row["session_id"])
        if not times:
            continue
        index = bisect.bisect_right(times, row["occurred_at"]) - 1
        if index >= 0 and row["occurred_at"] - times[index] <= window_s:
            out.append({"session_id": row["session_id"],
                        "egress_event_id": row["event_id"],
                        "gap_s": row["occurred_at"] - times[index]})
    return sorted(out, key=lambda r: r["egress_event_id"])


# --- Polars implementations ------------------------------------------------

#: Set by `with_parquet_projection` so the Polars backends read the analytical
#: projection instead of rebuilding a frame from Python dicts on every call.
#:
#: This distinction turned out to dominate every measurement. Constructing a
#: DataFrame from a list of dicts is the slowest way to get data into Polars and
#: is not what the architecture does — the plan specifies partitioned Parquet
#: projections, written once by the event buffer and scanned many times. Timing
#: the dict path and calling it "Polars" measures the adapter, not the engine.
_PROJECTION: Optional[str] = None


def with_parquet_projection(path: Optional[str]) -> None:
    global _PROJECTION
    _PROJECTION = path


def _frame(rows: Rows):
    import polars as pl

    if _PROJECTION:
        return pl.scan_parquet(_PROJECTION).collect()
    return pl.DataFrame(list(rows), infer_schema_length=None)


def _lazy_frame(rows: Rows):
    """Lazy from Parquet where a projection exists — the point of lazy is that
    the scan, the filter and the projection are planned together."""
    import polars as pl

    if _PROJECTION:
        return pl.scan_parquet(_PROJECTION)
    return _frame(rows).lazy()


def write_projection(rows: Rows, path: str) -> str:
    """Write the analytical projection once, as the event buffer would."""
    import polars as pl

    pl.DataFrame(list(rows), infer_schema_length=None).write_parquet(
        path, compression="zstd")
    return path


def _quantile_expr(q: float, alias: str):
    """Match the canonical index-based percentile exactly.

    Polars' default `quantile` interpolates, which produces a different number
    on the same data — a real equivalence failure the first run caught. The
    canonical form takes the value at `int(q * n)`, so `nearest` is not enough
    either; the index is computed explicitly.
    """
    import polars as pl

    n = pl.col("duration_us").len()
    index = (q * n).cast(pl.Int64).clip(upper_bound=n - 1)
    return pl.col("duration_us").sort().get(index).alias(alias)


def polars_latency_summary(rows: Rows, *, lazy: bool) -> List[Dict[str, Any]]:
    import polars as pl

    query = (_lazy_frame(rows) if lazy else _frame(rows)).group_by("event_type").agg(
        pl.len().alias("events"),
        _quantile_expr(0.50, "p50_us"),
        _quantile_expr(0.95, "p95_us"),
        _quantile_expr(0.99, "p99_us"),
    ).sort("event_type")
    result = query.collect(engine="streaming") if lazy else query
    return result.to_dicts()


def polars_mission_replay(rows: Rows, *, lazy: bool) -> List[Dict[str, Any]]:
    import polars as pl

    query = (
        (_lazy_frame(rows) if lazy else _frame(rows))
        .sort("sequence_number")
        .group_by("mission_id")
        .agg(
            pl.len().alias("events"),
            pl.col("sequence_number").last().alias("last_sequence"),
            pl.col("event_type").last().alias("last_event_type"),
            pl.col("decision").str.starts_with("DENIED").sum().alias("denials"),
            (pl.col("occurred_at").last() - pl.col("occurred_at").first()
             ).alias("span_s"),
        )
        .sort("mission_id")
    )
    result = query.collect(engine="streaming") if lazy else query
    return result.to_dicts()


def polars_denial_scan(rows: Rows, *, lazy: bool) -> List[Dict[str, Any]]:
    import polars as pl

    query = (
        (_lazy_frame(rows) if lazy else _frame(rows))
        .filter(pl.col("decision").str.starts_with("DENIED"))
        .group_by(["tenant_id", "rule_id", "decision"])
        .agg(pl.len().alias("denials"))
        .sort(["tenant_id", "rule_id", "decision"])
    )
    result = query.collect(engine="streaming") if lazy else query
    return result.to_dicts()


def polars_secret_then_egress(rows: Rows, *, lazy: bool,
                              window_s: int = 600) -> List[Dict[str, Any]]:
    import polars as pl

    base = _lazy_frame(rows)
    reads = (base.filter(pl.col("event_type") == "secret.read")
             .select("session_id", pl.col("occurred_at").alias("secret_at"))
             .sort("secret_at"))
    egress = (base.filter(pl.col("event_type") == "network.egress")
              .select("session_id", pl.col("occurred_at").alias("egress_at"),
                      pl.col("event_id").alias("egress_event_id"))
              .sort("egress_at"))
    query = (
        egress.join_asof(reads, left_on="egress_at", right_on="secret_at",
                         by="session_id", strategy="backward")
        .filter(pl.col("secret_at").is_not_null())
        .with_columns((pl.col("egress_at") - pl.col("secret_at")).alias("gap_s"))
        .filter(pl.col("gap_s") <= window_s)
        .select("session_id", "egress_event_id", "gap_s")
        .sort("egress_event_id")
    )
    return query.collect(engine="streaming" if lazy else "in-memory").to_dicts()


# --- the runner ------------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    name: str
    canonical: Callable[[Rows], List[Dict[str, Any]]]
    polars: Callable[..., List[Dict[str, Any]]]
    invariant: str


WORKLOADS = (
    Workload("latency_summary", canonical_latency_summary,
             polars_latency_summary,
             "identical p50/p95/p99 and counts per event type"),
    Workload("mission_replay", canonical_mission_replay, polars_mission_replay,
             "identical derived state, last sequence and denial count per mission"),
    Workload("denial_scan", canonical_denial_scan, polars_denial_scan,
             "identical denial counts per tenant, rule and decision"),
    Workload("secret_then_egress", canonical_secret_then_egress,
             polars_secret_then_egress,
             "identical findings and cited events"),
)


def _normalize(rows: List[Dict[str, Any]]) -> List[tuple]:
    """Compare as values, not as objects.

    Polars returns numpy-backed integers that are equal to Python ints but not
    identical to them, and dict ordering differs between the two paths. Neither
    is a semantic difference, and treating either as one would bury a real
    mismatch under noise.
    """
    return [tuple(sorted((k, int(v) if isinstance(v, bool) else v)
                         for k, v in row.items())) for row in rows]


@dataclass
class Measurement:
    workload: str
    backend: str
    rows: int
    result_count: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    peak_kib: int
    matches_canonical: Optional[bool] = None
    mismatch: str = ""

    def as_row(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _time(fn: Callable[[], Any], *, repeats: int) -> tuple:
    timings, result = [], None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter_ns()
        result = fn()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    timings.sort()
    return timings, result


def measure(workload: Workload, rows: Rows, *, repeats: int = 3
            ) -> List[Measurement]:
    """One workload, three backends, results compared before timings reported."""
    out: List[Measurement] = []
    canonical_result: Optional[List[Dict[str, Any]]] = None

    backends = (
        ("canonical", lambda: workload.canonical(rows)),
        ("polars_eager", lambda: workload.polars(rows, lazy=False)),
        ("polars_lazy", lambda: workload.polars(rows, lazy=True)),
    )
    for name, fn in backends:
        tracemalloc.start()
        try:
            timings, result = _time(fn, repeats=repeats)
            peak = tracemalloc.get_traced_memory()[1] // 1024
        finally:
            tracemalloc.stop()

        measurement = Measurement(
            workload=workload.name, backend=name, rows=len(rows),
            result_count=len(result),
            p50_ms=round(timings[len(timings) // 2], 3),
            p95_ms=round(timings[min(len(timings) - 1,
                                     int(0.95 * len(timings)))], 3),
            min_ms=round(timings[0], 3), peak_kib=peak)

        if name == "canonical":
            canonical_result = result
        else:
            expected = _normalize(canonical_result or [])
            actual = _normalize(result)
            measurement.matches_canonical = expected == actual
            if not measurement.matches_canonical:
                measurement.mismatch = _describe(expected, actual)
        out.append(measurement)
    return out


def _describe(expected: List[tuple], actual: List[tuple]) -> str:
    if len(expected) != len(actual):
        return f"{len(expected)} canonical rows vs {len(actual)}"
    for index, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            differing = [k for (k, av), (_, bv) in zip(a, b) if av != bv]
            return (f"row {index} differs on {differing}: "
                    f"{dict(a)} vs {dict(b)}")
    return "unknown difference"


def crossover(measurements: Sequence[Measurement]) -> Dict[str, Optional[int]]:
    """The scale at which each Polars backend first beats canonical.

    Per workload, deliberately. A grouped replay and a latency aggregation cross
    at different sizes, and a single global threshold would be wrong for both.
    `None` means canonical still wins at every scale measured — which is a
    finding, not a gap.
    """
    by_workload: Dict[str, Dict[int, Dict[str, float]]] = {}
    for m in measurements:
        by_workload.setdefault(m.workload, {}).setdefault(m.rows, {})[m.backend] = m.p50_ms

    out: Dict[str, Optional[int]] = {}
    for workload, scales in by_workload.items():
        found = None
        for size in sorted(scales):
            timings = scales[size]
            best_polars = min((timings.get("polars_eager", float("inf")),
                               timings.get("polars_lazy", float("inf"))))
            if best_polars < timings.get("canonical", float("inf")):
                found = size
                break
        out[workload] = found
    return out
