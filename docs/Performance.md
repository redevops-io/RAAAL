# Performance — measured

**Date:** 2026-08-01 · **Hardware:** 32 cores · **Polars:** 1.43.2

Every number here is measured on this machine, not budgeted. The load-test plan
set engineering budgets to validate; this replaces them with observations.
Where a budget and a measurement disagree, the measurement is what shipped.

---

## 1. The compiler is effectively free

14,400 descriptions — 144 catalog strategies × 100 paraphrase variants — through
the full deterministic pipeline.

| Stage | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Stage 1 parse | 34 µs | 55 µs | 65 µs | 1.2 ms |
| Stages 2–8 compile | 9 µs | 11 µs | 12 µs | 0.4 ms |
| **Total per description** | **45 µs** | **66 µs** | **80 µs** | 1.3 ms |

The plan's budget for the deterministic acceptance chain was **150 ms p95**.
Measured: **0.066 ms**. Three orders of magnitude inside it.

That reshapes where optimization belongs. The architecture is

```
language model      expensive, one call, quarantined to stage 1
      ↓
deterministic       ~45 µs, and it is the part that decides anything
compiler
      ↓
runtime
```

"AI compiler" reads as expensive. Only the first step is. Everything that
determines what actually gets simulated costs less than a network round trip's
jitter, which is why it can afford to be exhaustive rather than approximate.

---

## 2. HarnessBench — canonical versus Polars

Four analytical workloads, three backends, p50 milliseconds. Results are
compared before any timing is reported: **all backends agreed on every workload
at every scale.** Polars is an execution backend, never a second owner of
semantics.

Read from a Parquet projection, which is what the architecture specifies.

| Workload | Scale | canonical | Polars eager | Polars lazy | best speedup |
|---|---:|---:|---:|---:|---:|
| latency_summary | 1K | **0.31** | 2.20 | 2.38 | 0.14× |
| | 10K | **1.87** | 3.71 | 2.08 | 0.90× |
| | 100K | 25.10 | 17.71 | **4.24** | 5.9× |
| | 1M | 315.70 | 71.57 | **26.93** | 11.7× |
| mission_replay | 1K | **2.45** | 3.85 | 5.10 | 0.64× |
| | 10K | 5.95 | 7.30 | **5.31** | 1.1× |
| | 100K | 54.39 | 22.66 | **9.28** | 5.9× |
| | 1M | 691.21 | 109.02 | **28.62** | 24.2× |
| denial_scan | 1K | **0.81** | 3.72 | 4.07 | 0.22× |
| | 10K | **4.99** | 9.93 | 7.76 | 0.64× |
| | 100K | 41.80 | 27.76 | **16.56** | 2.5× |
| | 1M | 385.63 | 63.77 | **20.04** | 19.2× |
| secret_then_egress | 1K | **0.13** | 2.06 | 3.40 | 0.06× |
| | 10K | **1.28** | 2.34 | 3.22 | 0.55× |
| | 100K | 24.16 | **4.31** | 6.62 | 5.6× |
| | 1M | 284.49 | **11.02** | 12.43 | 25.8× |

### Crossover, per workload

Measured and stored per workload, never one global constant:

```
mission_replay        10,000 events
latency_summary      100,000 events
denial_scan          100,000 events
secret_then_egress   100,000 events
```

`mission_replay` crosses an order of magnitude earlier than the rest. A single
threshold would be wrong for both sides of that gap — which is the argument for
measuring it rather than picking one.

Below the crossover Polars is *slower*, by up to 16×. Constant setup cost
dominates small queries, so routing a 1,000-event replay through Polars would
make an interactive path measurably worse.

---

## 3. The finding that mattered most

**How the data reaches Polars decides whether Polars helps at all.**

The same four workloads at 100,000 events, fed from Python dicts instead of a
Parquet projection:

| Workload | canonical | Polars eager | Polars lazy |
|---|---:|---:|---:|
| latency_summary | **26.7** | 248.8 | 246.3 |
| mission_replay | **55.0** | 247.3 | 241.5 |
| denial_scan | **41.9** | 251.4 | 248.5 |
| secret_then_egress | **23.0** | 238.2 | 239.6 |

Polars loses every workload, by 4–10×, and the ranking inverts completely.

Constructing a DataFrame from a list of dicts costs ~240 ms at this scale and
swamps every query. Timing that path and calling it "Polars" measures the
adapter, not the engine — and would have produced the confident and entirely
wrong conclusion that Polars is not worth adopting.

The plan already specified partitioned Parquet projections written once by the
event buffer and scanned many times. This is the measurement that says why that
detail is load-bearing rather than incidental.

---

## 4. What this means for the roadmap

Polars belongs **nowhere near** the interactive path:

```
parser · compiler · runtime validation · authorization · ledger writes
    already microsecond-scale; Polars would make them slower
```

It belongs where the volumes are:

```
replay · aggregation · trajectory mining · Discovery sweeps
validation scans · fleet monitoring · Mission evolution
    5-25x at a million events, and growing with scale
```

---

## 5. Reproducing

```bash
python3 scripts/run_load_corpus.py --per-strategy 100     # compiler corpus
python3 scripts/run_harnessbench.py                       # Polars crossover
python3 scripts/run_harnessbench.py --from-dicts          # the adapter cost
python3 scripts/compiler_dashboard.py                     # quality metrics
```

All four run on committed synthetic data. No credentials, no network.
