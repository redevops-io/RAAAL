# Three services, an evaluation engine, and a market-data lake

A design note, not a plan of record. It exists so the next decision is taken
against checked facts rather than remembered ones, and so the parts that are
*already true* of this repository are not rediscovered.

## What was checked

The premise was "DuckDB 1.4, because it allows multiple writers". Half right,
and the half that is wrong changes the design.

**DuckDB does not give multiple *processes* write access to a database file.**
Its MVCC and optimistic concurrency are within one process, across threads —
[Concurrency](https://duckdb.org/docs/current/connect/concurrency). Multi-process
writing is the `Quack` client/server protocol, beta as of v1.5.2 and expected
mature at v2.0, autumn 2026. Three services on Kubernetes are three processes,
so a shared `.duckdb` file is not the shape.

**What 1.4 actually gives is Iceberg writes.**
[DuckDB 1.4.0 "Andium"](https://duckdb.org/2025/09/16/announcing-duckdb-140) added
`INSERT`, `UPDATE`, `DELETE` and `MERGE` against Iceberg tables. So the version
choice is right and the reason is different: it is not that DuckDB learned to
share a file, it is that DuckDB learned to write to a format that was already
designed for many writers.

**Multiple writers is Iceberg's property, not DuckDB's.** Concurrency comes
from atomic snapshot commits through a catalog, which is also why
[writes require an Iceberg REST catalog](https://duckdb.org/docs/current/core_extensions/iceberg/writing) —
on AWS, S3 Tables or SageMaker Lakehouse. Files in a bucket are not enough.
This is the one hard requirement to design around.

**1.4 is the LTS line.** Codename Andium, community support to 16 September
2026, currently at
[1.4.5](https://duckdb.org/2026/06/17/announcing-duckdb-145). 1.5.5 is newer
and not LTS. For a market-data lake that outlives a release cycle, pinning the
LTS is the defensible choice, and `v1.4-andium` is that branch.

## What is already true here

The service split maps onto module boundaries this repository already has,
which is the cheapest kind of migration:

    src/workspace   the front end — pages, plans, the parameter table
    src/mission     the evaluation engine — compile, simulate, refuse by name
    src/market_data the data engine — access, provenance, licensing

QuantLib is already installed and in `requirements-core.txt`, so the evaluation
service starts with the vocabulary rather than acquiring it.

**The licensing gate is the constraint nobody should route around.**
`market_data.access.approved_snapshot()` re-reads the recorded answers to six
vendor licensing questions on every resolve, and returns nothing if one is
missing; `pilot_data_policy` is `SYNTHETIC_ONLY` today and Terraform will
refuse a value the application does not recognise. A market-data lake means
real vendor data at rest, and that is exactly what those questions are about —
redistribution, derived works, retention, egress. Standing up the lake is a
licensing decision before it is an engineering one, and the gate should govern
the *data engine*, not be re-implemented beside it.

## What QuantLib buys as an engine

Fixed income, annuities and structured products, which is the stated reason and
a real one — bonds, swaps, swaptions, caps, floors, options, and the term
structures they price against. None of that is expressible today.

Worth stating plainly: the current simulator is not a subset of QuantLib. It
executes a *plan* — contributions on a schedule, rebalancing, benchmarks
receiving the same money on the same days — and QuantLib prices *instruments*.
A migration is not a port; it is one service calling the other for the
instrument half while keeping the plan half. The capability manifest, the
refusals by name and 5600 tests are built around the plan half, and they are
the thing that makes a refusal honest rather than a crash.

## The shape

    quantify-web        pages, sessions, plans          → users database (Postgres)
    quantify-evaluate   QuantLib + the plan simulator   → stateless
    quantify-data       ingest, provenance, licensing   → Iceberg lake (S3 + REST catalog)

Two databases, as described: the existing Postgres keeps users and their plans;
market data lives in the lake. They should not be the same store — one is
somebody's private workspace and the other is licensed vendor data, and the
retention, erasure and egress rules differ for exactly that reason.

## What it costs

Honesty about the jump: today this is one `t3.small` running docker compose
behind a Cloudflare tunnel, with an internal ALB and RDS. Managed Kubernetes
plus an Iceberg catalog plus a data engine is a different operational class —
more moving parts, a real monthly bill, and a deployment story that has already
cost this project several evenings at its current size.

The staged path that keeps the deployment working throughout:

1. **Split evaluation out first.** It is stateless, it already has QuantLib,
   and it is the only service that can be extracted without touching data
   licensing or session handling. If the split is going to be painful, this is
   where it shows, and it is reversible.
2. **Stand up the lake read-only.** S3 Tables as the REST catalog, DuckDB 1.4
   LTS as the reader, synthetic data first. The licensing gate stays
   `SYNTHETIC_ONLY` and nothing about vendor terms is decided yet.
3. **Answer the licensing questions for data at rest**, then let the data
   engine write. This is the step that needs a person and not a deploy.
4. **Kubernetes last.** Three services on compose on one host is a valid
   intermediate state and tests the split without the cluster. Moving to EKS
   is then a deployment change rather than a redesign.

## The trigger for the lake is reproducibility, not volume

This section replaces an earlier one that said the trigger was "data volume or
a second consumer". That was wrong, and wrong in a way this project has spent
months learning to recognise elsewhere.

When Quantify says *"8.7%, a 19.2% maximum drawdown, ending at $413,280"*, the
questions that decide whether the number means anything are: which SPY
observations, adjusted or unadjusted, which corporate actions, which calendar,
which FX, which curve, which inflation series, which snapshot — and **what was
known on each date**. Without answers, the arithmetic can be formally proved
while the economic history fed into it is wrong.

That is the same defect class as the one Discovery exists to prevent, one
layer down. Discovery refuses to guess what a sentence meant; the data
substrate currently guesses what the market did. Volume is irrelevant: a
hundred rows nobody can reproduce is a worse position than a billion rows that
anybody can.

## What is already true, and it is more than expected

**The ledger exists.** `accounting.Fill` is a line — date, ticker, shares,
price, notional, cost, reason — described in its own docstring as "what
actually happened, at the price that was actually available".
`PortfolioPath` carries end-of-day value, cash, holdings per ticker, external
flows, the fills, and the orders that *could not* execute. Time-weighted and
money-weighted returns are computed from it.

So the engine is already a historical portfolio accounting engine. What it is
not is one whose ledger anybody can see: nothing renders the fills. The page
shows a figure and a chart derived from a ledger the person is never shown,
which is a presentation gap rather than an engine rewrite.

**The reproducibility question is already asked, and cannot be answered.**
Every run records a `market_data_access_event` carrying `snapshot_id`,
`provenance_digest` and `frame_digest` — "the digest of the exact canonical
frame that was handed over". The schema comment is explicit that a snapshot id
alone is insufficient because two provenances differing only in access time are
different records.

So the system already knows it must identify the exact bytes it computed on.
What it cannot do is *rebuild* them: the digest names a frame that no store can
reconstruct from raw observations. The lake is not a new idea being introduced
here — it is the missing half of a mechanism that is already load-bearing.

## The layering

    RAW          vendor observations, as received, never edited
                 Yahoo / Polygon / Nasdaq / FRED / Treasury / EDGAR
        |
    NORMALIZED   instrument identity, calendar, currency,
                 corporate actions, prices, rates
        |
    CANONICAL    total-return series, cash rates, FX, inflation,
                 yield curves, benchmark series
        |
    SNAPSHOT     market-snapshot:<hash>   immutable, published
        |
    EVALUATION   strategy + snapshot + engine version
        |
    MissionResult

The property this buys is that an evaluation becomes close to a pure function:

    evaluate(strategy_hash, market_snapshot_hash, engine_version) -> MissionResult

`quantify-evaluate` must not query vendor tables. It consumes a published
snapshot and nothing else — otherwise "which observations" becomes a question
about when the query ran, which is exactly the state `frame_digest` was added
to escape.

RAW is kept unedited on purpose. A normalisation that overwrites its input
destroys the only evidence that could settle a disagreement about what the
vendor actually said.

## Where QuantLib sits

    Mission strategy
          |
    Portfolio simulator          <- executes the plan, writes the ledger
          |-- equities/ETFs/cash  -> canonical observations
          |-- bonds               -> QuantLib
          |-- options             -> QuantLib
          |-- annuities           -> QuantLib
          |-- rates/curves        -> QuantLib + canonical data
          |
    Ledger -> Formal Core -> MissionResult

QuantLib prices instruments; the simulator executes plans and produces the
ledger. Neither replaces the other, and the ledger is where they meet.

What this opens: four strategies compared against *the same* snapshot rather
than against separately assembled series, which is the difference between a
comparison and a coincidence.

## Free sources, and the licensing gate they still meet

Much of a first data layer is available without a vendor contract:

| Data | Source | Suitability |
|---|---|---|
| Treasury rates and curves | US Treasury | excellent, public domain |
| CPI, Fed rates, macro | FRED / originating agency | excellent; check each series' terms |
| Company fundamentals | SEC EDGAR / XBRL | excellent, public domain |
| US equity and ETF daily prices | open datasets | fine for development; rights vary |
| Dividends and splits | open datasets | usable, provenance needs care |
| ETF holdings and metadata | issuer publications | often usable, terms vary |
| Index levels | index owner | frequently licensed |
| Options, intraday equities | commercial | exchange licensing applies |
| Corporate bonds and credit | fragmented | governments easy, corporates hard |

"Free to download" is not "free to redistribute, derive from, or retain", and
those are exactly the six questions `approved_snapshot()` re-reads on every
resolve. A public-domain Treasury series and a scraped index level are not the
same licensing object, and the lake must carry the distinction per series
rather than per bucket.

## What would make this wrong

- If the instrument work stays hypothetical, the evaluation split buys
  operational cost and no capability. The trigger for stage 1 is a real
  instrument somebody wants modelled, not the diagram.
- If the ledger is never shown and no second strategy is ever compared on the
  same snapshot, the layering is bookkeeping nobody reads. The cheapest test of
  this whole direction is to render the fills that already exist.
- If the licensing answers do not permit vendor data at rest, the lake stays
  synthetic — and reproducibility of a *synthetic* snapshot is still worth
  having, because it is what makes two runs comparable. That outcome shrinks
  the lake; it does not remove the reason for it.
