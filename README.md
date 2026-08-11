# Quantify

> Quantify compiles financial scenarios into transparent, versioned simulations.
> It records the methodology, evaluation protocol, calendar, tax and account
> treatment, cash flows, market-data policy, realized data, statistical
> assessment, and modelling limitations behind every result.

Not an AI portfolio optimizer, and not a backtesting dashboard. An event-driven
financial simulation and research runtime built on versioned knowledge and
execution artifacts.

**857 tests passing** · AGPL-3.0

---

## Two surfaces

### Quantify Library — public

> A version-controlled knowledge system for investment research: what is claimed,
> what supports it, what assumptions it depends on, how it was tested, what
> changed, and whether the result may responsibly be published.

Impersonal by construction. No holdings, income, taxes or objectives; no
personalized ranking.

### Quantify Scenarios — private

> Describe a financial workflow, confirm exactly what will be simulated, compare
> symmetric historical outcomes, save the plan privately, and track what happens
> forward. Quantify does not place orders or choose a course of action for you.

The boundary runs one way — a private plan may cite public research; no public
artifact may ever cite a private one.

---

## The principle

> Every meaningful behaviour is declared and realized; every declaration names
> the mechanism that enforces or checks it.

Two failure modes are closed by construction: **behaviour without declaration**,
and **declaration without behaviour**. The second is the harder one — a
methodology that declares a rule its executor ignores looks exactly like one that
enforces it.

This is not theoretical. Every release that made a hidden choice explicit
immediately exposed a real defect:

| Made explicit | Found |
|---|---|
| Execution lag and costs | A reported 13.00% return was actually **−2.83%** |
| Trading calendar | **31.1%** weekend padding inflating annualized figures |
| Rules naming their realization | Declared rules were **inert** — never executed |
| Tax treatment as a runtime | A Roth and a taxable account compared as identical |

---

## Quick start

```bash
pip install -r requirements-core.txt   # or requirements-core.lock, pinned
python3 -m pytest tests/ -q
uvicorn src.api:app --reload           # /ui library · /workspace private scenarios
```

The suite runs entirely on a committed **synthetic** price fixture — invented,
deterministic, no credentials and no network. Nothing measured on it is a claim
about any real security.

Licensed market data lives in a private, versioned S3 bucket, pinned by snapshot
id, object version and hash in `data/manifests/`. The bucket name is supplied by
`.env.market-data` (gitignored) so it stays out of a published repository; the
manifest carries everything needed to review what a result was computed against.
It is used by the opt-in tier:

```bash
source .env.market-data
pytest -m market_data_integration     # requires credentials; fails, never skips
python3 scripts/provision_market_data.py --dry-run   # plan a new snapshot upload
```

Describing a scenario in prose uses a language model for **stage 1 of the
compiler only** — recognising phrases, never deciding anything. Set
`ANTHROPIC_API_KEY` to enable it, and `QUANTIFY_PARSER_MODEL` to choose a model.
Without a key the compiler uses its deterministic phrase rules, recognises less,
and asks more questions; it never guesses to fill the gap.

```bash
python3 scripts/run_methodology.py   # execute a methodology under a protocol
python3 scripts/evaluate.py          # assessment → policy → publication
python3 scripts/publish_run.py       # record a run in the ledger
python3 scripts/assess.py            # statistical assessment only
```

---

## Documentation

| | |
|---|---|
| [Architecture.md](docs/Architecture.md) | Artifact model, runtime lifecycle, boundaries, comparability, regulatory posture |
| [Features.md](docs/Features.md) | What the system does, by surface |
| [Implementation.md](docs/Implementation.md) | Status, defect history, acceptance criteria, the architecture freeze and Closed Pilot v1 |
| [Performance.md](docs/Performance.md) | Measured latency, HarnessBench, the Polars crossover |
| [docs/errata/](docs/errata/) | Published corrections |

The one place status is easy to misread: `Investigation` is **implemented as a
knowledge artifact** — persisted, queryable, rendered — and **not yet implemented
as a durable unit of work**. Lifecycle transitions, Discovery-driven creation and
conclusion-to-Finding routing are the remaining work. See
[Implementation.md §8](docs/Implementation.md).

---

## Research grounding

Source papers are artifacts under `evidence/` rather than a bibliography, so a
claim can be supported, qualified or contradicted by them and the relationship is
queryable:

- **López de Prado, M. (2016).** "Building Diversified Portfolios That Outperform
  Out of Sample." *Journal of Portfolio Management* 42(4), 59–69.
  [doi](https://doi.org/10.3905/jpm.2016.42.4.59) — hierarchical risk parity
- **Jegadeesh, N. & Titman, S. (1993).** Returns to buying winners and selling
  losers — cross-sectional momentum
- **Bailey, D. & López de Prado, M.** Deflated Sharpe ratio, probability of
  backtest overfitting, minimum track record length
- **Vuletic, M. (2025).** *Multi-asset financial markets: mathematical modelling
  and data-driven approaches* (Oxford DPhil thesis) — regime detection
- **CFA Institute Research Foundation (2025).** *AI in Asset Management*
  — [monograph](https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/rf_aiinassetmanagement_full-monograph_online.pdf)
- **Guo, J. & Li, Y. (2026).** *Salience Theory and Risk Anomalies*
  — [SSRN](https://ssrn.com/abstract=4603171)

The cash-proxy finding shows how these are used.
`finding/hrp-degenerates-to-cash-proxy@1` **refutes** one claim, **qualifies**
another — López de Prado's result holds for comparable-risk universes, not any
universe — **invalidates the results of** two methodology versions, **motivates**
a third, and **introduces** a constraint-precedence assumption. One conclusion,
five typed impacts.

---

## Deployment

The static dashboard builds to `reports/` and deploys to Cloudflare Pages:

```bash
python3 -m src.history --start 2015-01-01 --end $(date +%Y-%m-%d) --step 5
python3 -m src.visualization.bokeh_app --output reports/regime_dashboard.html

export DOMAIN="quantify.club"
./deploy_cloudflare.sh
```

DNS is a `CNAME` from `@` (or `www`) to the Pages project; Cloudflare provisions
TLS automatically. Daily rebuilds run from
[.github/workflows/daily-deploy.yml](.github/workflows/daily-deploy.yml) and need
`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_EMAIL` as
repository secrets.

---

## Licence

**Source-available, not open source.** AGPL-3.0-or-later plus the Commons
Clause condition.

AGPL §13 extends copyleft to **network use**: serving users over a network with
this code obliges offering them the corresponding source of the combined work.
The Commons Clause additionally removes the right to *sell* the software,
including hosting it for a fee — which AGPL on its own permits.

See [LICENSE.md](LICENSE.md) and [LICENSE-COMMONS-CLAUSE](LICENSE-COMMONS-CLAUSE).
