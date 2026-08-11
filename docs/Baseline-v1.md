# Quantify v1 — the baseline the pilot starts from

Frozen before any pilot evidence changes anything, so there is a clean
before/after point. Every version below is read out of the tree by
`tests/test_baseline_v1.py` rather than typed here, because a baseline that
records what somebody believed is not a baseline.

    commit                  89959bd
    Discovery schema        quantify-discovery-schema@6/eb01a824e4f43d02
    capability manifest     quantify/capability-manifest@1
    hosted prompt           quantify-hosted-prompt@1
    fusion pipeline         quantify-pipeline@1
    MWR contract            quantify/mwr-contract@1
    drawdown semantics      drawdown@2
    Formal Core             v1 — 89 theorems, 57 guards

The schema is at **@6**, not @3. It moved three times: `@4` added
`reserve_policy`, `bucket_policy` and a `leverage_multiplier` qualifier; `@5`
added `asset_location`; `@6` added `selection_rule` and `holding_period`. Each
was a bump because the *content* changed, which is the rule that keeps two runs
from looking comparable when they are not.

## The schema moved after this baseline was frozen

Recorded rather than rewritten. `@5` was the version frozen here; `@6` came
after, and a baseline that quietly renumbered itself would destroy the
before/after point it exists to provide.

    frozen at               quantify-discovery-schema@5/ca8f3b7785ff5d70
    now                     quantify-discovery-schema@6/eb01a824e4f43d02
    moved by                the strategy evaluation benchmark, before any
                            pilot evidence

The benchmark found `momentum-rotation` executing as buy-and-hold: "hold
whichever performed best" produced a plan with two holdings and a monthly
cadence, the selection silently gone. Closing it needed a dimension that could
*represent* the selection so Mission could refuse it by name — `selection_rule`
— and `holding_period` alongside it, because the same investigation found "hold
VTI for 200 days" and "buy VTI below its 200-day moving average" compiling to
the identical plan.

This is the authored corpus doing the job it was kept for. The pilot has not
started; the change is attributable to the benchmark and to nothing else, and
that attribution is the reason to write it down here rather than edit a line.

## What is settled

- **Formal Core v1** — `docs/FormalCore.md`, with every public claim mapped to
  a named theorem.
- **Discovery serving profile** — hosted model plus the Stanza syntax witness,
  `WitnessProfile.BOTH`, with presence guards so a dropped material dimension
  becomes a question rather than a silence.
- **Mission** — capability manifest refuses by name; relations reach the
  refusal path; `objective` classified.
- **Pilot instrumentation** — seven events, structured context only, per
  participant consent, transcripts off unless declared.

## What is open, and stays open

- `ANTHROPIC_API_KEY` is not configured, so the pre-Lean gate is closed on
  `producer='unknown'`. Every semantic condition passes.
- Mission computes no volatility and no drawdown. `max_drawdown` is declared
  absent in `worksheet_view` with a reason.
- Six prompts are `UNSTABLE_SAFE` and watched.

## What reopens engineering

Four triggers, and nothing else. Everything outside them is pilot evidence, not
a coding task:

1. **wrong executable meaning** — a plan that does not mean what was said
2. **unsafe or silent reduction** — an unsupported intent producing a figure
3. **supported journey blocked** — describe → clarify → save → figure → reopen
4. **repeated user demand crossing a roadmap trigger** — counted, not felt

The fourth is deliberately about frequency rather than architecture. If six of
ten people ask for withdrawals, that earns withdrawal capability. If nobody
asks for factor tilts, the schema being able to represent them is not a reason
to build them.

## The corpora, and which one now matters more

    authored / synthetic     the regression laboratory
    real pilot prompts       the evidence for what deserves investment

The authored corpus becomes secondary in **product prioritisation**, not in
**correctness**. It stays the adversarial and rare-case suite, and nothing
below it may be thinned because a cohort did not happen to type those
sentences.

It found the accumulation bias, the crossing-versus-persistent defect, the
moving-average off-by-one, the MWR two-root case and the drawdown opening-level
gap. Ten people would have produced none of them. What it cannot do is say what
anyone wants, which is the only thing the pilot prompts are better at — so they
become primary for deciding what to build, and for nothing else.

## What no amount of this proves

Whether real people express intentions the schema has not anticipated,
understand a refusal when they meet one, and trust the explanation enough to
approve what is about to be evaluated. That is the pilot's question and nothing
above answers it.
