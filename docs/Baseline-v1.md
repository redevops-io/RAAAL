# Quantify v1 — the baseline the pilot starts from

Frozen before any pilot evidence changes anything, so there is a clean
before/after point. Every version below is read out of the tree by
`tests/test_baseline_v1.py` rather than typed here, because a baseline that
records what somebody believed is not a baseline.

    commit                  89959bd
    Discovery schema        quantify-discovery-schema@6/eb01a824e4f43d02
    capability manifest     quantify/capability-manifest@1
    hosted prompt           quantify-hosted-prompt@1
    serving reader          gpt-4.1-2025-04-14@1
    fusion pipeline         quantify-pipeline@1
    MWR contract            quantify/mwr-contract@1
    drawdown semantics      drawdown@2
    Formal Core             v1 — 89 theorems, 57 guards

The schema is at **@6**, not @3. It moved three times: `@4` added
`reserve_policy`, `bucket_policy` and a `leverage_multiplier` qualifier; `@5`
added `asset_location`; `@6` added `selection_rule` and `holding_period`. Each
was a bump because the *content* changed, which is the rule that keeps two runs
from looking comparable when they are not.

## The serving reader changed provider after this baseline was frozen

Recorded for the same reason the schema move is, and it matters more: every
number in this document was measured through a reader that is no longer the one
serving.

    frozen with             claude-sonnet-5@1
    now                     gpt-4.1-2025-04-14@1
    moved by                the pilot having no Anthropic credential

The corpus was re-recorded under the new reader — both sets are kept, keyed by
reader id, so the old readings remain as history rather than being overwritten.

**What survived the change, and what did not.** This is the strongest available
test of the claim that the guarantees are properties of the architecture rather
than of one model, and the answer is mixed in an informative way.

Held:

- **No silent reduction.** Nothing executes a meaning the person did not ask
  for, which is the class this project exists to remove.
- **No wrong executable meaning.** No contrast pair collapsed to one plan.
- **42 of 43** answerable corpus cases produce the same value as before.
- Every Lean theorem, every accounting identity, every refusal-by-name path —
  none of it touches the reader.

Did not hold:

- **One false claim of support.** "an annual $40,000 withdrawal" is refused for
  `objective` rather than `sell_action`: the right outcome for the wrong stated
  reason, which tells somebody the system cannot do a thing it never considered.
- **Two unnecessary refusals**, on `conditional_amount`, for moving-average
  sentences the previous reader did not emit that dimension for.
- **Two unnecessary questions** about an amount the sentence states.
- **One case reads backwards.** "buy the index rather than through an ETF"
  settles `assets='ETF'` — the instrument the sentence rejects — accepted on a
  single witness because syntax was silent. Recorded in `MODEL_IS_WRONG`.

The pattern is worth stating plainly: **the safety properties are
reader-independent and the precision properties are not.** Changing provider did
not make the runtime execute anything wrong; it made it refuse and ask less
accurately.

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

## The Discovery baseline, frozen on admissible evidence

The pre-Lean gate is **open**, and what makes that citable is not the verdict
but the artifact behind it: the first one ever produced by CI from the actual
serving stack, carrying every identity the gate checks.

    run                     31603973383
    commit                  a982959
    schema                  quantify-discovery-schema@6/eb01a824e4f43d02
    serving reader          gpt-4.1-2025-04-14@1
    hosted prompt           quantify-hosted-prompt@1
    fusion pipeline         quantify-pipeline@1
    producer                github-actions
    mode                    workflow_dispatch
    draws                   3 per prompt, 36 prompts

    execution_unsafe            0
    silently_reduced_any_draw   0
    silently_reduced            0
    watched_crossed             0
    closure_witness             gpt-4.1-2025-04-14@1

    STABLE_EXECUTABLE 4 · STABLE_REFUSAL 12 · STABLE_CLARIFICATION 2
    UNSTABLE_SAFE 18

`corpus/parser/drift.json` is that artifact, committed rather than described.
It expires: the gate refuses evidence about stochastic behaviour older than
seven days, so this freezes what was measured without letting it be cited
forever.

**Why the two former blockers matter more than the count.** Both now produce
*one distinct executable identity across all three draws*, which is a stronger
statement than "never unsafe". It says the serving path reproduces the same
executable meaning, rather than merely containing stochasticity behind
refusals.

**`UNSTABLE_SAFE` is 18, and the decomposition is the useful part.**

    12  REFUSAL <-> REFUSAL            different stated reasons
     4  CLARIFICATION <-> REFUSAL      different journey shape
     2  CLARIFICATION <-> CLARIFICATION  a different ambiguity asked about

Only the last two vary which question a person is asked. None of it reopens
the safety gate, and none of it is a pre-pilot zero target — it is a UX queue.

## What is open, and stays open

- **Closed, and left here as the shape of the problem.** The gate was blocked
  for weeks on `producer='unknown'`, which read as an operational to-do —
  press *Run workflow*. It was not: `drift-lane.yml` lived on a feature branch
  and GitHub registers `workflow_dispatch` only from the default branch, so
  the lane had never been startable. Merging it, configuring
  `OPENAI_API_KEY`, and dispatching produced the artifact above.
- The gate now pins `hosted_model_id` and checks the closure report's witness.
  It was blind to both, and a provider swap would have walked straight through
  with a GPT drift artifact and a Claude closure report reaching one verdict.
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
