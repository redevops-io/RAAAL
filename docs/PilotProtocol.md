# Pilot protocol

Five to ten people. The objective is not conversion and not a demo — it is to
find out whether governed, replayable execution makes a workflow more useful
than a normal chatbot or agent.

## What the deployment must declare

    QUANTIFY_PARSER_MODE=RUNTIME
    QUANTIFY_PILOT_READER=HOSTED
    QUANTIFY_PARSER_MODEL=claude-sonnet-5
    ANTHROPIC_API_KEY=…

`RUNTIME` is a declaration, not a flag: without it `/workspace/new` serves the
legacy interpreter and the pilot measures the wrong product. The preflight
refuses a `RUNTIME` deployment with no key rather than passing startup and
refusing every request.

`HOSTED`, not `RECORDED`. A deployment replaying fixtures would serve answers
from a file and report them as the model's.

## The supported journey

    describe → clarify → save → figure → reopen → the same figure

That is the Era 1 launch journey, unchanged, and
`tests/test_pilot_journey.py` runs it. Anything outside it — proposals,
observations, counterfactuals, rerun — is on the legacy path and is not part of
this experiment.

## Freeze

No new behaviour before the cohort, with two exceptions: a correctness defect,
or something that blocks the supported journey. Everything else waits for
evidence that it matters.

## What the telemetry answers

`pilot_events.summary()`, at a prompt. Counts and names, each carrying the
deployment profile so a model-only cohort stays separable from a later
dual-witness one:

- which capabilities were refused, by name and count
- plans saved, and plans saved with questions still open
- plans reopened
- results produced, against refusals produced

Every one of those is a thing that happened. None is a claim about what anyone
thought of it.

## What only the interview answers

Same five questions for everyone, asked after the task, in this order. The
order matters: expectation before interpretation, because asking what they
thought the result meant first will make them reconstruct an expectation to
match it.

1. **What did you expect it to do** when you typed your sentence?
2. **What did you think the result meant?**
3. **Was any question or refusal confusing?** Which one, and what did you think
   it was asking?
4. **Did the evidence make the result more trustworthy** — the reader, the
   pinned intent, the fact that reopening recomputes from what you confirmed?
   Or was it noise?
5. **What would you try next** if you were using this for real?

Ask them the same way each time and write the answers down verbatim. A summary
written from memory is a summary of what the interviewer expected to hear.

**Do not** turn these into a form, a rating, or a telemetry field. A number
derived from a five-point scale on "was this trustworthy" is a number that will
be quoted later without the conversation that produced it, and the conversation
is the evidence.

## What would make this a failure worth having

The pilot succeeds as an *experiment* if it produces a clear answer, including
a negative one. Specifically these are all useful outcomes:

- people ask for capabilities the manifest refuses, repeatedly and by name —
  the refusal boundary is right and the manifest is too small
- people do not care that reopening replays — the runtime's distinctive
  property is not the one that matters to them
- people abandon at the clarification question — the interpretation is right
  and the interaction is wrong
- nobody wants to type a sentence at all — the surface is wrong, which no
  amount of runtime work would have discovered

The outcome that would waste the pilot is a set of numbers with no
conversations attached, because the questions worth answering are not the ones
the system can count about itself.
