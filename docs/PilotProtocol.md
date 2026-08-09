# Pilot protocol

Five to ten people. The objective is not conversion and not a demo — it is to
find out whether governed, replayable execution makes a workflow more useful
than a normal chatbot or agent.

## What the deployment must declare

    QUANTIFY_PARSER_MODE=RUNTIME
    QUANTIFY_PILOT_READER=HOSTED
    QUANTIFY_PARSER_MODEL=claude-sonnet-5
    ANTHROPIC_API_KEY=…
    QUANTIFY_PILOT_TRANSCRIPTS=yes      # see "What is retained" below

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

## The six observations, and where each one comes from

These are the things worth knowing about a participant. They do not all come
from the same place, and the table exists because treating them as if they did
is how a study ends up with a number standing in for a conversation.

| Observation | Source | Honest? |
|---|---|---|
| the exact prompt they entered | `pilot_transcripts`, verbatim | yes, if retention was declared |
| whether the runtime asked an unnecessary clarification | **interview**, with `clarifications_answered_from_the_prompt` pointing at which transcripts to read | the counter is a proxy, not the finding |
| whether the refusal matched what they expected | **interview only** | nothing mechanical can answer this |
| whether they edited and resubmitted | `plan_resubmitted`, with `text_changed` | yes |
| whether they reopened saved plans | `plan_reopened` | yes |
| whether they went back to the legacy workspace | `left_for_legacy` | yes |

Four are counted, one is a pointer, one is a conversation.

### The proxy, stated plainly

`clarifications_answered_from_the_prompt` counts the times somebody answered a
question by repeating a phrase their sentence already contained. If a
participant wrote *invest $500 monthly into VTI*, was asked what to hold, and
typed *VTI*, the reading missed something that was in front of it — a fact about
the parse, not a judgement about the person.

It cannot see the opposite error, a question that should have been asked and was
not, and it misses anyone who answers a redundant question in different words.
So it is not the finding. A capability appearing there repeatedly is a parser
defect with a name attached, and it is worth chasing precisely because it
arrives without anybody being asked anything.

## What is retained, and what people are told

Two stores, kept apart on purpose.

    pilot_events        counts and codes. No prose. Needs no permission.
    pilot_transcripts   what someone actually typed. Off unless declared.

    QUANTIFY_PILOT_TRANSCRIPTS=yes      keep participants' sentences
    QUANTIFY_PILOT_TRANSCRIPT_DAYS=30   and drop them after this long

Off by default, and only an explicit affirmative turns it on — a typo in that
variable must fail towards *not* keeping what people typed.

Requests carry an opaque participant token in a cookie. It is not identity: no
name, no address, nothing derived from the request. It exists because a revision
chain is invisible without it, and because forty compiles is a busy pilot or one
person struggling and those are opposite conclusions.

**Tell the cohort this, in these words**, before they start:

> Quantify records what you type here and what the system did with it, so the
> conversation afterwards is about your actual sentences rather than anyone's
> memory of them. It is kept for thirty days. It is not linked to your name, and
> if you want it deleted at any point, say so and it is deleted.

`pilot_session.forget(participant)` is that deletion, and it works today —
written before the pilot rather than after somebody asks.

## What the telemetry answers

`pilot_events.summary()`, at a prompt. Counts and names, each carrying the
deployment profile so a model-only cohort stays separable from a later
dual-witness one:

- which capabilities were refused, by name and count
- plans saved, and plans saved with questions still open
- plans reopened
- results produced, against refusals produced
- resubmissions, and how many changed the wording
- departures to the legacy workspace
- how many participants all of the above is spread across

Every one of those is a thing that happened. None is a claim about what anyone
thought of it.

Read `participants` first. Every other number is ambiguous without it.

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
