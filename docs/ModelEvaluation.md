# Stage-1 model evaluation — first bounded run

**Date:** 2026-08-01 · **Model:** `claude-sonnet-5` (resolved id as returned by
the provider) · **Compiler:** commit `7b3f9f9`, version 3 · **Cases:** 205 ·
**Calls:** 205 · **Retries:** 0 · **Cap:** 215

41 families × 5 wordings: three semantically identical (extraction accuracy and
paraphrase convergence), one contradictory, one underspecified.

Two things were true before this ran, and they are why the result is
interpretable rather than anecdotal: paraphrases already converged to one
canonical Mission (100%), and specifications already round-tripped without
identity or provenance drift (100%). A failure here is attributable to **model
extraction**.

---

## 1. Against the targets

| Metric | Target | Measured | |
|---|---:|---:|---|
| `rule_hash` exact | ≥ 95% | **99.5%** | 204/205 |
| `schedule_hash` exact | ≥ 95% | **99.0%** | 203/205 |
| `content_hash` exact | ≥ 90% | **98.0%** | 201/205 |
| Field provenance exact | — | **99.5%** | 204/205 |
| False inference | ≤ 1% | **0.98%** | 2/205 |
| Contradiction recall | 100% | **100%** | 41/41 |
| Family convergence (3/3) | ≥ 90% | **92.7%** | 38/41 |

**Hard gates, all clean:** 0 saveable Missions with open questions, 0 schema
failures, 0 retries, 0 silent recommendations, 100% of responses passed
deterministic validation before simulation.

Operational: p50 5.7 s, p95 12.1 s, p99 15.5 s. 661 input / 577 output tokens
per call; 135,568 / 118,283 total.

---

## 2. The quarantine did real work

145 model proposals were refused:

| Refusals | Why |
|---:|---|
| 90 | a ticker that does not appear in the description |
| 36 | the field was already recognised |
| 14 | a value outside the vocabulary *(this one was our bug — see §4)* |
| 5 | not a field stage 1 recognises |

Ninety attempts to resolve a company name to a symbol the user never wrote. That
is the check that exists to stop a scenario pricing the wrong security, and it
fired on 44% of calls.

The model contributed 34 readings the phrase rules missed — 33 `funding_source`
and one `weighting`.

---

## 3. Three divergences, three different causes

**Two are genuine false inferences.** `WM-0012#s1` and `WM-0037#s1` describe a
plain monthly contribution with no conditional buying at all, and the model set
`funding_source: additional_cash`. Nothing in either description mentions where
extra money comes from. This is the field with the clearest financial
consequence — as additional cash the plan invests more, and more money in a
rising market always looks like a better rule.

**One says the deterministic compiler is wrong.** `WM-0009#CONT` — "I buy $500
of SPY every week" — compiled to *no assets at all* under the phrase rules,
because `SPY` sits on a reserved list that stops it being read as a holding. It
is on that list because it is usually the *signal* in a trend rule. The model
read it as the holding, which is what the sentence says. The deterministic
reading is the defective one.

---

## 4. Two harness defects, and one product one

The first summary reported **family convergence 0%** and **provenance exact
19%**. Both were the harness.

Convergence was measured across all five wordings per family, including the
contradictory and underspecified ones that are *meant* to differ. Provenance
compared quoted spans rather than field provenance — two parsers reading one
sentence legitimately quote different substrings, so it measured phrasing.

Full capture is what let both be corrected without spending the budget twice.

A third defect was real: the quarantine rejected `sells_allowed: "False"` on
capitalization, because the vocabulary is lowercase and a JSON boolean
serializes capitalized. It refused a **correct** reading of "I don't sell
anything" fourteen times. Values now compare case-insensitively; field names
stay exact.

---

## 5. The finding with product consequences

**80.5% of cases (165/205) gained an extra question** the deterministic compiler
does not ask — almost always a note that the account type ("my brokerage
account") maps to nothing in the vocabulary.

The model is not wrong. Account type genuinely is not a field stage 1 can
represent, so it correctly declines to place it. But every one of those notes
becomes a user-facing question, so four in five plans would show a question
about something the user already said clearly.

That is a vocabulary gap, not a model failure, and it is the single largest
difference between the deterministic and model-assisted paths.

---

## 6. Verdict

On the error distribution: **mostly exact, with rare uncertainty** — so model
stage 1 is usable behind confirmation, which is where it already sits.

Before exposing it in the pilot:

1. add `account_type` to the vocabulary, or suppress `unclear` notes for
   phrases the compiler deliberately does not model — 80.5% extra questions is
   not shippable friction;
2. decide whether `funding_source` should be model-readable at all, given both
   false inferences landed there;
3. remove `SPY` from the reserved list when it is the object of a purchase.

Cross-model comparison comes after those, on the same 205 cases.

---

## 7. What was fixed as a result

All three, in the commit after the run.

**Account type is now in the vocabulary.** `tax_treatment` had always been on
the scenario and in the content hash, and nothing ever set it — so every plan
compiled from prose was `NONE_APPLIED`, and a Roth compared as identical to a
taxable account. This project's founding example of a defect was still live for
the entire conversational path. Taxable, Roth IRA, Roth 401(k), traditional IRA
and 401(k) are now read and represented; an account the compiler cannot place
(a donor-advised fund, an inherited IRA, "my retirement accounts") is asked
about rather than guessed, because guessing between traditional and Roth is
precisely the defect.

**SPY is a holding when it is bought.** Written as a signal test rather than a
list of purchase verbs — the first attempt enumerated buy/put/invest and missed
"goes into", which the stability benchmark caught within one run.

**`funding_source` was kept model-readable.** Both false inferences landed
there, but that is an argument that the field is economically important, not
that it should be removed: it changes invested capital, benchmark equivalence,
the TWR/MWR reading and the schedule hash. Better to ask one more question than
to assume.

### The metric that replaced the hash comparison

    accepted without a question      69.3%   (1,597 of 2,304)
    accepted after 1-2 questions     30.7%
    needs three or more questions     0.0%
    silently changed                  0.0%

"rule_hash exact 99.5%" answers a question nobody asks. This one is what a user
lives with.

---

## 8. Reproducing

```bash
python3 scripts/run_model_eval.py --max-calls 215   # billable, ~21 minutes
python3 scripts/analyze_model_eval.py               # offline, from the bundle
```

`reports/modeleval/bundle.jsonl` holds one full capture record per case: source,
expected Mission, model response hash, accepted and rejected proposals, actual
Mission, typed diff, tokens, latency and pins.
