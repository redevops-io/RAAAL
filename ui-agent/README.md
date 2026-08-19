# ui-agent — browser checks for quantify.club

Modelled on `/projects/ui-agent`, and narrowed to the one thing this project
kept getting wrong.

**Why it exists.** The suite has 5500 tests and every one of them reads a
response body. A response can be perfectly correct while the page does
nothing, and in one week that shipped twice:

- `/auth/login` existed, was tested, and redirected to the provider with PKCE.
  No page linked to it. From a browser the deployment was indistinguishable
  from one with no login at all, and was reported as exactly that.
- The strategy selector rendered both dropdowns correctly and its script
  returned at its first line, because the partial is included above the
  textarea it fills and `getElementById` ran during parsing. Choosing a kind
  narrowed nothing; choosing a strategy filled nothing.

Neither is visible in HTML. Both are one click to find.

## Two layers, and the cheap one runs in the suite

### `tests/test_selector_in_a_browser.py` — no server, no credentials

Renders the template, drives it in Chromium, asserts what a person gets. Runs
in the ordinary suite in about four seconds and needs nothing deployed.

| Check | Guards against |
|---|---|
| Choosing a kind narrows the strategy list | the script returning before it attached a listener |
| The remaining group is the kind chosen | narrowing to the wrong group |
| Choosing a strategy writes its sentence into the box, and submits | the same dead script; a menu that orders nothing |
| What lands in the box is ≥ 8 words | the catalogue pasting fragments — "a 60/40 portfolio" is three |
| A narrowed-out strategy cannot be reached by keyboard | hiding `<option>`s with CSS, which several browsers let you arrow past |
| Text somebody typed survives a declined replacement | losing a written sentence to a mis-click |

Verified by restoring the original defect — one line, `readyState` back to
running immediately — and watching six of the seven fail.

### `tests/test_catalogue_sweep.py` — every offered strategy, no browser

Asks whether the *page* is coherent for all 43 catalogue entries under all
three recorded readers — 776 checks in eight seconds, no provider calls,
because the readings are recorded.

| Check | Guards against |
|---|---|
| The table and the button agree | a button offering to fill in blanks the table does not show, which loops when pressed |
| Every question appears as a row | a question that cannot be answered from the page |
| No parameter appears twice | two rows saying different things about one field |
| Every blank shows what to type | a dimension explained at length with nothing to enter |
| Every refusal says why | "something is wrong" without saying what |
| The plan runs, asks, or refuses | the fourth outcome, which nobody can act on and nobody can report |

It found five catalogue strategies whose pages were silent under every reader
— `stated-weights` among them, reachable from the menu — because a dimension
that was both read and refused emitted the settled row and dropped the
refusal. The page showed `60/40` as an ordinary value with no reason and no
sign the plan would not run.

### `catalogue_sweep.py` — every strategy in a browser, with the details changed

What the fast sweep cannot see: a reading can be coherent while the form
rendering it is wired wrong. It also **changes the sentence** — substituting
the amount, the holding and the period — because a strategy that works with
VTI at $500 a month and breaks with NVDA at `200usd` is the surprise a user
finds first. `200usd` is in the variant list because a real submission of
exactly that produced a page offering to fill in a blank it did not show.

```bash
python ui-agent/catalogue_sweep.py --url https://quantify.club \
    --email you@example.com --password '...' --variants 1 --limit 5
```

Variants are read live, so this costs provider calls and belongs on demand
rather than in CI. `--limit` exists so a first run is cheap.

It also carries the three checks on **assumed** values, which only a browser can
make. The catalogue supplies a stand-in portfolio for strategies whose sentence
names no holding, and the whole design rests on that value staying *ours* until
somebody says otherwise. If the input arrives pre-filled, pressing "run it"
posts the guess back and it returns authored `USER` — the strongest author there
is, which no later reading may overwrite — for anybody who did not read the row.

| Check | Guards against |
|---|---|
| No assumed input arrives pre-filled | a click promoting our guess to the user's word |
| An assumed row has an input | a value decided on somebody's behalf |
| The page says "assumed, not stated" | a guess reading as something they said |

The first is invisible to a rendered-template test, which checks the template
rather than the page as served — the same gap that let a dead selector script
ship past a passing suite.

### `strategy_evaluate_sweep.py` — take the defaults, run it, report

The catalogue sweep asks whether each page is *coherent*. This one asks whether
each strategy *runs*: it does what a person does to get a first result — picks a
strategy from the menu, accepts the suggested default for every blank the plan
needs (by **Tab**, the one-keystroke fill the page now offers), presses run — and
reports the outcome, strategy by strategy.

| Outcome | Means | Attention? |
|---|---|---|
| `evaluated` | ran and produced a figure (captured, for the graph comparison next) | clean |
| `refused` | the build declines this strategy by name, with a reason | clean — a named refusal is the honest answer, not a defect |
| `blocked` | a required blank has no default to Tab into — the one-keystroke path cannot complete it | **yes** |
| `unresolved` | everything was filled and the page still neither ran nor refused | **yes** — the outcome nobody can act on |
| `error` | the page threw, or a control the flow depends on was missing | **yes** |

`blocked`, `unresolved` and `error` set a non-zero exit; every strategy is listed
either way, so the run is a map of what the default-fill path completes and what
it does not.

```bash
python ui-agent/strategy_evaluate_sweep.py --url https://quantify.club \
    --email pilot@quantify.club --password '...' [--limit 5] [--url-nav]
```

Each strategy is a live read, so this costs provider calls and runs on demand,
not in CI. `--limit` keeps a first run cheap; `--url-nav` selects by URL rather
than driving the dropdown control (the control's own behaviour is already pinned
by `tests/test_selector_in_a_browser.py`). It signs in through the provider like
the others — and now skips the post-login MFA-setup prompt, which a pilot account
with no authenticator is offered rather than forced.

### `regression_smoke.py` — against the deployed site

A template that is right can still be included on a page that is wrong, and a
provider that answers can still publish the wrong issuer. This asks the
deployed site.

| Check | Guards against |
|---|---|
| The provider publishes an `https` issuer | it published `http://`, so every token would be rejected on a valid signature |
| Signed out, `/workspace/` sends you to sign in | the login governing nothing while a shared password was the real gate |
| A signed-in page links to `/auth/*` | the login that existed for a week with nothing linking to it |
| The offered list has ≥ 30 entries | the library failing to load and the page rendering what it could |
| Choosing a kind narrows the list | the dead script, on the page as served rather than as rendered |
| Choosing a strategy fills the box | the same |
| The sentence is ≥ 8 words | fragments reaching the box |

```bash
python ui-agent/regression_smoke.py --url https://quantify.club
python ui-agent/regression_smoke.py --url https://quantify.club \
    --email you@example.com --password '...'
```

Without credentials the public checks run and the rest print **NOT RUN**
rather than passing. A check that did not execute is not evidence, and the
count at the bottom says so separately from the failures.

Exit 0 only if every check that ran passed.

## Requirements

```bash
pip install playwright && playwright install chromium
```

The pytest file skips itself if either is missing, so a machine without a
browser still runs the rest of the suite — it reports a skip rather than a
pass, which is the same rule the smoke script follows for credentials.

## What this does not do

No LLM, no exploratory crawling, no UX judgement. `/projects/ui-agent` has an
LLM-driven half for that and it is worth adding here later. This half is the
one that can run on every push, and every check in it is a bug we shipped.
