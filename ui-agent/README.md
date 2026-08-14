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
