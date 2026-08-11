"""Post-deploy acceptance, run against a live deployment.

    python deploy/acceptance.py https://quantify.example.com
    python deploy/acceptance.py https://quantify.example.com --record FILE

Prose checklists get skimmed. This performs the checks and prints what it
found, so "we verified the deployment" is a transcript rather than a memory.

It exercises the public surface only — no database access, no imports from
`src` — because that is what a user reaches and what an operator can run from
anywhere. Anything requiring the database is in the runbook's restore drill.

`--record` writes that transcript to a file as the pilot's deployment
evidence. **It is half the evidence, by construction.** This script cannot see
the startup-proof facts — the commit, image digest, migration head and
snapshot id — because `/health` deliberately withholds them, which is a
property this script itself checks four lines at a time. Publishing them here
to make one tidy record would mean weakening the thing being recorded.

So the other half is captured by the operator from the private channel:

    docker compose logs api | grep "deployment proof"

Two records, joined by whoever ran them. See docs/Runbook.md § Deployment
evidence.

Exit code 0 means every check passed. Nothing here writes except `--record`.
"""
from __future__ import annotations

import datetime
import json
import sys
import urllib.error
import urllib.request

TIMEOUT = 20

#: The facts `/health` must not publish, and the tokens an error must not carry.
#: Named once because the checks below and the inventory beneath them are both
#: built from these — two hand-written copies would drift, and the copy that
#: drifts silently is the one describing what was *not* run.
PRIVATE_BUILD_FACTS = ("image_digest", "commit", "migration_head", "snapshot_id")
LEAK_TOKENS = ("psycopg", "sqlite3", "Traceback", "DETAIL:", "postgresql://")

#: Every check this script performs, in order, whether or not it gets that far.
#: A record of an abandoned run has to say what was skipped: "one check, failed"
#: reads as a one-check suite rather than a sixteen-check suite that stopped.
ALL_CHECKS = (
    "liveness answers",
    "readiness reports ready",
    "readiness says nothing about why",
    "build reports observable",
    *(f"build does not publish {fact}" for fact in PRIVATE_BUILD_FACTS),
    "personalisation is off",
    "the private surface requires a credential",
    "an error carries a correlation id",
    *(f"an error does not leak {token!r}" for token in LEAK_TOKENS),
)

#: Final outcomes, and the category of failure when there is one.
PASSED = "PASSED"
UNREACHABLE = "UNREACHABLE"
CHECKS_FAILED = "CHECKS_FAILED"
INVENTORY_DRIFTED = "INVENTORY_DRIFTED"


def fetch(base, path, headers=None):
    """Never raises. A status of 0 means the deployment was not reachable.

    An unreachable host is a result, not a crash. Left to propagate, a wrong
    URL or a refusing instance produced a forty-line traceback and — worse —
    `--record` never ran, so the one run that most needed recording, the one
    where the deployment would not serve, was the one that left no evidence.
    """
    # Identify the tool. Behind a CDN with bot protection, the default
    # `Python-urllib/3.x` agent is refused before the request reaches the
    # application — Cloudflare returns 403 with error 1010, and every check
    # then reads the block page instead of the deployment. Five checks failed
    # against a site that was serving correctly, which is a checklist
    # reporting on the wrong thing rather than a broken deployment.
    sent = {"User-Agent": "quantify-acceptance/1 (+deploy/acceptance.py)",
            "Accept": "*/*"}
    sent.update(headers or {})
    request = urllib.request.Request(base.rstrip("/") + path, headers=sent)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace"), \
                dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), \
            dict(error.headers)
    except Exception as error:                                   # noqa: BLE001
        # URLError, timeouts, DNS, TLS verification, a reset connection.
        return 0, f"unreachable: {type(error).__name__}: {error}", {}


def _write_record(record_to, base, results, outcome, category=None):
    """The pilot's deployment evidence: what was checked, and what answered.

    A failed run is recorded too, and an abandoned one. Evidence that only
    exists when the answer was good is not evidence — the question a month from
    now is what configuration was live when the cohort began, and "we re-ran it
    until it passed" is part of that answer.

    The record has to be readable without the script beside it, so it carries
    what was *not* attempted as well as what was. A run that stopped at the
    first check and recorded only that check describes itself as a one-check
    suite, which is the reading that makes an abandoned run look like a
    thorough one.
    """
    attempted = [name for name, _, _ in results]
    written = {
        "target": base,
        "checked_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "outcome": outcome,
        "failure_category": category,
        "passed": outcome == PASSED,
        "checks": [{"name": name, "passed": passed, "detail": detail}
                   for name, passed, detail in results],
        "not_attempted": [name for name in ALL_CHECKS if name not in attempted],
        "checks_declared": len(ALL_CHECKS),
        "startup_proof": (
            "NOT CAPTURED HERE. The build identity, migration head and "
            "snapshot id are withheld from /health on purpose — this script "
            "checks that they are. Capture them from the operator log "
            "(`docker compose logs api | grep \"deployment proof\"`) and keep "
            "that output beside this file. See docs/Runbook.md § Deployment "
            "evidence."),
    }
    with open(record_to, "w", encoding="utf-8") as handle:
        json.dump(written, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nrecorded to {record_to}")


def main(base, record_to=None):
    results = []

    def check(name, passed, detail=""):
        results.append((name, passed, detail))

    status, body, _ = fetch(base, "/health/live")
    check("liveness answers", status == 200,
          body if status == 0 else f"status {status}")
    if status == 0:
        # Every later check would fail for the same single reason, and sixteen
        # failures read like sixteen problems. Say the one true thing.
        print(f"FAIL  {base} is not reachable — {body}")
        skipped = len(ALL_CHECKS) - len(results)
        print(f"\n{skipped} of {len(ALL_CHECKS)} checks were not attempted. "
              "Fix reachability and re-run.")
        if record_to:
            _write_record(record_to, base, results, UNREACHABLE, UNREACHABLE)
        return 1

    status, body, _ = fetch(base, "/health/ready")
    ready = status == 200 and json.loads(body or "{}").get("ready") is True
    check("readiness reports ready", ready, f"status {status} body {body[:120]}")
    check("readiness says nothing about why",
          "migration" not in body.lower() and "postgres" not in body.lower(),
          "a client learns ready or not, never which check failed")

    status, body, _ = fetch(base, "/health")
    try:
        health = json.loads(body)
    except ValueError:
        health = {}
    build = health.get("build", {})
    check("build reports observable", build.get("observable") is True,
          "all four build stamps must be set in production")
    for private in PRIVATE_BUILD_FACTS:
        check(f"build does not publish {private}", private not in build,
              "these describe how to attack the deployment")

    status, body, _ = fetch(base, "/info")
    info = json.loads(body) if status == 200 else {}
    check("personalisation is off",
          info.get("personalization", {}).get("enabled") is False,
          "the publisher's exclusion depends on impersonal output")

    status, body, _ = fetch(base, "/workspace/")
    check("the private surface requires a credential", status in (401, 403),
          f"status {status} — the pilot workspace must not be open. Run this "
          "against the public URL: pointed straight at the application it "
          "bypasses the proxy that holds the credential, and passes nothing")

    # An unauthenticated path that still reaches the application.
    #
    # This probed `/workspace/nonexistent-page`, which the proxy rejects with a
    # 401 and an empty body before the application sees it. The correlation
    # check then failed — correctly, there was no application response — and,
    # worse, the five leak checks below passed against an empty string. Five
    # green ticks proving nothing is the more dangerous half of that.
    status, body, headers = fetch(base, "/nonexistent-page")
    # Case-insensitively: the header goes out lowercased on the wire, and a
    # check that only matched the spelling in our source would have reported a
    # missing correlation id on a deployment that had one.
    lowered = {name.lower() for name in headers}
    check("an error carries a correlation id",
          "request_id" in body.lower() or "x-request-id" in lowered,
          "a user must be able to quote something an operator can find")
    for leak in LEAK_TOKENS:
        check(f"an error does not leak {leak!r}", leak not in body)

    # The inventory and the run must agree. `not_attempted` is derived by
    # subtracting one from the other, so a check added above and not declared
    # in ALL_CHECKS would quietly make every abandoned record understate what
    # it skipped — a wrong answer in exactly the file kept as evidence.
    drift = set(name for name, _, _ in results).symmetric_difference(ALL_CHECKS)
    if drift:
        check("the checklist inventory matches the run", False,
              f"declared and executed sets differ: {sorted(drift)}")

    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, passed, detail in results:
        mark = "ok  " if passed else "FAIL"
        print(f"{mark}  {name.ljust(width)}  {detail if not passed else ''}")
        failed += 0 if passed else 1

    if record_to:
        _write_record(
            record_to, base, results,
            PASSED if not failed else CHECKS_FAILED,
            None if not failed
            else INVENTORY_DRIFTED if drift else CHECKS_FAILED)

    print()
    if failed:
        print(f"{failed} of {len(results)} checks failed")
        print("Do not invite users until these pass. See docs/Runbook.md.")
        return 1
    print(f"all {len(results)} checks passed")
    print()
    print("Still to do by hand — they need a browser and a person:")
    print("  * run both launch journeys to a worksheet with a figure")
    print("  * confirm the synthetic-data notice appears on every page")
    print("  * confirm a saved plan names the model that interpreted it")
    print("  * take and restore one backup (docs/Runbook.md § Backup and restore)")
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    record_to = None
    if "--record" in arguments:
        index = arguments.index("--record")
        if index + 1 >= len(arguments):
            print("--record needs a path")
            raise SystemExit(2)
        record_to = arguments[index + 1]
        arguments = arguments[:index] + arguments[index + 2:]
    if len(arguments) != 1:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(arguments[0], record_to))
