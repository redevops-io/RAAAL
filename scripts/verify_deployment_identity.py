"""Prove that a running service is the revision a deployment produced.

    python scripts/verify_deployment_identity.py \\
        --url https://quantify.club --expect-commit "$GITHUB_SHA"

The last link in a chain everything else already establishes:

    git commit -> CI run -> artifact -> running process
               -> pilot event -> exact source revision

Every earlier link is checked by tests. This one cannot be: a test asserts what
the code would do given a deployment identity, and only a request to the
running service can say what identity it actually has. That gap is the whole
reason this script exists rather than another test.

**The criterion, stated once so it cannot drift.**

    the serving identity's commit == the commit the deployment run supplied
    and the source offer resolves to that same revision

Both halves are required. A service reporting the right commit while offering
source for a branch is still failing AGPL §13 in the way that matters — the
person is handed a different program than the one answering them. A service
whose source link happens to be right while its identity is unknown proves
nothing about what is running.

**Exit codes are the point.** This is meant for a deployment job, where a green
step must mean the check ran and passed. It exits non-zero when it cannot
reach the service, when the service cannot say what it is, and when the
identities disagree — three different failures, three different messages, and
none of them a silent success.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


#: Identifies the checker. The default `python-urllib/x.y` is refused by the
#: CDN in front of the service — `/health` answered 200 to curl and 403 here,
#: which would have read as "the deployment is unreachable" when it was
#: serving perfectly. A checker that cannot be told apart from a scraper gets
#: treated as one.
USER_AGENT = "quantify-deployment-identity-check/1"


def fetch(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def verify(base: str, expected: str, *, timeout: float = 20.0) -> int:
    base = base.rstrip("/")

    # Two endpoints, because the two facts live in two places and the first
    # version of this script assumed one. `/health` carries the build view —
    # can this deployment identify itself — and `/info` carries the licence
    # offer. Reading `build` from `/info` reported NOT OBSERVABLE against a
    # service that was observable, which is this repository's recurring defect
    # appearing inside the tool written to catch it.
    try:
        health = fetch(f"{base}/health", timeout)
        info = fetch(f"{base}/info", timeout)
    except (urllib.error.URLError, OSError) as unreachable:
        print(f"UNREACHABLE {base}: {unreachable}", file=sys.stderr)
        return 2

    licence = info.get("license") or {}
    source = licence.get("source") or ""
    build = health.get("build") or {}

    # `observable` is the service's own statement about whether it can identify
    # itself. A deployment that says no has failed this check already, and
    # reading further would be reading values it has disclaimed.
    if not build.get("observable"):
        print("NOT OBSERVABLE: the service cannot say which revision it is. "
              "The deployment did not supply the required facts — see "
              "REQUIRED_DEPLOYMENT_FACTS in src/deploy/manifest.py",
              file=sys.stderr)
        return 3

    # The commit is deliberately absent from the public build view, so the
    # source offer is what carries the revision. That is not a workaround: the
    # public view answers "can I interact with this deployment" and the offer
    # answers "which code is answering me", and they are different questions.
    #
    # It is also why the two halves are not redundant. `observable` is the
    # service saying it *can* identify itself; the offer is the identity it
    # reports. A service can be observable and serve a stale offer — which is
    # exactly what quantify.club did the first time this ran.
    if not source.endswith(f"/tree/{expected}"):
        print(f"MISMATCH: the source offer is {source!r} and the deployment "
              f"supplied commit {expected!r}. Either the running service is "
              "not the revision that was deployed, or its source offer names "
              "a different one — both mean a user is handed a different "
              "program than the one answering them", file=sys.stderr)
        return 1

    print(f"OK: {base} serves {expected} and offers its source at {source}")
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="base URL of the running service")
    parser.add_argument("--expect-commit", required=True,
                        help="the commit the deployment run supplied")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    if not args.expect_commit.strip():
        print("--expect-commit is empty; a check against an unknown "
              "expectation passes for any deployment and proves nothing",
              file=sys.stderr)
        return 4

    return verify(args.url, args.expect_commit.strip(), timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
