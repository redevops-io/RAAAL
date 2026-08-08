#!/usr/bin/env python3
"""Fail if the vendored contract has drifted from its source.

`src/contracts/` is a copy of `runtime-contracts`, taken because that package
is private and this repository is public, so a clean clone cannot resolve it as
a dependency. A copy nobody compares is a fork, and a private fork of a contract
two runtimes are supposed to agree on is exactly what the contract package
exists to prevent.

Run in CI and before a deploy:

    python scripts/check_vendored_contracts.py [--source /path/to/runtime-contracts]

Exit codes:

    0  identical, or the source is not present on this machine (see below)
    1  drifted — the copy and the source disagree
    2  the copy is missing or unreadable

**A missing source is not a failure.** The source lives outside this repository
and will be absent on most machines, including CI. Failing there would train
everyone to ignore this check, which is worse than not having it. It reports
SKIPPED and says so loudly; the check that matters runs where the source is.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VENDORED = HERE / "src" / "contracts"
DEFAULT_SOURCE = Path("/projects/runtime-contracts")

#: vendored filename -> path within the source repository
FILES = {
    "intent.py": "runtime_contracts/models/intent.py",
    "canonical.py": "runtime_contracts/canonical.py",
}

#: The one edit the copy is allowed: `..canonical` has no parent package here.
REWRITES = ((b"from ..canonical import", b"from .canonical import"),)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), type=Path)
    args = parser.parse_args()

    if not VENDORED.exists():
        print(f"MISSING  {VENDORED} does not exist", file=sys.stderr)
        return 2

    if not args.source.exists():
        print(f"SKIPPED  {args.source} is not on this machine, so the vendored "
              "contract could not be compared. This is expected off the "
              "authoring machine and is not a pass.")
        return 0

    drifted = []
    for name, relative in FILES.items():
        copy = VENDORED / name
        origin = args.source / relative
        if not copy.exists() or not origin.exists():
            drifted.append((name, "missing on one side"))
            continue

        expected = origin.read_bytes()
        for before, after in REWRITES:
            expected = expected.replace(before, after)

        actual = copy.read_bytes()
        if actual != expected:
            drifted.append(
                (name, f"copy {digest(actual)} != source {digest(expected)}"))

    if drifted:
        print("DRIFTED  the vendored contract no longer matches its source:")
        for name, why in drifted:
            print(f"           {name}: {why}")
        print("\n  Fix it in runtime-contracts and re-copy. Editing the copy "
              "forks a contract two runtimes must agree on.")
        return 1

    print(f"OK       {len(FILES)} vendored file(s) match {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
