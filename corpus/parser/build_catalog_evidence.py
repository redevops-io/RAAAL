"""Regenerate `src/workspace/catalog_evidence.py` from the recorded readings.

The structured-path table (STATES + OPEN) is what a picked entry means without a
model. It has to say exactly what the *prose* path — the recorded reading of the
entry's own sentence — concludes, or the two doors reach different plans and
`test_catalog_structured_path` fails. Hand-authoring that agreement is how it
drifts. This produces it from the readings instead.

    docker run ... python3 corpus/parser/build_catalog_evidence.py            # write
    docker run ... python3 corpus/parser/build_catalog_evidence.py --check    # diff only

Needs the hosted recordings (`record_hosted.py`) and the syntax parses
(`record_parses.py`) to cover every library sentence; it reads them, never a
provider.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.discovery.hosted_recording import RecordedHostedReader     # noqa: E402
from src.discovery.schema import QUANTIFY_SCHEMA                     # noqa: E402
from src.discovery.syntax_stanza import RecordedReader              # noqa: E402
from src.discovery.witnesses import BOTH                            # noqa: E402
from src.workspace.pilot import read                                # noqa: E402
from src.workspace.strategy_library import LIBRARY                  # noqa: E402

TARGET = REPO / "src" / "workspace" / "catalog_evidence.py"
READER_ID = "gpt-5.4-2026-03-05@1"


def prose_intent(text: str):
    hosted = RecordedHostedReader()
    hosted.recorded_with = dict(hosted.recorded_with, reader_id=READER_ID)
    return read(text, hosted, schema=QUANTIFY_SCHEMA, profile=BOTH,
                syntax_reader=RecordedReader()).intent


def evidence_for(text: str):
    """`(states, open_rows)` a reading concludes for one sentence."""
    intent = prose_intent(text)
    states = {name: field.value for name, field in intent.fields.items()}
    open_rows = tuple(
        (u.dimension, u.reason.value, u.detail, bool(u.result_changing))
        for u in intent.unresolved)
    return states, open_rows


def render(groups) -> str:
    states_lines, open_lines = [], []
    for group in groups:
        states_lines.append(f"    # --- {group.key} "
                            + "-" * max(2, 58 - len(group.key)))
        open_lines.append(f"    # --- {group.key} "
                          + "-" * max(2, 58 - len(group.key)))
        for entry in group.entries:
            states, open_rows = evidence_for(entry.text)
            states_lines.append(
                f"    {entry.key!r}: "
                + "{" + ", ".join(f"{k!r}: {v!r}"
                                  for k, v in states.items()) + "},")
            if not open_rows:
                open_lines.append(f"    {entry.key!r}: (),")
                continue
            open_lines.append(f"    {entry.key!r}: (")
            for row in open_rows:
                open_lines.append(f"        {row!r},")
            open_lines.append("    ),")
    return "\n".join(states_lines), "\n".join(open_lines)


def splice(source: str, states_body: str, open_body: str) -> str:
    """Replace the two dict literals, keep the docstring and the functions."""
    head, _, rest = source.partition("STATES = {\n")
    _, _, after_states = rest.partition("\n}\n")
    open_head, _, open_rest = after_states.partition("OPEN = {\n")
    _, _, after_open = open_rest.partition("\n}\n")
    return (head + "STATES = {\n" + states_body + "\n}\n"
            + open_head + "OPEN = {\n" + open_body + "\n}\n" + after_open)


def main(argv) -> int:
    check = "--check" in argv
    states_body, open_body = render(LIBRARY)
    source = TARGET.read_text()
    updated = splice(source, states_body, open_body)

    if check:
        # Diff the regenerated table against what is committed, entry by entry,
        # so drift in an existing reading is visible rather than silently
        # overwritten.
        import importlib
        mod = importlib.import_module("src.workspace.catalog_evidence")
        importlib.reload(mod)
        drift = 0
        for group in LIBRARY:
            for entry in group.entries:
                new_states, new_open = evidence_for(entry.text)
                old_states = dict(mod.STATES.get(entry.key, {}))
                old_open = tuple(mod.OPEN.get(entry.key, ()))
                if new_states != old_states:
                    drift += 1
                    print(f"STATES  {entry.key}:\n  old={old_states}\n  "
                          f"new={new_states}")
                if new_open != old_open:
                    drift += 1
                    print(f"OPEN    {entry.key}: "
                          f"{len(old_open)} -> {len(new_open)} rows")
        print(f"\n{drift} field(s) drifted")
        return 1 if drift else 0

    TARGET.write_text(updated)
    print(f"wrote {TARGET} "
          f"({sum(len(g.entries) for g in LIBRARY)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
