"""Re-record the handful of readings a schema-prompt change orphaned.

`record_hosted.py --refresh` re-asks `wanted()`. A few sentences carry recordings
but are not in `wanted()`: route and replay tests read them through the recorded
reader. A change to the reader's prompt (a new `allocation_method` value list)
moves the question digest, so their stored `@6` readings stop answering the
current question and every path that reads one raises. This re-asks exactly those
— under the reader whose id they carry — and stamps them with the current schema
and its digest.

The `@2` entries are left alone on purpose: they are the deliberately-unproven
controls `test_unsupported_families` checks carry no digest.

    docker run -e OPENAI_API_KEY -e QUANTIFY_PARSER_PROVIDER=openai \\
        -e QUANTIFY_PARSER_MODEL=gpt-5.4-2026-03-05 ... \\
        python3 corpus/parser/refresh_orphans.py

Run once per reader the corpus carries (gpt-5.4, gpt-4.1, claude-sonnet-5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.parser.record_hosted import wanted                   # noqa: E402
from src.discovery.hosted_recording import (                     # noqa: E402
    key, question_digest, to_json,
)
from src.discovery.readers_quantify import (                     # noqa: E402
    configured_hosted_reader,
)
from src.discovery.schema import QUANTIFY_SCHEMA                  # noqa: E402

OUT = Path(__file__).resolve().parent / "hosted.json"
STALE_VERSION = "quantify-discovery-schema@6"


def main() -> int:
    reader = configured_hosted_reader()
    if not reader.available():
        print(f"{reader.api_key_env} is not set; nothing recorded",
              file=sys.stderr)
        return 1

    document = json.loads(OUT.read_text())
    by_key = {key(e["text"], e["reader_id"]): e for e in document["readings"]}
    covered = set(wanted())
    targets = [e for e in document["readings"]
               if e["schema_version"] == STALE_VERSION
               and e["text"] not in covered
               and e["reader_id"] == reader.id]
    print(f"{len(targets)} orphan reading(s) to refresh under {reader.id}")

    for entry in targets:
        reading_set = reader.read(entry["text"], QUANTIFY_SCHEMA)
        by_key[key(entry["text"], reader.id)] = to_json(
            reading_set, entry["text"],
            schema_version=QUANTIFY_SCHEMA.version,
            question=question_digest(QUANTIFY_SCHEMA))
        print(f"  {'!' if not reading_set.ok else ' '} {entry['text'][:55]}")

    document["readings"] = [by_key[k] for k in sorted(by_key)]
    document["count"] = len(document["readings"])
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"updated -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
