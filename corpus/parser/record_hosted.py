"""Records the hosted reader's reply for every corpus sentence, once.

    python corpus/parser/record_hosted.py            # only what is missing
    python corpus/parser/record_hosted.py --refresh  # re-ask everything
    python corpus/parser/record_hosted.py --subset 8 # the drift lane's sample

Writes `hosted.json` beside the parses. One call per sentence, and only for
sentences that have none, so a normal run costs nothing and a first run costs
the corpus once.

**Why record at all.** The hosted reader is the second independent witness, so
it runs on every utterance rather than only when syntax is silent. That is the
right topology and the expensive one: without recordings, every corpus run
would make one provider call per case and parser CI would depend on network and
provider availability. Recording buys reproducibility; the drift lane buys
noticing when the provider's answer changes.

**What a recording is not.** It is not an answer. It replays what the model
proposed, and fusion decides whether that proposal proceeds exactly as it would
for a live reply. Nothing here or downstream may treat a stored reading as a
settled field because it happens to be on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from corpus.parser.loader import load                              # noqa: E402
from src.discovery.hosted_recording import (                       # noqa: E402
    PROMPT_VERSION, RECORDING_SCHEMA, key, to_json,
)
from src.discovery.readers_quantify import HostedReader            # noqa: E402
from src.discovery.schema import QUANTIFY_SCHEMA                   # noqa: E402

OUT = Path(__file__).resolve().parent / "hosted.json"

#: Sentences that must have a recording whatever the corpus does, because the
#: acceptance cases are about them specifically.
ACCEPTANCE = (
    # The observed syntax failure: Stanza attaches `year end` to *contribute*.
    # If the model reads it correctly this is a real disagreement rather than a
    # manufactured one; if the model agrees with the bad parse, that is itself
    # evidence and means the hosted reader is not an independent witness here.
    "I contribute monthly and rebalance at year end",
    "maintain 60/40 by contributions and rebalance at year end",
    "contribute $500 monthly, rebalanced annually",
    "rebalance to 70/30",
)


def wanted() -> list:
    texts = [c.text for c in load() if c.tier == "semantics" and c.language == "en"]
    return sorted(set(texts) | set(ACCEPTANCE))


def main(argv: list) -> int:
    refresh = "--refresh" in argv
    subset = 0
    if "--subset" in argv:
        subset = int(argv[argv.index("--subset") + 1])

    reader = HostedReader()
    if not reader.available():
        print(f"{reader.api_key_env} is not set; nothing recorded",
              file=sys.stderr)
        return 1

    existing = {}
    if OUT.exists():
        document = json.loads(OUT.read_text())
        existing = {key(e["text"], e["reader_id"]): e
                    for e in document["readings"]}

    texts = wanted()
    if subset:
        # Deterministic sample: every nth sentence rather than a random one, so
        # the drift lane asks the same questions each time and a change in the
        # answer is a change in the model rather than in the sample.
        texts = texts[::max(1, len(texts) // subset)][:subset]

    recorded, called, failures = dict(existing), 0, []
    for text in texts:
        if not refresh and key(text, reader.id) in existing:
            continue
        reading_set = reader.read(text, QUANTIFY_SCHEMA)
        called += 1
        if not reading_set.ok:
            # Recorded rather than dropped. A failure that vanishes looks like a
            # sentence nobody asked about.
            failures.append((text, reading_set.failed))
        recorded[key(text, reader.id)] = to_json(
            reading_set, text, schema_version=QUANTIFY_SCHEMA.version)
        print(f"  {'!' if not reading_set.ok else ' '} {text[:60]}")

    OUT.write_text(json.dumps(
        {"schema": RECORDING_SCHEMA,
         "recorded_with": {"reader_id": reader.id, "model": reader.model,
                           "prompt_version": PROMPT_VERSION,
                           "schema_version": QUANTIFY_SCHEMA.version,
                           "max_tokens": reader.max_tokens},
         "count": len(recorded),
         "note": ("Replayed by the corpus. A recording is not an answer — "
                  "fusion decides whether the model's proposal proceeds, "
                  "exactly as it would for a live reply."),
         "readings": [recorded[k] for k in sorted(recorded)]},
        indent=2, ensure_ascii=False) + "\n")

    print(f"\n{called} call(s); {len(recorded)} recordings -> {OUT}")
    if failures:
        print(f"{len(failures)} failed and were recorded as failures:")
        for text, why in failures:
            print(f"  {text[:50]}: {why[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
