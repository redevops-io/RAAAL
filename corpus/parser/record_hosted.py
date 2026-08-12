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
from src.discovery.readers_quantify import (                       # noqa: E402
    HostedReader, OpenAIReader,
)
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

    # Negative controls for `asset_location`. Naming an account you own is not
    # placing anything in it, and a relation that fires on any sentence
    # mentioning a Roth would refuse ordinary contribution plans by name — a
    # boundary that triggers on the wrong sentences is worse than none, because
    # the refusals read as authoritative.
    "I have a Roth",
    "contribute $500 monthly to my Roth IRA",
)


def strategy_family_sentences() -> list:
    """The twenty-family tier, recorded alongside the rest.

    Kept here rather than in a separate recorder so there is one recording
    store and one drift lane. The question these sentences answer is whether
    the *hosted* reader preserves meanings the deterministic one drops — and
    that question is worthless without recordings, because the answer would
    change under us between runs with nothing saying it had.
    """
    path = Path(__file__).resolve().parent / "strategy_families.json"
    if not path.exists():
        return []
    return [c["text"] for c in json.loads(path.read_text())["cases"]]


def benchmark_sentences() -> list:
    """The strategy evaluation benchmark, recorded alongside the rest.

    One recording store and one drift lane. The benchmark asks whether
    equivalent phrasings agree, which is a question about the model's answer —
    worthless without recordings, because the answer would change under us
    between runs with nothing saying it had.
    """
    path = (Path(__file__).resolve().parent.parent / "benchmark" / "suite.json")
    if not path.exists():
        return []
    return list(json.loads(path.read_text())["prompts"])


def harvested_sentences() -> list:
    """The annotated strategy statements from the harvested corpus.

    Recorded here with everything else, for the same reason: one recording
    store, one drift lane. A harvested sentence with no recording would drop
    silently out of the survival denominator, which raises the rate by exactly
    the sentences nobody managed to read.
    """
    path = (Path(__file__).resolve().parent.parent / "harvested"
            / "annotations.json")
    if not path.exists():
        return []
    return [e["text"] for e in json.loads(path.read_text())["annotations"]]


def wanted() -> list:
    texts = [c.text for c in load() if c.tier == "semantics" and c.language == "en"]
    return sorted(set(texts) | set(ACCEPTANCE) | set(strategy_family_sentences())
                  | set(benchmark_sentences()) | set(harvested_sentences()))


def main(argv: list) -> int:
    refresh = "--refresh" in argv
    subset = 0
    if "--subset" in argv:
        subset = int(argv[argv.index("--subset") + 1])

    # The reader this deployment declares, not a hardcoded one. Recordings are
    # keyed by reader id, so a recorder that always built the Anthropic reader
    # would write one provider's answers under another's key the moment the
    # serving provider changed.
    from src.deploy.context import (PROVIDER_DEFAULT_MODEL,       # noqa: E402
                                    ParserProvider, current)

    declared = current().model
    cls = (OpenAIReader if declared.provider is ParserProvider.OPENAI
           else HostedReader)
    reader = cls(model=declared.model
                 or PROVIDER_DEFAULT_MODEL[declared.provider])
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
