"""Records Stanza's parse of every corpus sentence, so tests need no model.

    python corpus/parser/record_parses.py            # every language present
    python corpus/parser/record_parses.py en de      # only these

Writes `parses.json` beside `cases.json`. Each entry carries the parser and
model version that produced it, because a recording whose provenance is unknown
cannot be checked against anything later — and an unchecked recording is worse
than no recording, since it keeps the tests above it green while they measure
whatever the fixture happens to say.

The download is deliberate and separate: models are fetched by
`stanza.download`, not by this script, so recording never silently pulls half a
gigabyte on someone's laptop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from corpus.parser.loader import load  # noqa: E402
from src.discovery.syntax_stanza import StanzaReader, key, to_json  # noqa: E402

OUT = Path(__file__).resolve().parent / "parses.json"


def main(languages: list) -> int:
    cases = load()
    wanted = sorted({c.language for c in cases}
                    if not languages else set(languages))

    existing = {}
    if OUT.exists():
        document = json.loads(OUT.read_text())
        existing = {key(e["text"], e["language"]): e
                    for e in document["parses"]}

    recorded_with = {}
    entries = dict(existing)
    for language in wanted:
        texts = sorted({c.text for c in cases if c.language == language})
        try:
            reader = StanzaReader(language)
            reader._load()
        except Exception as failure:                      # noqa: BLE001
            # Named rather than swallowed. A language quietly absent from the
            # recording is a tier of the corpus quietly not running.
            print(f"  {language}: SKIPPED — {type(failure).__name__}: "
                  f"{str(failure)[:120]}")
            print(f"      run: python -c \"import stanza; "
                  f"stanza.download('{language}')\"")
            continue

        for text in texts:
            entries[key(text, language)] = to_json(
                reader.parse(text, language), text)
        recorded_with[language] = {"parser": f"stanza@{reader._version}",
                                   "model": f"{language}/tokenize,pos,lemma,depparse"}
        print(f"  {language}: {len(texts)} sentences")

    if not recorded_with:
        print("nothing recorded; no model was loadable")
        return 1

    OUT.write_text(json.dumps(
        {"schema": "quantify-parse-recording@1",
         "recorded_with": recorded_with,
         "count": len(entries),
         "parses": [entries[k] for k in sorted(entries)]},
        indent=2, ensure_ascii=False) + "\n")
    print(f"{len(entries)} parses -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
