"""Round-trip fidelity: Mission -> words -> the same Mission.

    python3 scripts/run_roundtrip.py

Only SPECIFICATION claims losslessness. SUMMARY reports what it drops rather
than pretending to be complete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.catalog import load_strategies                # noqa: E402
from src.loadtest.paraphrase import Klass, corpus               # noqa: E402
from src.loadtest.roundtrip import cycles, run                  # noqa: E402
from src.mission.render import Purpose                          # noqa: E402

OUT = Path("reports/loadtest")


def main() -> int:
    texts = [p.text for p in corpus(load_strategies(), 16)
             if p.klass in {Klass.COMPLETE, Klass.PERSISTENT_VS_EVENT,
                            Klass.EQUAL_WEIGHT, Klass.FUNDING_SOURCE,
                            Klass.CALENDAR_VS_SESSION}]
    print(f"{len(texts):,} missions\n")
    report = run(texts)
    summary = report.summarize()

    for purpose, stats in summary.items():
        claims = Purpose(purpose).claims_lossless
        print(f"  {purpose:14} {stats['exact_rate']:6.1f}% exact   "
              f"rule {stats['rule_hash_kept']:,}/{stats['n']:,}   "
              f"schedule {stats['schedule_hash_kept']:,}/{stats['n']:,}"
              f"   {'(claims lossless)' if claims else '(may omit)'}")

    losses = report.losses(Purpose.SUMMARY)
    if losses:
        print("\n  fields a summary drops")
        for path, count in list(losses.items())[:8]:
            print(f"    {count:>6,}  {path}")

    drifted = [t for t in texts[:400] if len(set(cycles(t, 3))) != 1]
    print(f"\n  identity drift over 3 cycles: {len(drifted)}/400")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "roundtrip.json").write_text(json.dumps({
        "missions": len(texts), "summary": summary,
        "summary_losses": losses, "drifted": len(drifted),
    }, indent=2) + "\n")
    print(f"\nwrote {OUT}/roundtrip.json")
    return 0 if summary[Purpose.SPECIFICATION.value]["exact_rate"] == 100.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
