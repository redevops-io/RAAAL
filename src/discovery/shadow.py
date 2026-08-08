"""Two readers, one sentence, and what to do when they differ.

The shadow phase's whole purpose is to make disagreement *visible and
countable* before anything depends on the new reader. So this module resolves
nothing. It records what each reader saw, marks where they differ, and hands
the difference on — to a human during Phase 3, and to the corpus afterwards.

**No reader is privileged.** There is no precedence table here and there must
not be one. This project has the counter-example in both directions: on
"crosses below" the model was right and the regex was wrong; on cadence and
window collisions the regex has been right where a model would happily have
agreed with the user's phrasing. A rule that picked a winner would have been
wrong in one of those cases and nobody would have noticed.

**Silence is not agreement.** A reader that did not mention a dimension has not
endorsed the other reader's view of it. `ONE_SIDED` exists so the count of
"agreements" does not quietly include every dimension one reader never looked
at — which is the easiest way to make a shadow phase report success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from .reader import Reading, ReadingSet, Schema

AGREED = "AGREED"
"""Both read it, and read it the same."""

CONTESTED = "CONTESTED"
"""Both read it, differently. The only state that needs a person."""

ONE_SIDED = "ONE_SIDED"
"""One read it, the other did not mention it. Not agreement, and not a fight:
it is a coverage difference, and it is the most common thing a new reader does
that an old one did not."""

UNREAD = "UNREAD"
"""Neither read it. The sentence probably does not say."""


@dataclass(frozen=True)
class FieldComparison:
    dimension: str
    state: str
    values: Mapping[str, Any] = field(default_factory=dict)
    """reader_id -> value, for whoever read it."""

    spans: Mapping[str, str] = field(default_factory=dict)

    @property
    def readers(self) -> Sequence[str]:
        return tuple(sorted(self.values))

    def to_json(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "state": self.state,
                "values": dict(self.values), "spans": dict(self.spans)}


@dataclass(frozen=True)
class Comparison:
    """What two or more readers made of one sentence."""

    text: str
    fields: Sequence[FieldComparison] = ()
    failed_readers: Mapping[str, str] = field(default_factory=dict)
    """reader_id -> why. Kept out of every count below: a transport failure is
    not a reading, and scoring it as disagreement would make an outage look
    like a semantic problem."""

    def by_state(self, state: str) -> Sequence[FieldComparison]:
        return tuple(f for f in self.fields if f.state == state)

    @property
    def contested(self) -> Sequence[FieldComparison]:
        return self.by_state(CONTESTED)

    @property
    def usable(self) -> bool:
        """Whether this comparison says anything at all.

        False when fewer than two readers actually read something. A reader
        that returned successfully and read nothing is as useless for
        comparison as one that crashed, and only the second is in
        `failed_readers` — so counting failures alone would call a comparison
        usable when one side contributed no opinion at all.
        """
        contributing = {reader for f in self.fields for reader in f.values}
        return len(self.failed_readers) == 0 and len(contributing) >= 2

    def to_json(self) -> Dict[str, Any]:
        return {"text": self.text,
                "fields": [f.to_json() for f in self.fields],
                "failed_readers": dict(self.failed_readers),
                "usable": self.usable,
                "counts": {state: len(self.by_state(state))
                           for state in (AGREED, CONTESTED, ONE_SIDED, UNREAD)}}


def _same(left: Any, right: Any, mode: str = "TEXT") -> bool:
    """Whether two readings mean the same thing.

    Two rules, and the line between them is deliberate.

    **Numbers compare as numbers.** The first live run scored `"1000"` against
    `"$1,000"` as a disagreement, which is a difference in how two readers
    write a number and not a difference in what the sentence said. Stripping
    currency symbols and separators before comparing is a type coercion — it
    cannot make two different amounts look equal.

    **Words compare as words.** `"annual"` against `"yearly"` stays a
    disagreement, even though a person can see they match, because resolving it
    needs a synonym table and a synonym table is a third reader with no
    evidence and no name. If those two readings appear often, that is a finding
    about the schema's vocabulary and belongs in adjudication, not here.
    """
    a, b = str(left).strip().lower(), str(right).strip().lower()
    if a == b:
        return True
    if mode == "NUMBER":
        left_number, right_number = _as_number(a), _as_number(b)
        return left_number is not None and left_number == right_number
    if mode == "SET":
        return _as_set(a) == _as_set(b) and bool(_as_set(a))
    return False


def _as_number(text: str):
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _as_set(text: str) -> frozenset:
    """Tokens, ignoring which conjunction a reader chose.

    `"VTI and BND"` and `"VTI, BND"` name the same two instruments. Which
    English word joined them is this module's problem, not a reading.
    """
    import re

    tokens = re.split(r"[,;/]|\band\b|\bor\b|\balongside\b|\bplus\b", text)
    return frozenset(t.strip() for t in tokens if t.strip())


def compare_relations(sets: Sequence[ReadingSet], schema: Schema
                      ) -> Sequence[FieldComparison]:
    """One comparison per relation kind, structural rather than textual.

    Agreement requires the kind, the member roles, the canonical subjects, the
    per-member qualifiers and the whole-relation attributes to match. Source
    spans and evidence sit outside it, exactly as for `evaluation_period`: two
    readers can ground the same relation in slightly different text and still
    agree about its meaning.

    Ordering follows the schema. `account_transition` is ordered, because
    from/to reversed is the opposite transaction; `portfolio_sleeves` is not,
    because a core and a satellite are the same portfolio whichever was
    mentioned first.
    """
    ok = [one for one in sets if one.ok]
    out = []
    for spec in schema.relations:
        values, spans, comparable = {}, {}, {}
        for reader in ok:
            found = reader.relation_of(spec.kind)
            if found is not None:
                values[reader.reader_id] = found.to_json()
                spans[reader.reader_id] = found.source_span
                comparable[reader.reader_id] = found.comparable(spec.ordered)

        if not values:
            state = UNREAD
        elif len(values) < 2 or len(values) < len(ok):
            state = ONE_SIDED
        else:
            first = next(iter(comparable.values()))
            state = (AGREED if all(v == first for v in comparable.values())
                     else CONTESTED)
        out.append(FieldComparison(dimension=f"rel:{spec.kind}", state=state,
                                   values=values, spans=spans))
    return tuple(out)


def compare(text: str, sets: Sequence[ReadingSet], schema: Schema) -> Comparison:
    """One comparison per dimension in the schema, for every reader."""
    ok = [s for s in sets if s.ok]
    failed = {s.reader_id: s.failed for s in sets if not s.ok}

    fields = []
    for name in sorted(schema.names):
        values, spans = {}, {}
        for one in ok:
            reading = one.value_of(name)
            if reading is not None:
                values[one.reader_id] = reading.value
                spans[one.reader_id] = reading.source_span

        if not values:
            state = UNREAD
        elif len(values) < 2:
            # One reader saw it. That is ONE_SIDED whether the other reader
            # looked and declined, or was never there at all.
            #
            # The first version tested `len(values) < len(ok)`, which is the
            # same thing only while two readers survive. With one — a provider
            # outage, a missing key — every dimension it read came back AGREED,
            # and a shadow run against a dead endpoint reported perfect
            # agreement. "Silence is not agreement" was in this module's
            # docstring and not in its code.
            state = ONE_SIDED
        elif len(values) < len(ok):
            state = ONE_SIDED
        else:
            declared = schema.dimension(name)
            mode = declared.compare_as if declared else "TEXT"
            first = next(iter(values.values()))
            state = (AGREED
                     if all(_same(first, v, mode) for v in values.values())
                     else CONTESTED)

        fields.append(FieldComparison(dimension=name, state=state,
                                      values=values, spans=spans))

    fields.extend(compare_relations(sets, schema))
    return Comparison(text=text, fields=tuple(fields), failed_readers=failed)


def evidence_for(comparison: Comparison, dimension: str) -> Sequence:
    """Every reader's view of one dimension, as contract `DecisionEvidence`.

    Including the losers. A field that was contested and then settled is a
    different fact from one that was never in doubt, and only the first
    justifies asking again when the readers change.
    """
    from ..contracts import DecisionEvidence, ReaderKind

    for one in comparison.fields:
        if one.dimension != dimension:
            continue
        return tuple(
            DecisionEvidence(
                reader_id=reader,
                # Every reader here is a reader; which *kind* it is belongs to
                # the reader, not to this comparison, and guessing from the id
                # would encode a naming convention as a fact.
                kind=ReaderKind.RULE if "compiler" in reader else ReaderKind.MODEL,
                value=value,
                source_ref=one.spans.get(reader, ""))
            for reader, value in sorted(one.values.items()))
    return ()


# ── the report that says more than a percentage ────────────────────────────

def matrix(comparisons: Sequence[Comparison], schema: Schema,
           readers: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Per dimension, five counts — and the two `ONE_SIDED` halves separated.

    A single overall agreement percentage hides the only question worth asking
    of a shadow run. "One reader read it" is two very different findings:

        model only      a field the legacy reader never represented.
                        Evidence *for* the new one, not a problem.
        compiler only   a field the new reader missed that the old one has.
                        A reader defect, and the thing to go and look at.

    Collapsing them into `ONE_SIDED` was enough to see there was coverage
    movement and not enough to say which direction it went.
    """
    out: Dict[str, Dict[str, int]] = {}
    left, right = readers[0], readers[1]

    # Relations are compared and must also be *reported*. The first version
    # iterated `schema.names`, which is dimensions only, so schema@2's two
    # relation kinds were measured and then silently dropped from the matrix —
    # a run that fixed the largest structural gap and could not show it.
    subjects = sorted(schema.names) + [f"rel:{k}"
                                       for k in sorted(schema.relation_kinds)]
    for name in subjects:
        counts = {"both_agree": 0, "both_contested": 0,
                  f"{left}_only": 0, f"{right}_only": 0, "neither": 0}
        for comparison in comparisons:
            for one in comparison.fields:
                if one.dimension != name:
                    continue
                if one.state == AGREED:
                    counts["both_agree"] += 1
                elif one.state == CONTESTED:
                    counts["both_contested"] += 1
                elif one.state == UNREAD:
                    counts["neither"] += 1
                else:
                    who = next(iter(one.values), "")
                    key = f"{who}_only"
                    if key in counts:
                        counts[key] += 1
        out[name] = counts
    return out


def render(matrix_by_dimension: Dict[str, Dict[str, int]],
           readers: Sequence[str],
           provenance: Optional[Mapping[str, str]] = None) -> str:
    """The matrix, with what adjudication concluded about each dimension.

    The last column is the difference between a report and evidence. A count
    of contested rows says how often two readers differed; it does not say
    whether the difference was a defect in one of them or a gap in the schema
    that made both answer wrongly. Only adjudication says that, and a matrix
    without it invites the reader to guess.
    """
    provenance = provenance or {}
    left, right = readers[0], readers[1]
    head = (f"{'dimension':24s} {'agree':>6s} {'contest':>8s} "
            f"{'cmplr':>7s} {'model':>7s} {'neither':>8s}  adjudicated")
    lines = [head, "-" * len(head)]
    for name, counts in sorted(
            matrix_by_dimension.items(),
            key=lambda kv: (-kv[1]["both_contested"], -kv[1]["both_agree"])):
        verdict = provenance.get(name, "")
        lines.append(
            f"{name:24s} {counts['both_agree']:6d} "
            f"{counts['both_contested']:8d} {counts[f'{left}_only']:7d} "
            f"{counts[f'{right}_only']:7d} {counts['neither']:8d}  {verdict}")
    return "\n".join(lines)
