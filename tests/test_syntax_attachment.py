"""Tier 2: which token governs which, against parses Stanza actually produced.

No model is loaded here. `RecordedReader` replays real output, so the evidence
is a real parser's and the run is a second — and there is no hand-written
stand-in parser anywhere, because a stand-in would be the legacy regex compiler
with a linguistics vocabulary.

The case this whole layer exists for:

    "invest $500 monthly and rebalance annually"

A reader collecting cadences finds two and picks one. A parse says `monthly`
hangs off `invest` and `annually` off `rebalance`, and the ambiguity is gone
without anything here knowing what a contribution is.
"""
from __future__ import annotations

import pytest

from corpus.parser.loader import load
from src.discovery.syntax import (
    ATTACHMENTS, align, normalize, score_attachment, score_value,
)
from src.discovery.syntax_stanza import RecordedReader

RECORDED = RecordedReader()

DEPENDENCY = [c for c in load()
              if c.tier == "dependency"
              and ("head_lemma" in c.asserts or "shares_head_with" in c.asserts)]

HAVE_PARSE = [c for c in DEPENDENCY if RECORDED.has(c.text, c.language)]


def find(parse, surface: str):
    """The token whose text is `surface`, or the first of a multi-word span."""
    wanted = surface.strip().lstrip("$").lower()
    for sentence in parse.sentences:
        for token in sentence.tokens:
            if token.text.lower().strip("$") == wanted:
                return sentence, token
    return None, None


class TestTheRecordingCoversTheCorpus:
    def test_every_english_dependency_case_has_a_parse(self):
        """A case with no recording does not fail — it vanishes. Asserting the
        coverage is what stops the tier below from shrinking quietly."""
        missing = [c.id for c in DEPENDENCY
                   if c.language == "en" and not RECORDED.has(c.text, "en")]
        assert not missing, (
            f"{len(missing)} English cases have no recorded parse: "
            f"{missing[:6]}. Run `python corpus/parser/record_parses.py en`")

    def test_the_active_corpus_is_english_only(self):
        """The multilingual dependency fixtures are deferred, not deleted —
        they live in `deferred_multilingual.json` with their reason. Fetching
        two gigabytes of models so a counter reached zero would have corrupted
        what the report means, and deleting them would have lost the evidence
        of what was intended."""
        import json
        from pathlib import Path

        absent = sorted({c.language for c in DEPENDENCY
                         if not RECORDED.has(c.text, c.language)})
        assert absent == [], f"unparsed languages in the active corpus: {absent}"

        deferred = json.loads(
            (Path(__file__).resolve().parent.parent / "corpus" / "parser"
             / "deferred_multilingual.json").read_text())
        assert deferred["status"] == "NOT PART OF CURRENT COVERAGE"
        assert {c["language"] for c in deferred["cases"]} == {"es", "de",
                                                              "fr", "ru"}


@pytest.mark.parametrize("case", HAVE_PARSE, ids=lambda c: c.id)
def test_the_dependent_attaches_to_the_stated_head(case):
    """One edge per case, walked up the governor chain.

    Not only the immediate head: "invest five hundred a month" puts a noun
    between the adverbial and the verb in most parses, and a check that looked
    one step up would fail for a structural reason rather than a semantic one.
    """
    parse = RECORDED.parse(case.text, case.language)
    sentence, token = find(parse, case.asserts["dependent"])
    assert token is not None, (
        f"{case.asserts['dependent']!r} is not a token of {case.text!r}")

    if "shares_head_with" in case.asserts:
        # Which weight belongs to which sleeve is a shared head, not a chain:
        # `60` is `nummod` of a `%` and `VTI` is `nmod` of that same `%`. A
        # head *lemma* would not tell the two `%` tokens of one sentence apart,
        # which is precisely the sentence that matters.
        _, partner = find(parse, case.asserts["shares_head_with"])
        assert partner is not None, (
            f"{case.asserts['shares_head_with']!r} is not a token")
        assert token.head == partner.head, (
            f"{token.text!r} hangs off token {token.head} and "
            f"{partner.text!r} off {partner.head} — they do not share a head, "
            f"so nothing pairs them. {case.note}")
        return

    expected = case.asserts["head_lemma"]
    chain = [g.lemma for g in sentence.governor_chain(token)]
    assert expected in chain, (
        f"{case.asserts['dependent']!r} governed by {chain} — expected "
        f"{expected!r}. {case.note}")

    if "relation" in case.asserts:
        assert token.relation.split(":")[0] == case.asserts["relation"], (
            f"{token.text!r} attaches with {token.relation!r}, expected "
            f"{case.asserts['relation']!r}")


class TestNormalisationHappensBeforeScoring:
    """The ordering constraint the tokenisation defect forced.

    A scorer reasoning over tokens is reasoning over whatever the tokenizer
    happened to do. `align()` hands it normalised values matched by character
    span instead, so the unit it reasons about is the same in every sentence.
    """

    ACCOUNTS = "401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)"

    def test_the_tokenizer_is_genuinely_inconsistent_here(self):
        """Establish the premise before relying on it. If Stanza ever
        tokenises these three uniformly, the alignment layer stops earning its
        place here and this test says so."""
        parse = RECORDED.parse(self.ACCOUNTS)
        numeric = [t.text for s in parse.sentences for t in s.tokens
                   if any(ch.isdigit() for ch in t.text)]
        assert "50/50" in numeric, "one ratio survives as a single token"
        assert "85" in numeric and "15" in numeric, "another is split apart"
        assert len({"50/50" in numeric, "85/15" in numeric}) == 2, (
            f"the three ratios are tokenised uniformly now: {numeric}. The "
            "alignment layer was added for this inconsistency — if it is gone, "
            "say so rather than leaving a test that no longer describes it")

    def test_alignment_presents_three_uniform_values(self):
        """Three shapes in, three identical shapes out."""
        parse = RECORDED.parse(self.ACCOUNTS)
        aligned = align(parse, normalize(self.ACCOUNTS))
        assert [a.value.canonical for a in aligned] == [(50, 50), (85, 15), (70, 30)]
        assert all(a.anchor is not None for a in aligned)

    def test_a_value_the_parse_does_not_cover_is_dropped_not_guessed(self):
        """A span no token overlaps means the parse and the normaliser disagree
        about the text. Picking a nearby token would invent an attachment
        neither of them made."""
        from src.discovery.syntax import Value

        parse = RECORDED.parse("invest $500 monthly")
        beyond = Value(kind="duration", canonical=90, source_span="90 days",
                       start_char=900, end_char=907, unit="days")
        assert align(parse, [beyond]) == ()

        # And the control: values whose spans *are* covered all come back, so
        # the rule is "drop what does not overlap" rather than "drop
        # everything". Both the amount and the cadence are covered here.
        real = normalize("invest $500 monthly")
        assert len(align(parse, real)) == len(real) == 2

    def test_scoring_a_value_and_scoring_its_anchor_agree(self):
        """`score_value` is a wrapper, not a second implementation. If the two
        could differ, the layer would have two scoring rules and one of them
        would rot."""
        text = "invest $500 monthly and rebalance annually"
        parse = RECORDED.parse(text)
        for aligned in align(parse, normalize(text)):
            direct = score_attachment(aligned.sentence, aligned.anchor,
                                      ATTACHMENTS["cadence"])
            assert score_value(aligned, ATTACHMENTS["cadence"]) == direct


class TestScoringSeparatesTheTwoCadences:
    """The sentence the plan was written around."""

    TEXT = "invest $500 monthly and rebalance annually"

    def parse(self):
        assert RECORDED.has(self.TEXT, "en"), "recorded fixture missing"
        return RECORDED.parse(self.TEXT, "en")

    def score(self, surface: str):
        parse = self.parse()
        sentence, token = find(parse, surface)
        assert token is not None
        return score_attachment(sentence, token, ATTACHMENTS["cadence"])

    def test_the_contribution_cadence_scores_positive(self):
        score, features = self.score("monthly")
        assert score > 0, f"scored {score} with {features}"

    def test_the_rebalancing_cadence_scores_negative(self):
        """Negative, not merely unsupported. Both at zero is a reader that has
        learned nothing from a sentence containing two cadences."""
        score, features = self.score("annually")
        assert score < 0, f"scored {score} with {features}"

    def test_the_two_are_separated_by_the_score(self):
        assert self.score("monthly")[0] > self.score("annually")[0]

    def test_the_arithmetic_is_shown(self):
        """Interpretable rather than probabilistic. A bare number invites a
        threshold, and a threshold is a decision this layer may not make."""
        _, features = self.score("monthly")
        assert features and all(":" in f and "@" in f for f in features)

    def test_an_unknown_verb_scores_zero_rather_than_negative(self):
        """`Attachment.against` is not the complement of `supports`. Treating
        unknown as negative makes the layer confidently wrong on exactly the
        unanticipated phrasings it was added to handle."""
        text = "gift $500 monthly"
        if not RECORDED.has(text, "en"):
            pytest.skip("fixture not recorded")
        parse = RECORDED.parse(text, "en")
        sentence, token = find(parse, "monthly")
        score, _ = score_attachment(sentence, token, ATTACHMENTS["cadence"])
        assert score == 0
