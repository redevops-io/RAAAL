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
from src.discovery.syntax import ATTACHMENTS, score_attachment
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

    def test_the_non_english_cases_are_visibly_absent(self):
        """Recorded here as a known gap rather than discovered later as a
        surprise: only the English model has been fetched, so the Spanish,
        German, French and Russian cases have no parse and are not running."""
        absent = sorted({c.language for c in DEPENDENCY
                         if not RECORDED.has(c.text, c.language)})
        assert absent == ["de", "es", "fr", "ru"], (
            f"the set of unparsed languages changed to {absent}; update this "
            "and say which models were added")


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
