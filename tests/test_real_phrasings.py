"""What the layer does with attested language, recorded as it is today.

These are characterization tests, not correctness tests. The pack carries no
expected plans on purpose — labelling real phrasings with the answers I think
they should give would re-introduce the self-authorship problem one layer up.

So this file asserts *current behaviour*, and marks which of those behaviours
are defects. Two things follow, and both are the point:

    a green run means nothing changed, not that anything is right
    a red run means the layer moved, and the diff says on which sentence

When a parser upgrade or a scoring change fixes one of the marked defects, this
file fails and the fix is visible rather than silent. That is the only property
worth having here — a "known wrong" that nothing watches becomes a "known" that
nobody remembers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.discovery.syntax import ATTACHMENTS, normalize, score_attachment
from src.discovery.syntax_stanza import RecordedReader

PACK = json.loads(
    (Path(__file__).resolve().parent.parent
     / "corpus" / "parser" / "real_phrasings.json").read_text())
ENTRIES = PACK["entries"]
RECORDED = RecordedReader()


class TestThePackIsHonestAboutWhereItCameFrom:
    def test_every_entry_declares_its_provenance(self):
        """The pack's whole value is that it is not invented. An entry that
        does not say where it came from is indistinguishable from one that
        was."""
        for entry in ENTRIES:
            assert entry["provenance"] in (
                "user_reported", "search_summary", "variant")

    def test_attested_entries_carry_a_source(self):
        for entry in ENTRIES:
            if entry["provenance"] == "user_reported":
                assert entry["source"].startswith("http"), entry["id"]

    def test_the_collection_method_is_recorded(self):
        """Bogleheads 402s this fetcher and reddit and Stack Exchange are
        blocked to it. A pack that did not say so would read as scraped."""
        assert "402" in PACK["collection_note"]

    def test_the_invented_share_is_visible(self):
        """Just under half is mine. Stated rather than discovered later."""
        variants = sum(1 for e in ENTRIES if e["provenance"] == "variant")
        assert variants / len(ENTRIES) < 0.5, (
            f"{variants}/{len(ENTRIES)} entries are invented; the pack stops "
            "being evidence about real language when most of it is not")


class TestNormalisationOnRealPhrasings:
    def test_a_bare_ratio_is_read_without_a_head_noun(self):
        """Every constructed case wrote "a 60/40 portfolio". Real writing
        drops the noun, and the value must survive that."""
        for text in ("60/40", "I'm 60/40", "move to 60/40", "maintain 60/40"):
            found = [v for v in normalize(text) if v.kind == "ratio"]
            assert found and found[0].canonical == (60, 40), text

    def test_three_account_ratios_are_all_read(self):
        found = [v.canonical for v in
                 normalize("401k (50/50), Roth IRA (85/15), "
                           "taxable brokerage (70/30)")
                 if v.kind == "ratio"]
        assert found == [(50, 50), (85, 15), (70, 30)]

    def test_but_nothing_binds_a_ratio_to_its_account(self):
        """DEFECT, and a structural one. Tier 1 emits three ratios and no
        accounts, so a consumer reading only normalisation has the numbers and
        cannot say which is the 401k. The binding is a relation, and relations
        are tier 2 — this test exists so the gap is recorded rather than
        assumed to be covered."""
        values = normalize("401k (50/50), Roth IRA (85/15), "
                           "taxable brokerage (70/30)")
        assert all(not v.unit for v in values if v.kind == "ratio")

    def test_a_rebalancing_band_is_not_read_at_all(self):
        """DEFECT. `5/25` is the standard Bogleheads band — rebalance when a
        holding is 5 percentage points or 25% relative from target — and it is
        shaped exactly like an allocation. The sums-to-100 rule drops it
        silently, which is the right answer for `12/25` and the wrong one
        here, and no rule over digits alone can tell them apart."""
        assert not [v for v in normalize("I use a 5/25 band")
                    if v.kind == "ratio"]

    def test_a_worded_period_produces_no_value(self):
        """Recorded, not a defect. "at year end" is a day rule rather than a
        literal, and inventing a duration for it here would be tier 1 making a
        semantic decision."""
        assert not normalize("rebalance at year end")

    def test_two_ratios_arrive_with_nothing_marking_which_is_the_target(self):
        """DEFECT. In "my 60/40 is acting like 70/30" the first is the target;
        in "70/30 vs 60/40" neither is. Order does not settle it, so a
        consumer taking the first ratio is right by accident."""
        assert [v.canonical for v in normalize("my 60/40 is acting like 70/30")
                if v.kind == "ratio"] == [(60, 40), (70, 30)]
        assert [v.canonical for v in normalize("70/30 vs 60/40")
                if v.kind == "ratio"] == [(70, 30), (60, 40)]


def find(parse, surface: str):
    for sentence in parse.sentences:
        for token in sentence.tokens:
            if token.text.lower() == surface.lower():
                return sentence, token
    return None, None


@pytest.mark.skipif(not RECORDED.has("I contribute monthly and rebalance at year end"),
                    reason="real phrasings not recorded; run record_parses.py")
class TestTheParserOnRealPhrasings:
    def test_the_constructed_sentence_attaches_correctly(self):
        """The control, and the reason the next test matters. On the invented
        sentence the parser gets it right, which is exactly why a corpus of
        invented sentences said the layer worked."""
        parse = RECORDED.parse("invest $500 monthly and rebalance annually")
        sentence, token = find(parse, "annually")
        assert "rebalance" in [g.lemma for g in sentence.governor_chain(token)]

    def test_the_real_sentence_attaches_the_timing_to_the_wrong_verb(self):
        """**The finding of this pass.**

        In "I contribute monthly and rebalance at year end" Stanza makes `end`
        an `obl` of *contribute*, not of *rebalance*, and makes `rebalance` a
        conjunct of `monthly` rather than of `contribute`. So the parser says
        the year-end timing belongs to the contribution — confidently, and
        wrongly, on a sentence a person would actually write.

        This is the strongest available argument for the plan's own rule that
        syntax must never win by itself. Here syntax is not neutral or absent;
        it is wrong, and a fusion policy that let a high syntax score overrule
        the semantic reader would adopt the error.

        Asserted as the current behaviour. If a parser upgrade fixes it, this
        test fails and somebody reads the diff.
        """
        parse = RECORDED.parse("I contribute monthly and rebalance at year end")
        sentence, token = find(parse, "end")
        governors = [g.lemma for g in sentence.governor_chain(token)]
        assert governors[0] == "contribute", (
            f"the parser now attaches 'year end' to {governors[0]!r}. If that "
            "is 'rebalance', the defect this test records has been fixed — "
            "update the test and the fusion notes")

    def test_the_longer_form_of_the_same_sentence_attaches_correctly(self):
        """And this is why it cannot be fixed with a rule. The attested
        sentence — "maintain 60/40 by contributions and rebalance at year end"
        — attaches `end` to `rebalance`, correctly. Same phrase, same two
        verbs, opposite result, decided by how much else is in the clause."""
        parse = RECORDED.parse(
            "maintain 60/40 by contributions and rebalance at year end")
        sentence, token = find(parse, "end")
        assert "rebalance" in [g.lemma for g in sentence.governor_chain(token)]

    def test_the_cadence_scorer_inherits_the_error(self):
        """The consequence, made concrete. The scorer walks the governor chain,
        so a wrong chain produces a confident wrong score rather than an
        abstention — which is what a fusion rule would be handed."""
        parse = RECORDED.parse("I contribute monthly and rebalance at year end")
        sentence, token = find(parse, "monthly")
        score, features = score_attachment(sentence, token,
                                           ATTACHMENTS["cadence"])
        assert score > 0 and features

    def test_the_account_ratios_tokenise_three_different_ways(self):
        """DEFECT, and one no scoring rule reaches. In a single sentence
        Stanza keeps `50/50` as one token, splits `85/15` into three, and
        splits `70/30` into `70/` and `30`. An extractor written against any
        one of those shapes is wrong about the other two."""
        parse = RECORDED.parse(
            "401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)")
        surfaces = [t.text for s in parse.sentences for t in s.tokens
                    if any(ch.isdigit() for ch in t.text)]
        assert "50/50" in surfaces
        assert "85" in surfaces and "15" in surfaces
