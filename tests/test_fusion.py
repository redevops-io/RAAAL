"""Phase 5: syntax contributes evidence and never has authority.

Every test here traces to a measured fact rather than to a preference. The
central one:

    In "I contribute monthly and rebalance at year end" the parser attaches
    `year end` to *contribute*, with a clean governor chain and a confident
    score.

So "syntax wins when confidence is high" would adopt that error, and no
threshold separates a confident-and-right parse from a confident-and-wrong one.
That is why nothing in `fusion.py` reads a score's magnitude — only its sign,
and only as evidence.
"""
from __future__ import annotations

import pytest
from runtime_contracts import OpenReason

from src.discovery.fusion import (
    AMBIGUOUS_TERMS,
    Decision,
    Fusion,
    FusionReport,
    Proposal,
    Requirement,
    fuse,
)
from src.discovery.syntax import SyntaxEvidence


def model(dimension: str, value, span: str = "") -> Proposal:
    return Proposal(dimension=dimension, value=value,
                    reader_id="claude-sonnet-5@1", source_span=span)


def syntax(dimension: str, value, score: int, span: str = "") -> SyntaxEvidence:
    return SyntaxEvidence(dimension=dimension, proposed_value=value,
                          score=score, features=(f"{score:+d}:test",),
                          source_span=span, parser="stanza@1.14.0")


class TestOnlyAgreeProceeds:
    def test_model_and_syntax_agreeing_proceeds(self):
        decision = fuse("cadence", model=model("cadence", "monthly"),
                        syntax=[syntax("cadence", "monthly", 3)])
        assert decision.outcome is Fusion.AGREE and decision.proceeds
        assert decision.value == "monthly"

    def test_the_model_alone_proceeds(self):
        """"Syntax neutral" is not "syntax absent from the design". A dimension
        the parser has nothing to say about must still be readable, or adding
        this layer would have narrowed what Discovery can understand."""
        decision = fuse("cadence", model=model("cadence", "monthly"))
        assert decision.outcome is Fusion.AGREE

    @pytest.mark.parametrize("outcome", [
        Fusion.DISAGREE, Fusion.INSUFFICIENT_RELATION,
        Fusion.AMBIGUOUS_BY_LANGUAGE])
    def test_nothing_else_proceeds(self, outcome):
        assert not outcome.proceeds

    def test_an_open_decision_carries_no_value(self):
        """A value beside a refusal is a value a caller renders, and then a
        figure exists for a question that was never settled."""
        decision = fuse("cadence", model=model("cadence", "monthly"),
                        syntax=[syntax("cadence", "monthly", -4)])
        assert decision.outcome is Fusion.DISAGREE
        assert decision.value is None


class TestSyntaxNeverWins:
    def test_syntax_alone_never_carries_a_field(self):
        """However strong. This is the rule the whole module exists for."""
        decision = fuse("cadence", syntax=[syntax("cadence", "monthly", 99)])
        assert decision.outcome is Fusion.DISAGREE
        assert "authority" in decision.detail

    def test_a_contradiction_is_not_resolved_by_score(self):
        """The `year end` case in miniature: syntax is confident and wrong, so
        a bigger number must not become a stronger claim."""
        weak = fuse("cadence", model=model("cadence", "monthly"),
                    syntax=[syntax("cadence", "annually", 1)])
        strong = fuse("cadence", model=model("cadence", "monthly"),
                      syntax=[syntax("cadence", "annually", 99)])
        assert weak.outcome is strong.outcome is Fusion.DISAGREE

    def test_supporting_evidence_does_not_change_the_value_either(self):
        """Symmetry. Syntax cannot promote a reading any more than it can
        overrule one — the value that proceeds is always the model's."""
        decision = fuse("cadence", model=model("cadence", "monthly"),
                        syntax=[syntax("cadence", "monthly", 99)])
        assert decision.value == "monthly"

    def test_a_negative_score_on_the_models_own_value_contradicts(self):
        """The shape the real defect has: syntax saying "not attached here"
        rather than proposing somewhere else."""
        decision = fuse("cadence", model=model("cadence", "monthly"),
                        syntax=[syntax("cadence", "monthly", -4)])
        assert decision.outcome is Fusion.DISAGREE


class TestInsufficientRelation:
    def test_a_binding_dimension_without_its_binding_does_not_proceed(self):
        """`50/50` in a sentence naming three accounts is not an allocation
        until something says which account it belongs to."""
        decision = fuse("account_allocation",
                        model=model("account_allocation", "50/50"), bound=False)
        assert decision.outcome is Fusion.INSUFFICIENT_RELATION
        assert "account" in decision.detail

    def test_the_same_dimension_proceeds_once_bound(self):
        """The discriminating opposite: without it, "this dimension never
        proceeds" would satisfy the test above."""
        decision = fuse("account_allocation",
                        model=model("account_allocation", "50/50"), bound=True)
        assert decision.outcome is Fusion.AGREE

    def test_a_non_binding_dimension_is_unaffected(self):
        assert fuse("cadence", model=model("cadence", "monthly"),
                    bound=False).outcome is Fusion.AGREE


class TestAmbiguousByLanguage:
    def test_an_attested_ambiguous_term_asks_rather_than_scores(self):
        """Neither a parser failure nor a model failure. People writing about
        their own portfolios use `rebalance` for both "back to target" and
        "change the target", so no reading is recoverable from the sentence."""
        decision = fuse("periodic_rebalancing",
                        model=model("periodic_rebalancing", "annual",
                                    "rebalance to 70/30"),
                        available=["periodic_rebalancing", "stated_weights"])
        assert decision.outcome is Fusion.AMBIGUOUS_BY_LANGUAGE

    def test_it_outranks_agreement(self):
        """Deliberately checked first. Two readers agreeing on a word that
        carries two meanings is two readers making the same assumption, which
        looks like confirmation and is not."""
        decision = fuse("periodic_rebalancing",
                        model=model("periodic_rebalancing", "annual",
                                    "rebalance to 70/30"),
                        syntax=[syntax("periodic_rebalancing", "annual", 3,
                                       "rebalance to 70/30")],
                        available=["periodic_rebalancing", "stated_weights"])
        assert decision.outcome is Fusion.AMBIGUOUS_BY_LANGUAGE

    def test_an_unambiguous_sentence_is_not_caught(self):
        decision = fuse("cadence",
                        model=model("cadence", "monthly", "contribute monthly"))
        assert decision.outcome is Fusion.AGREE

    def test_the_word_alone_is_not_enough(self):
        """The narrowing. `rebalanced annually` carries the term and only one
        of its readings — the competing one needs a target the sentence does
        not contain. Without this, the outcome fires on its own vocabulary."""
        decision = fuse("periodic_rebalancing",
                        model=model("periodic_rebalancing", "annual",
                                    "rebalanced annually"),
                        available=["periodic_rebalancing", "cadence"])
        assert decision.outcome is Fusion.AGREE

    def test_a_different_dimension_is_not_caught_by_it(self):
        """The ambiguity is between two named fields. A third dimension in the
        same sentence is not made ambiguous by their argument."""
        decision = fuse("cadence",
                        model=model("cadence", "monthly", "rebalance to 70/30"),
                        available=["cadence", "stated_weights"])
        assert decision.outcome is Fusion.AGREE

    def test_every_ambiguous_term_cites_where_it_was_observed(self):
        """A list anyone may add a hunch to becomes a list of things nobody
        wants to implement. The point of this outcome is that the ambiguity was
        seen in how people write, not predicted from how the code is shaped."""
        for term, record in AMBIGUOUS_TERMS.items():
            assert record["source"].startswith("http"), term
            assert "|" in record["readings"], (
                f"{term} must name both readings it carries")


class TestWhatItMeansForSealing:
    def report(self) -> FusionReport:
        return FusionReport(decisions=(
            fuse("cadence", model=model("cadence", "monthly")),
            fuse("amount", model=model("amount", "500"),
                 syntax=[syntax("amount", "200", 4)]),
            fuse("day_rule", model=model("day_rule", "first"),
                 syntax=[syntax("day_rule", "last", 3)]),
        ))

    def test_material_open_decisions_are_what_block(self):
        report = self.report()
        blocking = [d.dimension for d in report.blocks_sealing]
        assert blocking == ["amount"], (
            "day_rule is open too, but it is declared immaterial — a "
            "non-result-changing dimension left open must not block sealing")

    def test_open_decisions_map_onto_the_contract(self):
        unresolved = self.report().unresolved_for_contract()
        by_dimension = {u.dimension: u for u in unresolved}
        assert by_dimension["amount"].reason is OpenReason.UNRESOLVED_DISAGREEMENT
        assert by_dimension["amount"].result_changing is True
        assert by_dimension["day_rule"].result_changing is False

    def test_a_language_ambiguity_is_not_asked_rather_than_disagreed(self):
        """`UNRESOLVED_DISAGREEMENT` means readers disagreed. Nobody disagreed
        here — nobody has put the question to the user yet, and recording it as
        a disagreement would misdescribe both the cause and the repair."""
        report = FusionReport(decisions=(
            fuse("periodic_rebalancing",
                 model=model("periodic_rebalancing", "annual",
                             "rebalance to 70/30"),
                 available=["periodic_rebalancing", "stated_weights"]),))
        (unresolved,) = report.unresolved_for_contract()
        assert unresolved.reason is OpenReason.NOT_ASKED

    def test_an_unknown_dimension_is_material_by_default(self):
        """Conservative on purpose: treating an undeclared dimension as
        immaterial would let anything new proceed unexamined."""
        decision = fuse("something_new", model=model("something_new", "x"),
                        syntax=[syntax("something_new", "y", 3)])
        assert decision.material is True

    def test_the_policy_version_is_stamped(self):
        """A decision made under one policy and read under another is a
        decision nobody can explain."""
        decision = fuse("cadence", model=model("cadence", "monthly"))
        assert decision.policy_version.startswith("quantify-fusion@")
        assert "policy_version" in decision.to_json()
