"""Who does this value modify — the layer fusion could refuse without.

Every case runs against a parse Stanza actually produced, replayed from the
recording. The sentence that motivated the module:

    401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)

Normalisation gives three ratios. Fusion can only refuse them. This is what
turns them into an answer, or says honestly that it cannot.

The discipline the tests hold to: **the binder emits structure and never
meaning.** A binding says a ratio is the appositive of `401k`. Nothing here
asserts that `401k` is an account or that the ratio is its allocation — those
are the schema's business, and a test asserting them would be the semantic
decision this module is built to not make.
"""
from __future__ import annotations

import pytest

from src.discovery.binding import (
    BindingStatus,
    RULES,
    bind,
    is_bound,
    phrase_of,
    value_id,
)
from discovery_runtime.fusion import Fusion
from src.discovery.claims import Proposal
from src.discovery.syntax import normalize
from src.discovery.syntax_stanza import RecordedReader

RECORDED = RecordedReader()

ACCOUNTS = "401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)"
WEIGHTS = "60% to VTI and 40% to BND"
SLEEVES = "put 70% in equities, 30% in bonds"


def bindings_for(text: str, relation: str = ""):
    parse = RECORDED.parse(text)
    found = bind(parse, normalize(text))
    return [b for b in found if not relation or b.relation == relation]


class TestTheSentenceTheModuleWasBuiltFor:
    def test_each_ratio_binds_to_its_own_account(self):
        """The payoff. Three ratios, three accounts, and structure says which
        goes with which — which normalisation alone could not, and fusion was
        right to refuse without."""
        bound = bindings_for(ACCOUNTS, "appositive_of")
        pairs = [(b.value_id, b.target_span) for b in bound
                 if b.status is BindingStatus.BOUND]
        assert pairs == [("ratio@6-11", "401k"),
                         ("ratio@24-29", "Roth IRA"),
                         ("ratio@51-56", "taxable brokerage")]

    def test_it_works_through_three_different_tokenisations(self):
        """Stanza splits these three ratios three different ways in one
        sentence. The binder sees normalised values aligned to spans, so the
        tokenizer's inconsistency never reaches it."""
        parse = RECORDED.parse(ACCOUNTS)
        numeric = [t.text for s in parse.sentences for t in s.tokens
                   if any(ch.isdigit() for ch in t.text)]
        assert len(numeric) > 3, f"tokenised uniformly now: {numeric}"
        assert all(b.status is BindingStatus.BOUND
                   for b in bindings_for(ACCOUNTS, "appositive_of"))

    def test_the_target_is_a_name_a_person_can_read(self):
        """`k` is structurally correct and useless. A target nobody can
        identify is a binding nobody can check."""
        for binding in bindings_for(ACCOUNTS, "appositive_of"):
            assert binding.target_span not in ("k", "IRA", "brokerage")

    def test_every_binding_carries_the_edges_that_justified_it(self):
        """Without evidence a binding is an assertion, and a year later nobody
        can tell a good one from a lucky one."""
        for binding in bindings_for(ACCOUNTS, "appositive_of"):
            assert binding.evidence
            assert all("->" in edge for edge in binding.evidence)


class TestWeightsBindToAssets:
    def test_each_weight_takes_its_own_asset(self):
        pairs = [(b.value_id, b.target_span)
                 for b in bindings_for(WEIGHTS, "shares_head_with")
                 if b.status is BindingStatus.BOUND]
        assert [span for _, span in pairs] == ["VTI", "BND"]

    def test_the_coordination_boundary_is_not_crossed(self):
        """The first version of this rule looked *upward* for a shared head,
        and `40%` reached across the `conj` edge to match `VTI` as well as
        `BND`. Looking downward at what hangs off the value's own phrase
        cannot cross that boundary."""
        for binding in bindings_for(WEIGHTS, "shares_head_with"):
            assert binding.status is BindingStatus.BOUND
            assert len(binding.candidates) == 1, binding.candidates

    def test_it_holds_for_a_different_preposition(self):
        pairs = [b.target_span for b in bindings_for(SLEEVES, "shares_head_with")
                 if b.status is BindingStatus.BOUND]
        assert pairs == ["equities", "bonds"]

    def test_a_name_part_is_never_the_target(self):
        """`Roth` in `Roth IRA` and `taxable` in `taxable brokerage` belong to
        the head's own name. Binding a ratio to one of them would be binding it
        to half of its own target, which is what `compound` in the participant
        set produced."""
        for binding in bindings_for(ACCOUNTS, "shares_head_with"):
            assert binding.target_span not in ("Roth", "taxable")


class TestValuesBindToTheirVerb:
    def test_an_amount_takes_its_governing_verb(self):
        text = "invest $500 monthly and rebalance annually"
        bound = {b.value_id: b for b in bindings_for(text, "governed_by")}
        money = next(b for k, b in bound.items() if k.startswith("money"))
        assert money.status is BindingStatus.BOUND
        assert money.target_span == "invest"

    def test_a_duration_takes_its_governing_verb(self):
        (binding,) = bindings_for("hold the bonus for 90 days", "governed_by")
        assert binding.target_span == "hold"


class TestWhatItRefusesToDecide:
    def test_a_value_with_no_structural_target_is_unbound(self):
        bound = bindings_for("60/40", "appositive_of")
        assert all(b.status is BindingStatus.UNBOUND for b in bound)

    def test_an_unbound_binding_names_no_target(self):
        """A target beside `UNBOUND` is a target a caller renders."""
        for binding in bindings_for("60/40", "appositive_of"):
            assert binding.target_span == ""
            assert binding.candidates == ()

    def test_the_rules_declare_what_they_are_evidence_for(self):
        """The label is on the rule, not in the output, so a consumer reading a
        binding cannot mistake a structural fact for a settled field."""
        for rule in RULES:
            assert "↔" in rule.supports
            assert rule.strategy and rule.relation

    def test_nothing_in_a_binding_names_a_semantic_field(self):
        """The line this module must not cross. `appositive_of` is structure;
        `account_allocation` would be a decision."""
        for binding in bindings_for(ACCOUNTS):
            assert binding.relation in {"appositive_of", "shares_head_with",
                                        "governed_by"}


class TestIdentityIsStable:
    def test_the_same_text_produces_the_same_ids(self):
        """A binding referring to a fresh id each run could not be stored
        beside an intent and read back."""
        first = [b.value_id for b in bindings_for(ACCOUNTS)]
        second = [b.value_id for b in bindings_for(ACCOUNTS)]
        assert first == second and first

    def test_the_id_is_derived_from_the_value_not_the_binding(self):
        values = normalize(ACCOUNTS)
        ids = {value_id(v) for v in values}
        assert {b.value_id for b in bindings_for(ACCOUNTS)} <= ids


class TestFusionConsumesRealBindings:
    """`INSUFFICIENT_RELATION` firing on live input rather than a parameter."""

    def test_a_bound_ratio_proceeds(self):
        from src.discovery.adapter import fuse_with_bindings

        parse = RECORDED.parse(ACCOUNTS)
        values = normalize(ACCOUNTS)
        found = bind(parse, values)
        decision = fuse_with_bindings(
            "account_allocation", values[0], bindings=found,
            model=Proposal("account_allocation", "50/50", "claude-sonnet-5@1"))
        assert decision.outcome is Fusion.AGREE

    def test_an_unbound_ratio_does_not(self):
        """The same dimension, the same model reading, a sentence whose
        structure establishes nothing — and the outcome changes. Without this
        pair, "it always agrees" and "it always refuses" would both pass."""
        from src.discovery.adapter import fuse_with_bindings

        text = "70/30 vs 60/40"
        parse, values = RECORDED.parse(text), normalize(text)
        found = bind(parse, values)
        decision = fuse_with_bindings(
            "account_allocation", values[0], bindings=found,
            model=Proposal("account_allocation", "70/30", "claude-sonnet-5@1"))
        assert decision.outcome is Fusion.INSUFFICIENT_RELATION

    def test_fusion_still_cannot_see_a_parse(self):
        """The seam, now that fusion is upstream.

        `fuse` takes a boolean; the binder produces it. The property used to be
        that Quantify's fusion module imported `is_bound` and no structural
        type. Fusion has since moved to discovery-runtime, which makes the
        first half *stronger* — the runtime cannot import Quantify at all — and
        moves the second half to the adapter, which is now the only place a
        parse and a fusion decision meet.

        Checked as a *dependency*, not as a substring. Two earlier versions of
        this test scanned the source text for the word "parse" — first inside
        `fuse`'s own explanation of why it does not parse, then inside a
        refusal message written for a user. A text search cannot tell an access
        from prose about the access, and an import graph says it exactly.
        """
        import ast
        from pathlib import Path

        import discovery_runtime.fusion as runtime_fusion

        from src.discovery import adapter

        def imports_of(module):
            tree = ast.parse(Path(module.__file__).read_text())
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names.update(f"{node.module}.{alias.name}"
                                 for alias in node.names)
                elif isinstance(node, ast.Import):
                    names.update(alias.name for alias in node.names)
            return names

        structural = {"Parse", "Sentence", "Token", "Aligned", "align"}

        runtime_imports = imports_of(runtime_fusion)
        offenders = [name for name in runtime_imports
                     if name.rsplit(".", 1)[-1] in structural
                     or "syntax_stanza" in name
                     or name.split(".")[0] in {"src", "quantify"}]
        assert not offenders, (
            f"discovery-runtime's fusion imports {offenders}; deciding whether "
            "a reading proceeds and reading structure are different jobs, and "
            "the seam between them is the only thing stopping fusion becoming "
            "a second parser")

        # The adapter is where the boolean is computed. It may see the binder's
        # predicate — that is its job — and still not hand structure onward.
        adapter_imports = imports_of(adapter)
        assert any("is_bound" in name for name in adapter_imports), (
            "nothing consumes the binder's predicate, so `requires_binding` is "
            "answered by a caller's guess rather than by a parse")
        leaked = [name for name in adapter_imports
                  if name.rsplit(".", 1)[-1] in structural]
        assert not leaked, (
            f"the adapter imports {leaked} and passes them toward fusion")

    def test_is_bound_is_the_only_predicate(self):
        parse, values = RECORDED.parse(ACCOUNTS), normalize(ACCOUNTS)
        found = bind(parse, values)
        assert is_bound(found, values[0])
        assert not is_bound(found, values[0], relation="governed_by")


def test_phrase_of_rejoins_by_offset_not_by_space():
    """`401` and `k` are adjacent in the source; `Roth` and `IRA` are not. Only
    the offsets know which, and the space-joined version needed a special case
    for `k` — the shape of a rule that breaks on the next abbreviation."""
    parse = RECORDED.parse(ACCOUNTS)
    sentence = parse.sentences[0]
    heads = {t.index: t for t in sentence.tokens}
    rebuilt = {phrase_of(sentence, t) for t in heads.values()}
    assert "401k" in rebuilt
    assert "Roth IRA" in rebuilt
