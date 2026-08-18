"""Several observations of one dimension become one claim before fusion sees it.

    raw reader observations
        -> dimension-owned normalise/aggregate
        -> one canonical claim per reader per dimension
        -> generic fusion

The step exists because the alternative is worse. The deterministic reader
emits one observation per share — `0.6` and `0.4` for a 60/40 split — and
without aggregation each reached fusion as a separate claim about
`stated_weights` and was compared individually against the model's `60/40`.
That reported a contradiction between two readers that agreed, and it stopped a
serving-path migration.

The fix is deliberately *not* to teach fusion that several values may sometimes
equal one. That would make fusion a domain interpreter. Whether several
observations are one value in pieces or several competing answers is a fact
about the dimension, so the domain answers it and fusion goes on comparing one
claim to one claim.
"""
from __future__ import annotations

from decimal import Decimal

from src.discovery import adapter


class _Observation:
    def __init__(self, dimension, value):
        self.dimension = dimension
        self.value = value
        self.source_span = ""


def _values(observations):
    return sorted(str(o.value) for o in observations)


def test_shares_emitted_separately_become_one_claim():
    """The case that stopped item 3."""
    out = adapter.one_claim_per_dimension([
        _Observation("stated_weights", "0.6"),
        _Observation("stated_weights", "0.4"),
    ])
    assert len(out) == 1, f"expected one claim, got {_values(out)}"
    assert adapter.same_value_for("stated_weights", out[0].value, "60/40"), (
        f"the aggregated claim {out[0].value!r} is not the same split as "
        "60/40, so fusion would still report a disagreement")


def test_set_members_aggregate_the_same_way():
    """One seam, not two.

    SET members were unioned by their own function before this existed. Both
    are the same question — is this one value in pieces? — and two answers to
    it is how they would drift.
    """
    out = adapter.one_claim_per_dimension([
        _Observation("assets", "bonds"),
        _Observation("assets", "stocks"),
    ])
    assert len(out) == 1
    assert set(out[0].value.split(", ")) == {"bonds", "stocks"}


def test_a_scalar_dimension_is_never_aggregated():
    """Two readings of a scalar genuinely compete.

    Joining them would invent a value nobody stated, which is the failure mode
    on the other side of this seam and the more dangerous one.
    """
    out = adapter.one_claim_per_dimension([
        _Observation("cadence", "monthly"),
        _Observation("cadence", "annual"),
    ])
    assert len(out) == 2, "a scalar dimension was aggregated"
    assert _values(out) == ["annual", "monthly"]


def test_the_scales_a_split_arrives_in_all_reconcile():
    """Fractions, percentages, and shares bound to holdings.

    `60/40`, `0.6/0.4` and `VTI=60,BND=40` are one split written three ways —
    by the model, by the deterministic reader, and by the derived reader that
    knows which holding takes which share.
    """
    same = adapter.same_value_for
    assert same("stated_weights", "60/40", "0.6/0.4")
    assert same("stated_weights", "60/40", "VTI=60,BND=40")
    assert same("stated_weights", "0.6/0.4", "VTI=60,BND=40")


def test_a_different_split_still_differs():
    """The half that makes the rest mean anything."""
    same = adapter.same_value_for
    assert not same("stated_weights", "60/40", "70/30")
    assert not same("stated_weights", "60/40", "0.7/0.3")
    assert not same("stated_weights", "60/40", "VTI=70,BND=30")


def test_an_unreadable_share_does_not_become_equal_to_anything():
    assert adapter.weights("bananas") is None
    assert not adapter.same_value_for("stated_weights", "bananas", "oranges")


def test_aggregation_replaces_the_right_attribute():
    """A deterministic candidate carries `proposed_value`, a reading `value`.

    Writing the wrong one leaves the original in place — an aggregation that
    reports success and changes nothing, which is exactly the shape of bug this
    file exists to catch.
    """
    class _Candidate:
        def __init__(self, value):
            self.proposed_value = value
            self.source_span = ""

    out = adapter.one_claim_per_dimension(
        [_Candidate(Decimal("0.6")), _Candidate(Decimal("0.4"))],
        value_of=lambda c: c.proposed_value,
        dimension_of=lambda c: "stated_weights")
    assert len(out) == 1
    assert adapter.same_value_for("stated_weights",
                                  out[0].proposed_value, "60/40")
