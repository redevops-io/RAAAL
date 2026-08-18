"""Quantify's own Discovery vocabulary: what its words mean, and what a
dimension needs.

The machinery that consumed these tables moved to `discovery-runtime` and is
tested there. The tables did not: which terms people demonstrably use for two
things, and what a dimension requires before a value means anything, are facts
about finance. They are supplied to the runtime through the `ambiguity` and
`material` seams.

`tests/test_fusion.py` used to hold the source-citation rule alongside tests of
the fusion implementation. The implementation is gone; the rule is not, so it
lives here rather than disappearing with the file that happened to contain it.
"""
from __future__ import annotations

import pytest

from src.discovery.vocabulary import AMBIGUOUS_TERMS, REQUIREMENTS, Requirement


@pytest.mark.parametrize("term", sorted(AMBIGUOUS_TERMS))
def test_every_ambiguous_term_cites_where_it_was_observed(term):
    """A list anyone may add a hunch to becomes a list of things nobody wants
    to implement.

    The point of the AMBIGUOUS_BY_LANGUAGE outcome is that the ambiguity was
    *seen* in how people write, not predicted from how the code is shaped — so
    an entry without a source is a prediction wearing the costume of an
    observation.
    """
    record = AMBIGUOUS_TERMS[term]
    assert record["source"].startswith("http"), (
        f"{term} cites no source; the outcome it triggers claims the ambiguity "
        "was observed")
    assert "|" in record["readings"], (
        f"{term} must name both readings it carries — an ambiguity with one "
        "reading is not one")


@pytest.mark.parametrize("term", sorted(AMBIGUOUS_TERMS))
def test_every_ambiguous_term_names_the_dimensions_it_is_between(term):
    """`between` is what stops the outcome firing on its own vocabulary.

    "rebalanced annually" carries the word and no ambiguity: the competing
    reading needs a target and the sentence has none. Without `between` the
    check matches every reading `periodic_rebalancing` ever produced.
    """
    between = AMBIGUOUS_TERMS[term].get("between", ())
    assert len(between) >= 2, (
        f"{term} names {between}; an ambiguity is between at least two "
        "dimensions or it is not one")


def test_requirements_declare_materiality_and_comparison():
    """Both are read by the adapter and handed to the runtime.

    A dimension absent here is material and unbound — the conservative reading,
    since treating an unknown dimension as immaterial lets anything new proceed
    unexamined.
    """
    assert REQUIREMENTS, "no requirements are declared"
    for name, requirement in REQUIREMENTS.items():
        assert isinstance(requirement.material, bool), name
        assert requirement.compare_as, f"{name} declares no comparison mode"


def test_an_undeclared_dimension_is_material_by_default():
    assert Requirement().material is True
    assert Requirement().compare_as == "TEXT"


def test_stated_weights_compares_as_weights():
    """The divergence that refused a split the compiler had been handed.

    The schema says SET for this dimension and the requirement says WEIGHTS.
    Fusion reads the requirement, so `60/40` and `VTI=60,BND=40` are one split
    — and reading the schema instead compared them as unordered tokens and
    called them different.
    """
    assert REQUIREMENTS["stated_weights"].compare_as == "WEIGHTS"
