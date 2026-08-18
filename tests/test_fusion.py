"""Fusion outcomes, comparison, and what they mean for sealing.

**This file tested an implementation that no longer exists.**

`src/discovery/fusion.py` was deleted when Quantify's serving path moved to
`redevops-io/discovery-runtime`. Comparison, aggregation, the fusion outcome
and the two-witness loop are that package's now, and its own suite tests them —
including the cases this file covered.

What did *not* move is Quantify's vocabulary, which is tested in
`tests/test_vocabulary.py`, and the adapter that supplies it, tested in
`tests/test_adapter_completeness.py`, `tests/test_claim_aggregation.py` and
`tests/test_two_witness_equivalence.py`.

Skipped rather than emptied so the record of what was once asserted here stays
readable in the history rather than only in a deletion diff.
"""
import pytest

pytest.skip(
    "tests src/discovery/fusion.py, deleted with the migration to "
    "discovery-runtime; the behaviour is tested upstream and the vocabulary "
    "in tests/test_vocabulary.py",
    allow_module_level=True)
