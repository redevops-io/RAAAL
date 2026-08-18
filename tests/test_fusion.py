"""Fusion outcomes, comparison, and what they mean for sealing.

**This file tested an implementation that no longer exists.**

`src/discovery/fusion.py` was deleted when Quantify's serving path moved to
`redevops-io/discovery-runtime`. What this file covered went to two places,
and saying "it is tested upstream" would be half true:

    the generic half   comparison modes, the outcome enum, contradiction
                       outranking agreement, an unreadable value equalling
                       nothing, merge_readings — `discovery-runtime`'s own
                       tests/test_fusion.py, seventeen cases

    the domain half    what `£2.5k` is worth, that `12m` is months in a window
                       and millions in an amount, that `an SPX ETF` and `SPX
                       ETF` are one holding, that `60/40` and `VTI=60,BND=40`
                       are one split — never upstream's to know, and covered
                       by tests/test_magnitude_suffixes.py,
                       tests/test_discovery_adapter.py and
                       tests/test_vocabulary.py

The vocabulary itself is in `tests/test_vocabulary.py` and the adapter that
supplies it in `tests/test_adapter_completeness.py`,
`tests/test_claim_aggregation.py` and `tests/test_two_witness_equivalence.py`.

Skipped rather than emptied so the record of what was once asserted here stays
readable in the history rather than only in a deletion diff.
"""
import pytest

pytest.skip(
    "tests src/discovery/fusion.py, deleted with the migration to "
    "discovery-runtime; the behaviour is tested upstream and the vocabulary "
    "in tests/test_vocabulary.py",
    allow_module_level=True)
