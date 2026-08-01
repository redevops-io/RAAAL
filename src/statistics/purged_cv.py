"""Purged, embargoed walk-forward splits.

Financial labels span time. A label formed at *t* using a horizon of *h* is still
resolving at *t+h*, so a naive split that puts *t* in train and *t+1* in test
leaks: the training observation already contains information the test observation
is being asked to predict.

Two controls, both from López de Prado's *Advances in Financial Machine Learning*:

* **Purge** — drop training observations whose label horizon overlaps the test
  window.
* **Embargo** — additionally drop training observations immediately *after* the
  test window, because serial correlation means near-adjacent observations carry
  the test period's information even without formal overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedSplit:
    """One train/test division, with the purged and embargoed regions recorded."""

    train_index: pd.Index
    test_index: pd.Index
    purged: int
    embargoed: int

    def to_json(self) -> dict:
        return {
            "train_size": len(self.train_index),
            "test_size": len(self.test_index),
            "purged": self.purged,
            "embargoed": self.embargoed,
            "train_start": str(self.train_index.min()) if len(self.train_index) else None,
            "train_end": str(self.train_index.max()) if len(self.train_index) else None,
            "test_start": str(self.test_index.min()) if len(self.test_index) else None,
            "test_end": str(self.test_index.max()) if len(self.test_index) else None,
        }


def purged_walk_forward_splits(
    index: pd.Index,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
    expanding: bool = True,
) -> List[PurgedSplit]:
    """Generate chronological splits with purging and embargo.

    Training always precedes testing — this is a time series, so a shuffled or
    symmetric split would place future observations in the training set.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")

    n = len(index)
    fold = n // (n_splits + 1)
    if fold < 1:
        raise ValueError(f"{n} observations cannot form {n_splits} splits")

    splits: List[PurgedSplit] = []
    for k in range(1, n_splits + 1):
        test_start = k * fold
        test_end = min((k + 1) * fold, n)
        if test_start >= test_end:
            continue

        test_index = index[test_start:test_end]

        train_end = test_start - purge
        train_start = 0 if expanding else max(0, train_end - fold * k)
        train_positions = list(range(max(train_start, 0), max(train_end, 0)))

        # Embargo the window immediately after the test block, for the case where
        # training resumes past the test period (non-expanding schemes).
        embargo_end = min(test_end + embargo, n)
        embargoed_positions = set(range(test_end, embargo_end))
        train_positions = [p for p in train_positions if p not in embargoed_positions]

        splits.append(
            PurgedSplit(
                train_index=index[train_positions],
                test_index=test_index,
                purged=purge,
                embargoed=len(embargoed_positions),
            )
        )
    return splits


def leakage_score(train_index: pd.Index, test_index: pd.Index) -> int:
    """Count observations present in both sides. Must always be zero."""
    return len(set(train_index) & set(test_index))
