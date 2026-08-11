"""Linking one intent to the one before it.

Separate from the store so the chain rule is one function rather than a detail
of an INSERT, and separate from `intent.py` so the planner does not import
storage.

The chain exists because trial accounting is derived from history. A total
computed over a chain somebody edited is not a smaller total — it is a total
with no meaning, and the difference has to be visible.
"""
from __future__ import annotations

import hashlib
import json


def semantic_form(intent) -> dict:
    """The fields a trial total actually depends on.

    Deliberately not the whole record. `created_at` and the raw instruction do
    not change what the chain asserts, and including them would make a
    retention policy that drops the sentence look like tampering.
    """
    return {
        "intent_id": intent.intent_id,
        "source_revision": intent.source_revision,
        "edit_effect": intent.edit_effect.value,
        "selection_basis": intent.selection_basis.value,
        "repetition_signature": intent.repetition_signature.key(),
        "alternatives_generated": intent.alternatives_generated,
        "results_visible": bool(intent.results_visible),
        "trial_effect": intent.trial_effect,
    }


def chain_link(previous_hash: str, intent) -> str:
    payload = json.dumps({"previous": previous_hash,
                          "intent": semantic_form(intent)},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
