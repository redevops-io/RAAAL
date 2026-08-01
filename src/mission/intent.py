"""Intent: what someone wants, before it is a plan they could run.

One intent legitimately produces several candidate Missions. "I want to invest
safely for retirement" is a real goal and not a program, and the honest response
is to offer monthly DCA, a glide path and risk parity as candidates rather than
to pick one and present it as the answer.

**But generating candidates is a search, and choosing among them after seeing
their backtests is selecting on outcome.**

That is the same statistical act as trying two hundred moving-average lengths and
keeping the best, with one aggravating difference: here the *platform* generated
the alternatives. If the candidates are evaluated and then chosen by result, every
candidate is a trial, and a deflated Sharpe computed as though there had been one
is overstated by exactly the amount the search inflated it.

This module makes that unavoidable rather than optional. `SelectionBasis` is a
required field, the trial count follows from it, and claiming to have chosen
before seeing results while having evaluated several candidates raises.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .spec import MISSION_SPEC_VERSION


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class SelectionBasis(str, Enum):
    """How the user arrived at the candidate they kept.

    The only field in this system whose value changes a statistic. Everything
    else records what happened; this records *why one thing was preferred*, and
    preference formed by looking at outcomes is a search.
    """

    NOT_SELECTED = "NOT_SELECTED"
    """Still choosing."""

    STATED_PREFERENCE = "STATED_PREFERENCE"
    """Chosen on a property the user named up front — "I don't want bonds" —
    with no outcome consulted. Not a search."""

    BEFORE_RESULTS = "BEFORE_RESULTS"
    """Chosen from the descriptions, before anything was simulated. One trial."""

    AFTER_RESULTS = "AFTER_RESULTS"
    """Chosen having seen the backtests. Every evaluated candidate is a trial,
    and this is the honest default whenever there is any doubt."""

    @property
    def is_outcome_driven(self) -> bool:
        return self is SelectionBasis.AFTER_RESULTS


class CandidateOrigin(str, Enum):
    USER = "USER"
    """The user described this candidate. Evaluating it is not a search."""

    PLATFORM = "PLATFORM"
    """The platform generated it, which puts the burden of proof on the platform
    that the generation was not itself a hidden search."""


#: Attributes a platform-generated candidate set may be built from. All are
#: properties of the *rule*, knowable before anything is run — which is the point.
#: Generating candidates from performance would make the platform the researcher,
#: conducting a selection it then does not report.
GENERATION_CONSTRAINTS = frozenset({
    "permits_bonds",
    "excludes_bonds",
    "passive",
    "tactical",
    "required_liquidity",
    "contribution_compatible",
    "allowed_asset_classes",
    "maximum_turnover",
    "public_methodology_category",
})


class HiddenSelection(ValueError):
    """The platform evaluated candidates it did not show.

    Generating ten, measuring them privately and presenting the best three is the
    platform performing a search and reporting one trial. It is the single most
    damaging thing this layer could do quietly, so it raises rather than warns.
    """


@dataclass(frozen=True)
class Candidate:
    """One way of turning an intent into something runnable."""

    key: str
    summary: str
    """Plain language, sufficient to choose between candidates without running
    them — which is what makes BEFORE_RESULTS a real option rather than a
    formality."""

    mission_ref: Optional[str] = None
    evaluated: bool = False
    rationale: str = ""
    origin: CandidateOrigin = CandidateOrigin.PLATFORM
    shown_to_user: bool = True
    """False means generated and measured but never presented. Every such
    candidate is still a trial, and `Intent` refuses to be constructed with one
    that was evaluated."""

    def to_json(self) -> Dict[str, Any]:
        return {"key": self.key, "summary": self.summary,
                "mission_ref": self.mission_ref, "evaluated": self.evaluated,
                "rationale": self.rationale, "origin": self.origin.value,
                "shown_to_user": self.shown_to_user}


@dataclass(frozen=True)
class Intent:
    """What the user asked for, and how the choice among candidates was made."""

    name: str
    version: int
    stated: str
    """Verbatim. The one field nobody but the user may write."""

    candidates: Sequence[Candidate] = ()
    selected: Optional[str] = None
    selection_basis: SelectionBasis = SelectionBasis.NOT_SELECTED
    generation_constraints: Sequence[str] = ()
    """The non-performance attributes a platform-generated set was built from.
    Required whenever the platform generated candidates, because a set built
    from anything else was built from results."""

    results_visible_before_selection: bool = False
    rejected_candidates: Sequence[str] = ()
    """Generated, considered, and not taken. Retained so the set that was offered
    can be reconstructed rather than inferred from the one that survived."""

    generated_at: Optional[str] = None
    opened_at: Optional[str] = None
    spec_version: str = MISSION_SPEC_VERSION

    def __post_init__(self) -> None:
        # Structural problems first. A malformed reference makes every later
        # check meaningless, and reporting a policy violation on an object that
        # does not hold together sends the reader to the wrong place.
        keys = {c.key for c in self.candidates}
        if self.selected and self.selected not in keys:
            raise ValueError(
                f"{self.artifact_id}: selected {self.selected!r} is not among "
                f"the candidates {sorted(keys)}"
            )
        if self.selected and self.selection_basis is SelectionBasis.NOT_SELECTED:
            raise ValueError(
                f"{self.artifact_id}: a candidate was selected but no basis is "
                "recorded. How the choice was made determines the trial count, "
                "so it cannot be left blank"
            )
        if (self.selection_basis is SelectionBasis.BEFORE_RESULTS
                and self.evaluated_count > 1):
            raise ValueError(
                f"{self.artifact_id}: basis is BEFORE_RESULTS but "
                f"{self.evaluated_count} candidates were evaluated. Seeing several "
                "results and then choosing is selection on outcome, whatever the "
                "order of operations felt like"
            )
        if (self.selection_basis is SelectionBasis.STATED_PREFERENCE
                and self.evaluated_count > 1):
            raise ValueError(
                f"{self.artifact_id}: basis is STATED_PREFERENCE but "
                f"{self.evaluated_count} candidates were evaluated. A preference "
                "stated up front does not need alternatives measured against it"
            )
        if (self.results_visible_before_selection
                and self.selection_basis in {SelectionBasis.BEFORE_RESULTS,
                                             SelectionBasis.STATED_PREFERENCE}):
            raise ValueError(
                f"{self.artifact_id}: results were visible before selection, so "
                f"the basis cannot be {self.selection_basis.value}"
            )

        # Then the policy checks: how the set came to exist.
        hidden = [c.key for c in self.candidates
                  if c.evaluated and not c.shown_to_user]
        if hidden:
            raise HiddenSelection(
                f"{self.artifact_id}: candidates {sorted(hidden)} were evaluated "
                "but never shown. Measuring alternatives privately and presenting "
                "the survivors makes the platform the researcher, reporting one "
                "trial for a search it conducted"
            )

        platform_generated = [c for c in self.candidates
                              if c.origin is CandidateOrigin.PLATFORM]
        if platform_generated and not self.generation_constraints:
            raise ValueError(
                f"{self.artifact_id}: the platform generated "
                f"{len(platform_generated)} candidate(s) with no declared "
                "generation constraints. A set not built from stated rule "
                "attributes was built from results"
            )
        unknown = set(self.generation_constraints) - GENERATION_CONSTRAINTS
        if unknown:
            raise ValueError(
                f"{self.artifact_id}: {sorted(unknown)} are not recognised "
                "generation constraints. Candidates must be generated from "
                "properties of the rule, knowable before anything is run"
            )

    @property
    def concept_id(self) -> str:
        return f"intent/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"intent/{self.name}@{self.version}"

    @property
    def evaluated_count(self) -> int:
        return sum(1 for c in self.candidates if c.evaluated)

    @property
    def trials_incurred(self) -> int:
        """Configurations this intent spent, for deflation purposes.

        Outcome-driven selection spends every candidate that was measured. Any
        other basis spends one, because the alternatives were never weighed by
        result — which is precisely the property that makes them free.
        """
        if self.selection_basis.is_outcome_driven:
            return max(self.evaluated_count, 1)
        return 1 if self.selected else 0

    @property
    def is_a_search(self) -> bool:
        return self.selection_basis.is_outcome_driven and self.evaluated_count > 1

    def disclosure(self) -> str:
        """What the result page has to say about how this plan was chosen.

        Surfacing the trial count to a retail user is the differentiator: a
        platform that lets someone try forty variants and reports the best one as
        though it were the first has built an overfitting machine with a
        progress bar.
        """
        if not self.selected:
            return "No plan selected yet."
        if not self.is_a_search:
            return (
                "This plan was chosen from its description rather than from its "
                "results, so its statistics carry no selection penalty."
            )
        return (
            f"This plan was chosen after comparing {self.evaluated_count} "
            f"candidates' results. All {self.evaluated_count} count as attempts, "
            "and the deflated statistics on this page already account for that — "
            "the best of several will always look better than it is."
        )

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "stated": self.stated,
            "candidates": sorted(
                ({"key": c.key, "mission_ref": c.mission_ref,
                  "evaluated": c.evaluated} for c in self.candidates),
                key=lambda d: d["key"],
            ),
            "selected": self.selected,
            "selection_basis": self.selection_basis.value,
            "generation_constraints": sorted(self.generation_constraints),
            "results_visible_before_selection": self.results_visible_before_selection,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "evaluated_count": self.evaluated_count,
            "trials_incurred": self.trials_incurred,
            "is_a_search": self.is_a_search,
            "disclosure": self.disclosure(),
            "dsr_countable_trials": self.trials_incurred,
            "rejected_candidates": list(self.rejected_candidates),
            "generated_at": self.generated_at,
            "opened_at": self.opened_at,
        }
