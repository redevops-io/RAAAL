"""WorksheetIntent: what was asked, before anything is changed.

    request -> intent -> planner -> proposed change -> confirmation -> revision

The planner classifies. Not the model, and not the template — because two of the
things it decides are trial accounting and comparability impact, and a system
that lets the requester decide how many trials their request counted for has no
trial accounting at all.

**Two axes, deliberately orthogonal.** What the edit touches is a different
question from why it was chosen:

    edit_effect      LAYOUT_ONLY | DERIVED_ANALYSIS | SCENARIO_CHANGE
    selection_basis  STATED_PREFERENCE | BEFORE_RESULTS | ANALYTICAL_ONLY
                     | VARIANT_EXPLORATION | AFTER_RESULTS

"Move the scope panel" and "add a rolling volatility chart" are both
`ANALYTICAL_ONLY`, and only one of them creates a derived artifact. "Replace SPY
with VTI" and "try SPY, VTI and VT and keep the best" are both
`SCENARIO_CHANGE`, and only one of them counts three trials.

**Selection basis is history-aware.** It cannot be read from a single
instruction: "add a 63-day rolling volatility" is analytical the first time and
parameter tuning the fourth. The repetition signature identifies the analytical
*decision* rather than the wording, so three differently-phrased requests against
the same metric and parameter belong to one family.

Nothing here mutates. A planner that applied its own proposal would be deciding
on the user's behalf exactly where the user's judgement is the point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class EditEffect(str, Enum):
    LAYOUT_ONLY = "LAYOUT_ONLY"
    """Presentation. Creates no trial and requires no rerun."""

    DERIVED_ANALYSIS = "DERIVED_ANALYSIS"
    """A new statistic or view over an existing run. No new simulation."""

    SCENARIO_CHANGE = "SCENARIO_CHANGE"
    """Changes an input to the simulation. Requires a run, and counts."""

    UNCLASSIFIED = "UNCLASSIFIED"
    """The planner did not recognise this instruction.

    A distinct state, not a fallback to `LAYOUT_ONLY`. Reading "we did not
    understand this" as "presentation, zero trials" is a semantic claim, and the
    most permissive one available. It never applied — `propose` refuses it — but
    it did reach telemetry, where a parser failure became indistinguishable from
    a genuine layout edit. Those are different product problems, and only one of
    them is fixed by improving the recognisers."""


class SelectionBasis(str, Enum):
    STATED_PREFERENCE = "STATED_PREFERENCE"
    BEFORE_RESULTS = "BEFORE_RESULTS"
    ANALYTICAL_ONLY = "ANALYTICAL_ONLY"
    """A diagnostic, not a choice between candidates."""

    VARIANT_EXPLORATION = "VARIANT_EXPLORATION"
    """Several values evaluated, none chosen yet. Every one still counts:
    a search that has not finished is still a search."""

    AFTER_RESULTS = "AFTER_RESULTS"
    """Chosen having seen the outcomes. The reading that inflates a result."""

    UNKNOWN = "UNKNOWN"
    """Why this was chosen cannot be read, because what it asks for cannot be
    read. Paired with `EditEffect.UNCLASSIFIED`, never on its own."""


@dataclass(frozen=True)
class RepetitionSignature:
    """The analytical decision, not the sentence.

    Keyed so that "63-day rolling volatility", "show me 90 days" and "try 126"
    land in one family — otherwise repeated tuning hides behind rephrasing.
    """

    target_run: str = ""
    block_type: str = ""
    metric: str = ""
    parameter_family: str = ""
    scenario_dimension: str = ""

    def key(self) -> str:
        return "|".join((self.target_run, self.block_type, self.metric,
                         self.parameter_family, self.scenario_dimension))

    def to_json(self) -> Dict[str, str]:
        return {"target_run": self.target_run, "block_type": self.block_type,
                "metric": self.metric, "parameter_family": self.parameter_family,
                "scenario_dimension": self.scenario_dimension,
                "key": self.key()}


@dataclass(frozen=True)
class WorksheetIntent:
    """What was asked, and what the planner made of it."""

    intent_id: str
    source_revision: int
    instruction: str

    edit_effect: EditEffect
    selection_basis: SelectionBasis
    repetition_signature: RepetitionSignature

    target_blocks: Sequence[str] = ()
    requested_parameters: Sequence[str] = ()
    alternatives_generated: int = 0
    results_visible: bool = False
    related_prior_intents: Sequence[str] = ()

    rerun_required: bool = False

    trial_effect: Optional[int] = 0
    """`None` means unknown, which is not the same as zero.

    An unrecognised instruction may have been asking for a single chart or for a
    sweep of forty parameters. Recording zero would answer a question nobody
    could answer, and the totals built on it would look complete."""

    comparability_impact: str = ""
    presentation_only: bool = False

    requires_user_confirmation: bool = False
    """The planner cannot proceed without the user restating what they meant."""

    @property
    def classified(self) -> bool:
        return self.edit_effect is not EditEffect.UNCLASSIFIED

    def to_json(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "source_revision": self.source_revision,
            "instruction": self.instruction,
            "edit_effect": self.edit_effect.value,
            "selection_basis": self.selection_basis.value,
            "repetition_signature": self.repetition_signature.to_json(),
            "target_blocks": list(self.target_blocks),
            "requested_parameters": list(self.requested_parameters),
            "alternatives_generated": self.alternatives_generated,
            "results_visible": self.results_visible,
            "related_prior_intents": list(self.related_prior_intents),
            "rerun_required": self.rerun_required,
            "trial_effect": self.trial_effect,
            "comparability_impact": self.comparability_impact,
            "presentation_only": self.presentation_only,
            "requires_user_confirmation": self.requires_user_confirmation,
            "classified": self.classified,
        }


# --- recognisers -----------------------------------------------------------
#
# Deterministic and narrow, like stage 1. A model may propose an instruction;
# it does not get to say how many trials the instruction counted for.

_LAYOUT = re.compile(
    r"\b(move|reorder|hide|show|collapse|expand|rename|position|put)\b"
    r"[^.]*\b(panel|section|block|below|above|first|last|order)\b", re.I)

_METRICS = (
    ("volatility", r"\bvolatilit(?:y|ies)\b|\bstd(?:ev)?\b"),
    ("drawdown", r"\bdrawdowns?\b"),
    ("sharpe", r"\bsharpe\b"),
    ("returns", r"\brolling returns?\b|\breturns?\b"),
    ("turnover", r"\bturnovers?\b"),
    ("correlation", r"\bcorrelations?\b"),
)
_ROLLING = re.compile(r"\brolling\b|\b\d+[- ]day\b|\bwindow\b", re.I)
_CHART = re.compile(r"\bchart\b|\bplot\b|\bgraph\b|\bshow\b|\badd\b", re.I)

#: Words that name an instrument or a scenario input. A change to one of these
#: is a change to what gets simulated, not to how it is displayed.
#: A verb followed by a ticker. Case-sensitive on the symbol *deliberately*:
#: under IGNORECASE `[A-Z]{2,5}` matches any short word, so "try 21, 63 and 126
#: day windows" read "day" as an instrument and became a scenario change.
_SCENARIO_INSTRUMENT = re.compile(
    r"(?i:\b(?:replace|swap|change|use|switch|try|test|compare|instead of)\b)"
    r"[^.]*?\b([A-Z]{2,5})\b")

#: Scenario inputs named in words rather than symbols. Case-insensitive, since
#: none of these is a ticker.
_SCENARIO_INPUT = re.compile(
    r"\b(contribut\w+|cadence|dividends?|account|roth|taxable|weighting|"
    r"rebalanc\w+|funding source)\b", re.I)

#: One substitution, not two candidates. "Replace SPY with VTI" names what is
#: leaving and what is arriving; counting both as trials would charge a user for
#: the holding they are removing.
_SUBSTITUTION = re.compile(
    r"\b(replace|swap|switch|change)\b[^.]*\b(with|for|to)\b", re.I)

_NUMBERS = re.compile(r"\b(\d{1,4})[- ]?(?:day|d)\b|\b(\d{1,4})\b")

#: Choosing, having seen the outcomes. The reading that inflates a result, and
#: the one a requester has the least incentive to declare.
_AFTER_RESULTS = re.compile(
    r"\b(keep|pick|choose|use)\b[^.]*\b(best|smoothest|cleanest|highest|lowest|"
    r"strongest|nicest|better|winner)\b"
    r"|\bwhichever\b[^.]*\b(looks|performs|is)\b"
    r"|\bshow (?:me )?the best\b", re.I)

_BEFORE_RESULTS = re.compile(
    r"\bbefore (?:i |we )?(?:see|look|run)\b|\bwithout looking\b", re.I)

_STATED = re.compile(
    r"\bmy (?:stated )?rule\b|\bbecause (?:that is|that's|it is|it's) "
    r"(?:my|what i)\b|\bi already\b|\bas i (?:said|described)\b", re.I)


def _metric_of(instruction: str) -> str:
    for name, pattern in _METRICS:
        if re.search(pattern, instruction, re.I):
            return name
    return ""


def _values_in(instruction: str) -> List[str]:
    """Distinct parameter values a request names.

    "21, 63 and 126 days" is three; "63-day" is one. The count is what turns an
    analytical request into a search.
    """
    found = [m.group(1) or m.group(2) for m in _NUMBERS.finditer(instruction)]
    return list(dict.fromkeys(v for v in found if v))


def _tickers_in(instruction: str) -> List[str]:
    reserved = {"AND", "THE", "USE", "ADD", "TRY"}
    return list(dict.fromkeys(
        t for t in re.findall(r"\b([A-Z]{2,5})\b", instruction)
        if t not in reserved))


#: A follow-up that names no subject: "try 21, 63 and 126", "keep 63". These
#: only mean anything against what came before, which is why the planner needs
#: history and a single instruction cannot be classified alone.
_FOLLOW_UP = re.compile(
    r"^\s*(try|keep|use|pick|choose|show|and)\b[^.]*?"
    r"(\b\d{1,4}\b|\bwindows?\b|\bbest\b|\bsmoothest\b|\bcleanest\b)", re.I)


def classify_effect(instruction: str,
                    previous: Optional["WorksheetIntent"] = None) -> EditEffect:
    """What the edit touches. Independent of why it was chosen.

    A bare follow-up inherits the effect of the request it continues. "Try 21,
    63 and 126 day windows" names no metric and no instrument; read alone it
    looks like nothing, and read after "add 63-day rolling volatility" it is
    obviously more of the same.
    """
    if _LAYOUT.search(instruction):
        return EditEffect.LAYOUT_ONLY
    if _metric_of(instruction) and (_ROLLING.search(instruction)
                                    or _CHART.search(instruction)):
        return EditEffect.DERIVED_ANALYSIS
    if _SCENARIO_INSTRUMENT.search(instruction) or _SCENARIO_INPUT.search(instruction):
        return EditEffect.SCENARIO_CHANGE
    if _metric_of(instruction):
        return EditEffect.DERIVED_ANALYSIS
    if previous is not None and _FOLLOW_UP.search(instruction):
        return previous.edit_effect
    # Nothing matched. Deliberately not LAYOUT_ONLY: see EditEffect.UNCLASSIFIED.
    return EditEffect.UNCLASSIFIED


def signature_for(instruction: str, *, effect: EditEffect,
                  target_run: str = "",
                  previous: Optional["WorksheetIntent"] = None
                  ) -> RepetitionSignature:
    metric = _metric_of(instruction)
    if not metric and previous is not None and _FOLLOW_UP.search(instruction):
        # Inherit the family, so three differently-worded requests against one
        # metric are one repetition family rather than three unrelated edits.
        return previous.repetition_signature
    if effect is EditEffect.SCENARIO_CHANGE:
        return RepetitionSignature(
            target_run=target_run, block_type="scenario", metric="",
            parameter_family="holdings" if _tickers_in(instruction) else "inputs",
            scenario_dimension="allocation_rule")
    if effect is EditEffect.DERIVED_ANALYSIS:
        return RepetitionSignature(
            target_run=target_run,
            block_type="rolling_metric" if _ROLLING.search(instruction) else "metric",
            metric=metric, parameter_family="window" if _ROLLING.search(instruction)
            else "", scenario_dimension="")
    return RepetitionSignature(target_run=target_run, block_type="layout")


def classify_basis(instruction: str, *, effect: EditEffect,
                   alternatives: int, prior_values: int,
                   results_visible: bool) -> SelectionBasis:
    """Why it was chosen. History-aware, because one instruction cannot say.

    A first analytical request is `ANALYTICAL_ONLY`. Repeated parameter
    variation against the same family is `VARIANT_EXPLORATION` — every value
    evaluated counts, even before one is kept. Choosing on the basis of what was
    seen is `AFTER_RESULTS`, and that is the reading a requester has the least
    incentive to declare.
    """
    if _AFTER_RESULTS.search(instruction):
        return SelectionBasis.AFTER_RESULTS
    if _STATED.search(instruction):
        return SelectionBasis.STATED_PREFERENCE
    if _BEFORE_RESULTS.search(instruction):
        return SelectionBasis.BEFORE_RESULTS

    if alternatives + prior_values > 1:
        # Several values against one decision, and none of them named a winner —
        # the `_AFTER_RESULTS` check above is what detects that. A search that
        # has not finished is still a search, so every value counts, but calling
        # it AFTER_RESULTS would assert a choice nobody has made.
        return SelectionBasis.VARIANT_EXPLORATION

    if effect is EditEffect.SCENARIO_CHANGE:
        return SelectionBasis.STATED_PREFERENCE
    return SelectionBasis.ANALYTICAL_ONLY


def plan(instruction: str, *, intent_id: str, source_revision: int,
         history: Sequence[WorksheetIntent] = (), target_run: str = "",
         results_visible: bool = True) -> WorksheetIntent:
    """Classify one request. Changes nothing.

    `results_visible` defaults to True because by the time a worksheet exists
    its figures have been rendered — assuming otherwise would understate every
    later edit.
    """
    previous = history[-1] if history else None
    effect = classify_effect(instruction, previous)

    if effect is EditEffect.UNCLASSIFIED:
        # Returned before anything is derived. Running the basis and trial
        # arithmetic over an instruction nobody could read would produce
        # confident numbers from no evidence — and they would be the numbers
        # most likely to be believed, because nothing about them looks uncertain.
        # The basis recognisers still run. Failing to read *what* an instruction
        # edits is no reason to discard evidence about *why* it was chosen:
        # "keep 63 because it looks smoothest" names no metric and no
        # instrument, and it is still plainly a choice made having seen the
        # outcomes. Unknown is the honest answer only where nothing is legible,
        # and a protective signal must never be lost to an unreadable target.
        declared = SelectionBasis.UNKNOWN
        if _AFTER_RESULTS.search(instruction):
            declared = SelectionBasis.AFTER_RESULTS
        elif _STATED.search(instruction):
            declared = SelectionBasis.STATED_PREFERENCE
        elif _BEFORE_RESULTS.search(instruction):
            declared = SelectionBasis.BEFORE_RESULTS

        return WorksheetIntent(
            intent_id=intent_id, source_revision=source_revision,
            instruction=instruction, edit_effect=EditEffect.UNCLASSIFIED,
            selection_basis=declared,
            repetition_signature=RepetitionSignature(target_run=target_run),
            results_visible=results_visible,
            trial_effect=None, requires_user_confirmation=True,
            comparability_impact=(
                "unknown: this instruction was not recognised, so its effect on "
                "the rule identity cannot be stated"))

    signature = signature_for(instruction, effect=effect, target_run=target_run,
                              previous=previous)

    related = [i for i in history
               if i.repetition_signature.key() == signature.key()]
    prior_values = sum(i.alternatives_generated for i in related)

    values = (_tickers_in(instruction) if effect is EditEffect.SCENARIO_CHANGE
              else _values_in(instruction))
    # A substitution is one change however many symbols it names. Counting the
    # instrument being removed as a trial would charge a user for a decision
    # they are undoing.
    alternatives = 1 if _SUBSTITUTION.search(instruction) else len(values)

    basis = classify_basis(instruction, effect=effect, alternatives=alternatives,
                           prior_values=prior_values,
                           results_visible=results_visible)

    # Layout never counts. A derived analysis counts its variants once a search
    # is under way. A scenario change counts every candidate simulated.
    if effect is EditEffect.LAYOUT_ONLY:
        trials = 0
    elif effect is EditEffect.SCENARIO_CHANGE:
        trials = max(alternatives, 1)
    else:
        trials = alternatives if basis in {SelectionBasis.VARIANT_EXPLORATION,
                                           SelectionBasis.AFTER_RESULTS} else 0

    impact = ""
    if effect is EditEffect.SCENARIO_CHANGE:
        impact = ("the rule identity changes, so benchmark comparability must "
                  "be re-established for this worksheet")
    elif basis in {SelectionBasis.VARIANT_EXPLORATION, SelectionBasis.AFTER_RESULTS}:
        impact = ("no comparability change; the added trials affect deflation "
                  "rather than whether the figures may be read together")

    return WorksheetIntent(
        intent_id=intent_id, source_revision=source_revision,
        instruction=instruction, edit_effect=effect, selection_basis=basis,
        repetition_signature=signature,
        target_blocks=_blocks_for(effect),
        requested_parameters=tuple(values),
        alternatives_generated=alternatives,
        results_visible=results_visible,
        related_prior_intents=tuple(i.intent_id for i in related),
        rerun_required=effect is EditEffect.SCENARIO_CHANGE,
        trial_effect=trials, comparability_impact=impact,
        presentation_only=effect is EditEffect.LAYOUT_ONLY,
    )


def _blocks_for(effect: EditEffect) -> Sequence[str]:
    if effect is EditEffect.SCENARIO_CHANGE:
        return ("StrategyDefinitionBlock", "PerformanceSummaryBlock",
                "BenchmarkComparisonBlock", "TrialAccountingBlock")
    if effect is EditEffect.DERIVED_ANALYSIS:
        return ("PerformanceSummaryBlock", "TrialAccountingBlock")
    return ()
