"""The RSU confirmation card: what will happen, before anything does.

Four questions, in the order a reader asks them:

    What did you describe?
    What did Quantify infer?
    What is still unresolved?
    What will and will not be modelled?

**No computed figures.** The card is built from declarations and versioned
runtime statements only. A projected balance or an expected concentration here
would be an answer to a question the user has not yet agreed to ask, and it
would arrive without the caveats a real result carries.

**Every field has a typed destination.** Each line is an engine input, a runtime
declaration, an unresolved question, or something explicitly not modelled. A
line with no destination is copy — text a user reads and agrees to that reaches
nothing — which is recognition without representation at the confirmation layer.

The "will and will not model" section reads the same runtime declarations that
build the result context after the run, so the card and the worksheet cannot
describe one run differently except by version drift, which is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..mission.rsu_declaration import Destination, RSUDeclaration


@dataclass(frozen=True)
class CardField:
    """One line, and where it goes."""

    label: str
    value: str
    destination: Destination
    why: str = ""
    defaults_ref: str = ""
    field_name: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"label": self.label, "value": self.value,
                "destination": self.destination.value, "why": self.why,
                "defaults_ref": self.defaults_ref, "field": self.field_name}


LABELS: Mapping[str, str] = {
    "grant_identity": "Grant", "employer_ticker": "Employer stock",
    "vest_schedule": "Vest dates", "gross_shares": "Shares per vest",
    "gross_value": "Value per vest", "withholding_method": "Withholding",
    "withholding_rate": "Withholding rate",
    "corporate_action_ref": "Corporate-action history",
    "disposition_policy": "What happens at vest",
    "blackout_schedule": "Blackout windows",
    "allocation_policy": "Where proceeds go",
    "concentration_cap": "Employer-stock cap",
    "account_destination": "Account", "tax_runtime_ref": "Tax rules",
    "account_runtime_ref": "Account rules", "market_data_ref": "Price data",
}

#: What an unresolved field needs, phrased as the question to ask. Present here
#: means "ask"; it never means "default".
QUESTIONS: Mapping[str, str] = {
    "withholding_method": ("Does your employer keep shares, or sell some to "
                           "cover the tax? They deliver different share counts."),
    "corporate_action_ref": ("Has this stock split, changed symbol or merged "
                             "since the grant? Share counts cannot be trusted "
                             "across one without knowing."),
    "blackout_schedule": "When are you barred from trading?",
    "account_destination": "Which account do the shares land in?",
    "disposition_policy": "Do you hold the vested shares, or sell some?",
    "allocation_policy": "Where should sale proceeds be invested?",
    "concentration_cap": ("You mentioned reducing concentration. What maximum "
                          "share of the portfolio should the employer stock be?"),
}


@dataclass(frozen=True)
class RSUConfirmationCard:
    """Everything the screen shows, decided here and computed nowhere."""

    described: Sequence[CardField] = ()
    inferred: Sequence[CardField] = ()
    unresolved: Sequence[CardField] = ()
    will_model: Sequence[str] = ()
    will_not_model: Sequence[str] = ()
    version_pin: str = ""
    versions: Mapping[str, str] = field(default_factory=dict)
    scope: Optional[Mapping[str, Any]] = None
    """The `ScopeDisclosure` this plan would run under.

    The card and the worksheet may word a rule differently; they must not derive
    different facts. Both render from one typed rule status and reason, so the
    only way they can disagree is if the runtimes moved between confirming and
    running — which `check_versions` refuses."""

    @property
    def blocking(self) -> Sequence[str]:
        """Unresolved fields the engine refuses to proceed without.

        A corporate-action reference and a withholding method are not defaults
        anyone can supply on the user's behalf: one decides whether share counts
        mean anything, the other decides how many shares arrive.
        """
        required = {"withholding_method", "corporate_action_ref"}
        return tuple(one.field_name for one in self.unresolved
                     if one.field_name in required)

    @property
    def can_run(self) -> bool:
        return not self.blocking

    @property
    def all_fields(self) -> Sequence[CardField]:
        return tuple(self.described) + tuple(self.inferred) + tuple(self.unresolved)

    def to_json(self) -> Dict[str, Any]:
        return {"described": [one.to_json() for one in self.described],
                "inferred": [one.to_json() for one in self.inferred],
                "unresolved": [one.to_json() for one in self.unresolved],
                "will_model": list(self.will_model),
                "will_not_model": list(self.will_not_model),
                "blocking": list(self.blocking), "can_run": self.can_run,
                "version_pin": self.version_pin,
                "versions": dict(self.versions),
                "scope": dict(self.scope) if self.scope else None}


def _render(value: Any) -> str:
    if isinstance(value, float) and 0 < value < 1:
        return f"{value:.0%}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(one) for one in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k} {v:.0%}" for k, v in value.items())
    return str(value)


def build(declaration: RSUDeclaration, *, runtime,
          inferred: Mapping[str, tuple] = (),
          defaults_ref: str = "",
          scope: Optional[Mapping[str, Any]] = None) -> RSUConfirmationCard:
    """Assemble the card from declarations and runtime statements.

    `runtime` is the `RSUVestingRuntime` whose assumptions and limitations also
    build the result context after the run. Reading the same source is what
    stops the card promising something the result then does not carry.
    """
    stated: List[CardField] = []
    guessed: List[CardField] = []
    open_questions: List[CardField] = []

    inferred = dict(inferred or {})
    for name, label in LABELS.items():
        value = getattr(declaration, name, None)

        if value is None:
            open_questions.append(CardField(
                label=label, value="not stated",
                destination=Destination.UNRESOLVED_QUESTION,
                why=QUESTIONS.get(name, f"{label} was not stated."),
                field_name=name))
            continue

        if name in inferred:
            guessed.append(CardField(
                label=label, value=_render(value),
                destination=Destination.ENGINE_INPUT,
                why=inferred[name][0] if inferred[name] else "",
                defaults_ref=defaults_ref, field_name=name))
            continue

        # Runtime references describe rules; everything else is an engine input.
        destination = (Destination.RUNTIME_DECLARATION
                       if name.endswith("_ref") else Destination.ENGINE_INPUT)
        stated.append(CardField(label=label, value=_render(value),
                                destination=destination, field_name=name))

    # Read from the runtime, not restated here. Restated, the card and the
    # result's modelling scope become two lists that drift.
    will_model = tuple(one.statement for one in runtime.assumptions)
    will_not_model = tuple(one.statement for one in runtime.limitations)

    if scope is None:
        from ..runtime.rsu import IMPLEMENTED as RSU_IMPLEMENTED
        from .scope_disclosure import for_rsu

        scope = for_rsu(runtime, implemented=RSU_IMPLEMENTED).to_json()

    return RSUConfirmationCard(
        scope=scope,
        described=tuple(stated), inferred=tuple(guessed),
        unresolved=tuple(open_questions),
        will_model=will_model, will_not_model=will_not_model,
        version_pin=declaration.versions.pin,
        versions={k: v for k, v in declaration.versions.to_json().items()
                  if k != "pin"})
