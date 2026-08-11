"""One compile, shared by the page and by the save.

F11 was not three missing arguments. It was:

    the system showed one compiled scenario to the user and persisted another

The preview passed `priceable` to `compile_scenario` and the save path did
not, so `_funding_policy` could find no subject and every saved plan was
written with `funding=None`. Both bodies were internally consistent, so
`content_hash` did not notice — it covers the compiled artifact, not the
inputs that decided it. The page showed a plan that ran; the stored artifact
could never execute.

Two mechanisms here, and only the first is a guarantee:

**`compile_draft` is the single compile.** Any argument that decides the
result is chosen here, once, so the preview and the save cannot disagree about
one. There is no longer a second call site to keep in step. This is
structural: it removes the opportunity rather than detecting the mistake.

**The draft token is a tripwire under it.** The page emits the digest of what
it rendered along with a digest of the inputs that produced it; the save
recomputes both. If the inputs are the same and the scenarios are not, the
save refuses with `DRAFT_DIVERGED` rather than silently choosing a version.
It exists because `compile_draft` is a convention — a future call site can
still bypass it — and a convention with no detector decays quietly.

**The token is not an authority.** It carries digests and nothing else. It is
never a source of a persisted value; the scenario that gets stored is always
the one this server compiled. So it is deliberately unsigned: forging it can
only suppress a self-check on your own submission, and cannot introduce a
value into anyone's plan. The stronger design the token is a stand-in for —
persist the compiled draft server-side and have the save reference it by id,
so nothing is recompiled at all — needs a draft store, which is more than this
correctness slice should build.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

#: Named so a refusal is greppable and so the reason survives into telemetry
#: as a value rather than as prose.
DRAFT_DIVERGED = "DRAFT_DIVERGED"

TOKEN_VERSION = "draft-token@1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def input_digest(describe: str, *, parsed, amendments=(), exclusions=()) -> str:
    """Everything the compile consumes, and nothing that varies per request.

    `recorded_at` is deliberately absent. Amendments are stamped with the
    wall-clock at the moment the form is handled, so including it would make
    every save look like divergent input and the gate would never compare
    anything. What identifies an amendment is which question it answers and
    with what.

    Sorted, because the form's field order is a property of the browser rather
    than of the plan: two submissions that carry the same answers in a
    different order describe the same draft.
    """
    return _digest({
        "v": TOKEN_VERSION,
        "describe": describe,
        "parsed": (parsed.to_json() if parsed is not None else None),
        "amendments": sorted((one.question_id, str(one.answer))
                             for one in amendments),
        "exclusions": sorted(one.item for one in exclusions),
    })


@dataclass(frozen=True)
class DraftInputs:
    """The answers and acknowledgements the preview compiled with.

    Emitted beside the token because a digest cannot be recompiled from. The
    save needs the actual set to reproduce the page's compile, and it must be
    the preview's own statement of what it used rather than a reconstruction
    from the POST body — the body also carries the decisions made on that very
    screen, which the preview did not have.

    Deliberately separate from the `answer:*` fields the form replays. Those
    are the interaction's accumulated state; this is one page's claim about
    what it compiled. Duplicating them is what makes the comparison
    independent rather than circular.
    """

    amendments: tuple
    exclusions: tuple

    @staticmethod
    def of(amendments=(), exclusions=()) -> "DraftInputs":
        return DraftInputs(
            amendments=tuple(sorted((one.question_id, str(one.answer))
                                    for one in amendments)),
            exclusions=tuple(sorted((one.item, one.reason, one.decision)
                                    for one in exclusions)))

    def encode(self) -> str:
        return json.dumps(
            {"amendments": [list(one) for one in self.amendments],
             "exclusions": [list(one) for one in self.exclusions]},
            sort_keys=True)

    @staticmethod
    def decode(raw: str) -> Optional["DraftInputs"]:
        """None for anything unreadable. The caller reports that as
        `NOT_COMPARED` with a reason, never as agreement."""
        try:
            body = json.loads(raw or "")
        except (ValueError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        try:
            return DraftInputs(
                amendments=tuple(sorted((str(field), str(value))
                                        for field, value
                                        in body.get("amendments") or ())),
                exclusions=tuple(sorted((str(item), str(why), str(decision))
                                        for item, why, decision
                                        in body.get("exclusions") or ())))
        except (TypeError, ValueError):
            return None

    def as_amendments(self, recorded_at: str):
        from ..mission.spec import ScenarioAmendment

        return tuple(ScenarioAmendment(question_id=field, answer=value,
                                       recorded_at=recorded_at)
                     for field, value in self.amendments)

    def as_exclusions(self, recorded_at: str):
        from ..mission.spec import ScenarioExclusion

        return tuple(ScenarioExclusion(item=item, reason=why,
                                       decision=decision,
                                       acknowledged_at=recorded_at)
                     for item, why, decision in self.exclusions)


@dataclass(frozen=True)
class DraftToken:
    """What the page rendered, in two digests."""

    inputs: str
    semantic: str

    def encode(self) -> str:
        return f"{TOKEN_VERSION}:{self.inputs}:{self.semantic}"

    @staticmethod
    def decode(raw: str) -> Optional["DraftToken"]:
        """None for anything unreadable, including the empty string.

        A malformed token is not an error to raise at the user. It means this
        submission carries no claim about what was shown, so there is nothing
        to compare — the same position as a form posted by a test or by an
        older page. The distinction that matters is recorded by the caller as
        `checked` versus `unchecked`, so a run where nothing was ever compared
        cannot be read as a run where everything agreed.
        """
        parts = (raw or "").split(":")
        if len(parts) != 3 or parts[0] != TOKEN_VERSION:
            return None
        if not parts[1] or not parts[2]:
            return None
        return DraftToken(inputs=parts[1], semantic=parts[2])


def token_for(scenario, describe: str, *, parsed, amendments=(),
              exclusions=()) -> str:
    return DraftToken(
        inputs=input_digest(describe, parsed=parsed, amendments=amendments,
                            exclusions=exclusions),
        semantic=scenario.semantic_digest).encode()


@dataclass(frozen=True)
class DraftCheck:
    """The outcome of the comparison, as a value rather than an exception.

    Three states, because two would hide the one that matters. `AGREED` and
    `DIVERGED` are the comparison; `NOT_COMPARED` says the comparison did not
    happen, and is emitted with a reason. A gate that reported only pass and
    fail would report a submission it never examined as a pass.
    """

    state: str
    reason: str = ""

    AGREED = "AGREED"
    DIVERGED = "DIVERGED"
    NOT_COMPARED = "NOT_COMPARED"

    @property
    def diverged(self) -> bool:
        return self.state == self.DIVERGED


def check(raw_token: str, raw_inputs: str, describe: str, *, parsed,
          name: str, at: str, context: str) -> DraftCheck:
    """Did the save path reproduce the scenario the page showed?

    Answered by replay, not by comparing the stored scenario with the token.
    A save always carries information the preview did not have — the answers
    and acknowledgements made on that very screen arrive in the same POST — so
    the scenario about to be stored is legitimately different from the one
    rendered, and comparing them directly would either fire on every
    productive save or, if relaxed to "only when the inputs match", compare
    nothing at all. The second is worse: it reports `NOT_COMPARED` on every
    real journey while looking like a working gate.

    So: recompile the preview's own stated inputs, here, in the save path's
    context, and require the result to be the scenario the preview rendered.
    Same inputs by construction, so any difference is drift between the two
    paths — which is exactly and only what F11 was.

    The extra compile is deterministic and makes no provider call; the parse
    is pinned by this point.
    """
    token = DraftToken.decode(raw_token)
    if token is None:
        return DraftCheck(DraftCheck.NOT_COMPARED, "no draft token submitted")

    stated = DraftInputs.decode(raw_inputs)
    if stated is None:
        return DraftCheck(DraftCheck.NOT_COMPARED,
                          "the draft token carries no statement of what the "
                          "preview compiled")

    amendments = stated.as_amendments(at)
    exclusions = stated.as_exclusions(at)
    if input_digest(describe, parsed=parsed, amendments=amendments,
                    exclusions=exclusions) != token.inputs:
        # The two hidden fields disagree with each other. Nothing can be
        # concluded from a self-inconsistent claim, and the safe reading is
        # that this submission makes none.
        return DraftCheck(DraftCheck.NOT_COMPARED,
                          "the submitted draft inputs do not match the token")

    replayed = compile_draft(describe, name=name, parsed=parsed,
                             amendments=amendments, exclusions=exclusions,
                             context=context)
    if replayed.scenario.semantic_digest != token.semantic:
        return DraftCheck(DraftCheck.DIVERGED,
                          "the same description and answers compiled to a "
                          "different scenario on the way to storage")
    return DraftCheck(DraftCheck.AGREED)


def priceable_for(context: str) -> Sequence[str]:
    """What this deployment can price, resolved the one way.

    Read through `routes` rather than `market_data.access` directly so that a
    test which stubs the route's data access is stubbing this too — otherwise
    the preview and the save could be given different frames by the test
    harness and the equivalence result would be about the harness.
    """
    from . import routes

    access = routes._market_data(context)
    columns = getattr(access.frame, "columns", None)
    if columns is None or not access.usable:
        return ()
    return tuple(columns)


def compile_draft(describe: str, *, name: str, version: int = 1, parsed=None,
                  amendments=(), exclusions=(), context: str = "draft"):
    """The compile. Both entry points come here.

    Every argument that can change the result is decided in this function.
    Callers supply what the user said and what the user has answered; they do
    not get a say in what the deployment can price, which benchmark policy
    applies, or any other compile input — those were the fields that drifted.
    """
    from ..mission.compiler import compile_scenario
    from . import routes

    return compile_scenario(describe, name=name, version=version,
                            benchmark_rule=routes.BENCHMARK_RULE,
                            parsed=parsed,
                            amendments=tuple(amendments),
                            exclusions=tuple(exclusions),
                            priceable=priceable_for(context))
