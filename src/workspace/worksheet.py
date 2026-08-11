"""The ResearchWorksheet: a saved research result you can return to.

    confirmed scenario -> historical run -> worksheet revision 1
                                         -> reopen -> replay stored artifacts

**References, never copied result state.** A worksheet that holds a copy of a
figure has two sources of truth for it, and the copy is the one that goes stale
without saying so. Every block names the artifacts it needs and reads them at
render time.

**Opening never recompiles.** The plan page already had this defect — it
recompiled the stored prose and simulated the fresh interpretation while
displaying the stored scenario — and a worksheet makes it worse, because a
worksheet is the thing a user comes back to. Reinterpretation is a separate,
named action that produces a *new revision*, never a silent redraw of an old
one.

The block registry is deliberately small. Python cells, arbitrary markdown,
custom charts and plugin blocks are all deferred: a worksheet nobody wants to
modify does not need an extension mechanism, and building one first guarantees
the wrong extension points.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

WORKSHEET_SPEC_VERSION = "0.1"


class Block(str, Enum):
    """The initial registry. Each entry earns its place by answering a question
    a reader has when they reopen a worksheet months later."""

    INTERPRETATION_SUMMARY = "InterpretationSummaryBlock"
    STRATEGY_DEFINITION = "StrategyDefinitionBlock"
    BENCHMARK_COMPARISON = "BenchmarkComparisonBlock"
    PERFORMANCE_SUMMARY = "PerformanceSummaryBlock"
    MODELING_SCOPE = "ModelingScopeBlock"
    TRIAL_ACCOUNTING = "TrialAccountingBlock"
    ARTIFACT_CHAIN = "ArtifactChainBlock"


@dataclass(frozen=True)
class BlockSpec:
    """What a block needs, and what it is allowed to be."""

    block: Block
    title: str
    requires: Sequence[str]
    """Artifact references, by field name on the worksheet. A block whose
    requirements are unmet is omitted with a reason rather than rendered empty —
    an empty panel reads as "nothing to report", which is a different claim."""

    affects_identity: bool = False
    """Whether changing it makes a new revision. Presentation does not."""

    deterministic: bool = True
    exportable: bool = True


REGISTRY: Mapping[Block, BlockSpec] = {
    Block.INTERPRETATION_SUMMARY: BlockSpec(
        Block.INTERPRETATION_SUMMARY, "What was understood",
        requires=("scenario_ref",), affects_identity=True),
    Block.STRATEGY_DEFINITION: BlockSpec(
        Block.STRATEGY_DEFINITION, "What was tested",
        requires=("scenario_ref",), affects_identity=True),
    Block.BENCHMARK_COMPARISON: BlockSpec(
        Block.BENCHMARK_COMPARISON, "Compared with",
        requires=("primary_run_ref", "benchmark_run_refs"),
        affects_identity=True),
    Block.PERFORMANCE_SUMMARY: BlockSpec(
        Block.PERFORMANCE_SUMMARY, "Results",
        requires=("primary_run_ref",), affects_identity=True),
    Block.MODELING_SCOPE: BlockSpec(
        Block.MODELING_SCOPE, "Modelling scope",
        requires=("primary_run_ref",), affects_identity=True),
    Block.TRIAL_ACCOUNTING: BlockSpec(
        Block.TRIAL_ACCOUNTING, "Alternatives evaluated",
        requires=("scenario_ref",), affects_identity=True),
    Block.ARTIFACT_CHAIN: BlockSpec(
        Block.ARTIFACT_CHAIN, "Provenance",
        requires=("scenario_ref",), affects_identity=False),
}

#: Reading order. Strategy before results before scope, because a figure read
#: before its exclusions is read as excluding nothing.
DEFAULT_LAYOUT: Sequence[Block] = (
    Block.INTERPRETATION_SUMMARY,
    Block.STRATEGY_DEFINITION,
    Block.BENCHMARK_COMPARISON,
    Block.PERFORMANCE_SUMMARY,
    Block.MODELING_SCOPE,
    Block.TRIAL_ACCOUNTING,
    Block.ARTIFACT_CHAIN,
)


class WorksheetError(ValueError):
    """A worksheet could not be created or is not what it claims to be."""


@dataclass(frozen=True)
class ResearchWorksheet:
    """One immutable revision.

    Holds references. The only values on it are identity and the reason it
    exists — everything a reader sees is fetched from the artifacts it names.
    """

    worksheet_id: str
    owner_id: str
    revision: int
    scenario_ref: str
    primary_run_ref: Optional[str] = None
    benchmark_run_refs: Sequence[str] = ()
    mission_ref: Optional[str] = None
    context_view_ref: Optional[str] = None
    layout: Sequence[Block] = DEFAULT_LAYOUT
    parent_revision: Optional[int] = None
    change_reason: str = ""
    title: str = ""
    created_at: Optional[str] = None
    spec_version: str = WORKSHEET_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise WorksheetError("a worksheet revision starts at 1")
        if self.revision > 1 and self.parent_revision is None:
            raise WorksheetError(
                f"{self.worksheet_id} revision {self.revision} names no parent; "
                "a revision that cannot say what it came from is not a history")
        if self.revision > 1 and not self.change_reason:
            raise WorksheetError(
                f"{self.worksheet_id} revision {self.revision} gives no reason. "
                "A history of unexplained changes cannot be reviewed, which is "
                "the only thing keeping revisions is for")

    def canonical_form(self) -> Dict[str, Any]:
        """Identity is the references and the layout, not the rendered result.

        `created_at`, the title and the change reason are excluded: two
        worksheets over the same artifacts are the same worksheet whenever they
        were made and whatever they are called.
        """
        return {
            "spec_version": self.spec_version,
            "scenario_ref": self.scenario_ref,
            "primary_run_ref": self.primary_run_ref,
            "benchmark_run_refs": sorted(self.benchmark_run_refs),
            "mission_ref": self.mission_ref,
            "context_view_ref": self.context_view_ref,
            "layout": [b.value for b in self.layout],
        }

    @property
    def canonical_hash(self) -> str:
        body = json.dumps(self.canonical_form(), sort_keys=True,
                          separators=(",", ":"))
        return "wsv1:" + hashlib.sha256(body.encode()).hexdigest()

    def blocks(self) -> List[BlockSpec]:
        return [REGISTRY[b] for b in self.layout]

    def unavailable_blocks(self) -> Dict[str, str]:
        """Blocks whose requirements are unmet, and what is missing.

        Named rather than skipped. An omitted panel is invisible; a panel that
        says "no run yet" is a fact.
        """
        out: Dict[str, str] = {}
        for spec in self.blocks():
            missing = [name for name in spec.requires if not getattr(self, name)]
            if missing:
                out[spec.block.value] = (
                    f"needs {', '.join(missing)}, which this worksheet does not "
                    "have yet")
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "worksheet_id": self.worksheet_id,
            "owner_id": self.owner_id,
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "change_reason": self.change_reason,
            "title": self.title,
            "created_at": self.created_at,
            "canonical_hash": self.canonical_hash,
        }


def from_json(payload: Mapping[str, Any]) -> ResearchWorksheet:
    return ResearchWorksheet(
        worksheet_id=payload["worksheet_id"],
        owner_id=payload["owner_id"],
        revision=int(payload["revision"]),
        scenario_ref=payload["scenario_ref"],
        primary_run_ref=payload.get("primary_run_ref"),
        benchmark_run_refs=tuple(payload.get("benchmark_run_refs") or ()),
        mission_ref=payload.get("mission_ref"),
        context_view_ref=payload.get("context_view_ref"),
        layout=tuple(Block(b) for b in (payload.get("layout") or ())) or DEFAULT_LAYOUT,
        parent_revision=payload.get("parent_revision"),
        change_reason=payload.get("change_reason", ""),
        title=payload.get("title", ""),
        created_at=payload.get("created_at"),
    )


def create(*, worksheet_id: str, owner_id: str, scenario_ref: str,
           primary_run_ref: Optional[str] = None,
           benchmark_run_refs: Sequence[str] = (),
           title: str = "", created_at: Optional[str] = None
           ) -> ResearchWorksheet:
    return ResearchWorksheet(
        worksheet_id=worksheet_id, owner_id=owner_id, revision=1,
        scenario_ref=scenario_ref, primary_run_ref=primary_run_ref,
        benchmark_run_refs=tuple(benchmark_run_refs), title=title,
        created_at=created_at)


def revise(previous: ResearchWorksheet, *, reason: str,
           created_at: Optional[str] = None, **changes: Any
           ) -> ResearchWorksheet:
    """A new revision. The old one is never edited.

    `reason` is required by the constructor rather than defaulted, because a
    history of unexplained changes cannot be reviewed and reviewing is the only
    thing keeping revisions is for.
    """
    fields = {**previous.__dict__, **changes,
              "revision": previous.revision + 1,
              "parent_revision": previous.revision,
              "change_reason": reason,
              "created_at": created_at}
    return ResearchWorksheet(**fields)
