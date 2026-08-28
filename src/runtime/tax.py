"""Tax as a runtime, because a string could not carry it.

`tax_treatment: str = "NONE_APPLIED"` was already in `ISOLATION_DIMENSIONS`,
compared by string equality — so two plans both saying `NONE_APPLIED` compared as
identical on that dimension even if one was a Roth and the other taxable. The
comparability engine would report `STRATEGY_EFFECT`, *attribution isolated*, for
a pair whose tax treatment differed completely. That is a wrong verdict the tests
could not catch, because there was nothing to catch it with.

The same shape `calendar` had before it became an artifact.

**Nothing here computes anybody's tax liability.** A runtime declaring
`models_capital_gains=False` is doing its job: it states, in a versioned and
comparable way, exactly which mechanics apply, and every mechanic it does not
perform is a named limitation rather than an absence someone has to notice.

Two layers, one boundary
------------------------
The enhancement plan (§9) splits tax into two explicit layers::

    TaxPolicyRuntime           declaration / jurisdiction / assumptions   (here)
             ↓
    TaxRealizationEngine       lots / realized gains / wash sales         (there)

This module **is** the declaration layer. `TaxRuntime` — aliased below as
`TaxPolicyRuntime`, the plan's declaration-layer name — states which mechanics
apply for evaluation and comparability; it never realizes a single lot. The
*realization* layer — execution-grade realized short/long gains, wash-sale
disallowance and cross-account awareness computed over a real lot ledger for a
specific household — lives in **wealth-manager's `TaxRealizationEngine`**
(`wealth_manager/tax_engine.py`), not here. RAAAL declares tax mechanics so two
strategies are genuinely comparable on tax; it does not compute a person's
liability. That boundary is load-bearing: an assumption declared here whose
`realized_by` mechanism is absent from `IMPLEMENTED` is reported by
`unrealized(...)` precisely because this layer must not pretend to compute what
only the realization engine can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Sequence

from .base import RuntimeArtifact, RuntimeAssumption, RuntimeLimitation


class LotMethod(str, Enum):
    FIFO = "FIFO"
    LIFO = "LIFO"
    SPECIFIC_ID = "SPECIFIC_ID"
    AVERAGE_COST = "AVERAGE_COST"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    """No lots are tracked — correct for a tax-deferred account, and a real
    answer rather than a missing one."""


@dataclass(frozen=True)
class TaxRuntime(RuntimeArtifact):
    """Which tax mechanics apply, declared and versioned."""

    kind: ClassVar[str] = "tax"
    undefined_without: ClassVar[Sequence[str]] = ("account",)
    """Untaxed gains mean "not yet" in a 401(k) and "we did not model it" in a
    taxable account. The same declaration, two different statements — so the
    declaration has no truth value until an account runtime is present."""

    name: str
    version: int
    jurisdiction: str
    """e.g. "US-federal", "US-CA", "UK". Part of comparability: the same rule
    under two jurisdictions is two different rules."""

    lot_method: LotMethod = LotMethod.NOT_APPLICABLE
    supplemental_withholding_rate: Optional[float] = None
    capital_gains_short_rate: Optional[float] = None
    capital_gains_long_rate: Optional[float] = None
    long_term_holding_days: Optional[int] = None
    dividend_rate: Optional[float] = None
    wash_sale_enabled: bool = False
    effective_from: str = ""
    title: str = ""
    citations: Sequence[str] = ()

    @property
    def models_capital_gains(self) -> bool:
        return self.capital_gains_short_rate is not None

    @property
    def models_withholding(self) -> bool:
        return self.supplemental_withholding_rate is not None

    def declared_form(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "jurisdiction": self.jurisdiction,
            "lot_method": self.lot_method.value,
            "supplemental_withholding_rate": self.supplemental_withholding_rate,
            "capital_gains_short_rate": self.capital_gains_short_rate,
            "capital_gains_long_rate": self.capital_gains_long_rate,
            "long_term_holding_days": self.long_term_holding_days,
            "dividend_rate": self.dividend_rate,
            "wash_sale_enabled": self.wash_sale_enabled,
            "effective_from": self.effective_from,
            "title": self.title,
            "citations": sorted(self.citations),
        }

    def comparable_form(self) -> Dict[str, Any]:
        """Everything that could move a number, and nothing that cannot.

        `title` and `citations` are excluded: correcting a citation is a real
        change to the artifact and no change at all to the figures it produced.
        Without the split, fixing a typo would sever comparability across a whole
        lineage of results.
        """
        declared = self.declared_form()
        for prose in ("title", "citations", "name", "version"):
            declared.pop(prose, None)
        return declared

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        out = []
        if self.models_withholding:
            out.append(RuntimeAssumption(
                name="supplemental-withholding",
                statement=(
                    f"Supplemental wages are withheld at "
                    f"{self.supplemental_withholding_rate:.0%} in "
                    f"{self.jurisdiction}."),
                realized_by="withholding_for",
                risk=("A withholding rate is not a tax rate. Anyone whose "
                      "marginal rate is higher under-withholds and owes the "
                      "difference at filing."),
                citation=self.citations[0] if self.citations else None,
            ))
        if self.models_capital_gains:
            out.append(RuntimeAssumption(
                name="capital-gains",
                statement=(
                    f"Realized gains are taxed at "
                    f"{self.capital_gains_short_rate:.0%} short and "
                    f"{(self.capital_gains_long_rate or 0):.0%} long, with lots "
                    f"selected {self.lot_method.value}."),
                realized_by="realize_gain",
                citation=self.citations[0] if self.citations else None,
            ))
        if self.wash_sale_enabled:
            out.append(RuntimeAssumption(
                name="wash-sale",
                statement="Losses on repurchase within the wash-sale window are "
                          "disallowed and added to basis.",
                realized_by="apply_wash_sale",
            ))
        return tuple(out)

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        out = []
        if not self.models_capital_gains:
            out.append(RuntimeLimitation(
                name="no-capital-gains",
                statement=("Gains are not taxed under this runtime. Figures are "
                           "pre-tax and are stated as such rather than assuming "
                           "a rate nobody supplied."),
                applicable_unless=("account:tax_deferred",),
            ))
        if not self.wash_sale_enabled:
            out.append(RuntimeLimitation(
                name="no-wash-sale",
                statement=("Wash-sale disallowance is not applied, so a strategy "
                           "that repurchases quickly after a loss will look "
                           "better here than it would after filing."),
            ))
        if self.lot_method is LotMethod.NOT_APPLICABLE and self.models_capital_gains:
            out.append(RuntimeLimitation(
                name="no-lot-tracking",
                statement="Gains are taxed without tracking individual lots.",
            ))
        return tuple(out)


#: The declaration-layer name from the plan (§9). ``TaxPolicyRuntime`` *is*
#: ``TaxRuntime`` — a plain alias, so every existing ``TaxRuntime`` caller,
#: instance, ``isinstance`` check and equality keeps working unchanged, and code
#: that wants to name the two-layer split explicitly can say ``TaxPolicyRuntime``
#: to mean "the declaration layer, which does not compute liability — that is the
#: realization engine's job, in wealth-manager". A subclass would fork identity
#: (dataclass ``__eq__`` compares ``__class__``); an alias keeps one type and one
#: comparable form, which is the whole point of a declaration artifact.
TaxPolicyRuntime = TaxRuntime


#: Nothing modelled, and it says so. The honest default, and a real artifact
#: rather than the absence of one — two plans citing it are genuinely comparable
#: on tax, which the old `"NONE_APPLIED"` string only appeared to establish.
PRE_TAX = TaxRuntime(
    name="pre-tax", version=1, jurisdiction="none",
    title="Pre-tax — no tax mechanics applied",
)

#: US federal supplemental withholding only. What the RSU template performs.
US_FEDERAL_WITHHOLDING = TaxRuntime(
    name="us-federal-withholding", version=1, jurisdiction="US-federal",
    supplemental_withholding_rate=0.22,
    effective_from="2026-01-01",
    title="US federal supplemental wage withholding",
    citations=("irs-pub-15-supplemental",),
)

#: A tax-deferred account: contributions and growth are untaxed in-account, so
#: no lots are tracked and no gains are realized. Declining to model gains here
#: is correct rather than incomplete.
TAX_DEFERRED = TaxRuntime(
    name="tax-deferred-account", version=1, jurisdiction="US-federal",
    lot_method=LotMethod.NOT_APPLICABLE,
    title="Tax-deferred account — no in-account taxation",
)

#: Implemented mechanics, for the realization check.
IMPLEMENTED = ("withholding_for",)
