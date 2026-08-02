"""RSU vesting: what a vest does, and what this system does not claim about it.

    grant -> vest event -> withholding -> delivered shares
                        -> optional disposition window
                        -> diversification allocation

A vest is the first event in this system that touches ownership, cash flow,
taxation, concentration, liquidity and benchmark attribution at once. A partially
modelled vest therefore produces a figure that looks complete while being wrong
in several directions simultaneously, which is why the declarations come before
any result surface.

**Six things kept apart.** Share delivery, withholding mechanics, sale, tax
liability, capital-gains treatment and allocation of proceeds are distinct
questions with distinct evidence. Collapsing them is how "78 shares arrived"
becomes "your after-tax position is X" — a claim this runtime does not make and
cannot currently support.

**Withholding is not tax.** A withholding rate is a statutory remittance rate,
not anybody's marginal rate. Someone whose true rate is higher under-withholds
and owes the difference at filing, and this runtime models the remittance only.

Nothing here infers a jurisdiction, a marginal rate, a cost-basis method, a
disposition treatment, a state tax, or whether proceeds land in a taxable or a
retirement account. Each is left unresolved and named, because a vest is exactly
the situation where a plausible default is indistinguishable from a fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence

from .base import (
    Exclusion,
    RuntimeArtifact,
    RuntimeAssumption,
    RuntimeLimitation,
)


class WithholdingMethod(str, Enum):
    SHARE_WITHHOLDING = "SHARE_WITHHOLDING"
    """The employer keeps shares. Fewer shares arrive; no order is placed."""

    SELL_TO_COVER = "SELL_TO_COVER"
    """All shares arrive and a broker sells some. A market transaction with
    costs and a price distinct from the vest price."""

    UNSPECIFIED = "UNSPECIFIED"
    """Not stated, and not inferred. The two methods deliver different share
    counts and different cost bases; picking one silently would decide the
    result."""


class DispositionPolicy(str, Enum):
    HOLD = "HOLD"
    """A real strategy, and what happens when nobody decides. Named so that
    diversifying is a choice rather than an intervention."""

    SELL_ALL_AND_DIVERSIFY = "SELL_ALL_AND_DIVERSIFY"
    SELL_HALF_AND_DIVERSIFY = "SELL_HALF_AND_DIVERSIFY"
    UNSPECIFIED = "UNSPECIFIED"


#: Facts a vest cannot be modelled without, and which must never be guessed.
#: Named individually so an unresolved plan says which one it is missing.
REQUIRED_TO_MODEL = ("employer_ticker", "vest_date", "gross_shares",
                     "vest_price_source", "withholding_method",
                     "market_data_ref", "corporate_action_ref")

#: Facts this runtime deliberately does not infer. Present here means "ask", not
#: "default". Each changes the answer, and none is recoverable from the others.
NEVER_INFERRED = (
    "tax_jurisdiction", "marginal_tax_rate", "cost_basis_method",
    "disposition_treatment", "state_tax", "account_destination",
)


@dataclass(frozen=True)
class VestEvent:
    """One vest, pinned. Unknown fields stay None and are reported, not filled."""

    grant_id: str
    employer_ticker: str
    vest_date: str
    gross_shares: float

    vest_price_source: Optional[str] = None
    """Which price series and which session decided the vest value. A vest
    price is the basis of the withholding, the external flow and the cost
    basis; taking it from whatever data happens to be loaded makes three
    figures depend on an unrecorded choice."""

    withholding_method: WithholdingMethod = WithholdingMethod.UNSPECIFIED
    withholding_rate: Optional[float] = None
    shares_withheld: Optional[float] = None
    shares_delivered: Optional[float] = None

    blackout_window: Sequence[tuple] = ()
    disposition_policy: DispositionPolicy = DispositionPolicy.UNSPECIFIED
    allocation_rule: Optional[str] = None
    account_destination: Optional[str] = None

    tax_runtime_ref: Optional[str] = None
    market_data_ref: Optional[str] = None
    corporate_action_ref: Optional[str] = None
    """The corporate-action policy this vest was computed under.

    Absent, the share count between grant and vest is being trusted blindly. A
    two-for-one split turns 100 granted shares into 200 vested ones, and a
    merger may replace them with something else entirely — so an unpinned vest
    is blocked rather than computed from raw numbers."""

    def unresolved(self) -> Sequence[str]:
        """Required facts this vest does not carry."""
        missing = []
        for name in REQUIRED_TO_MODEL:
            value = getattr(self, name, None)
            if value is None or value == "" or (
                    isinstance(value, Enum) and value.name == "UNSPECIFIED"):
                missing.append(name)
        return tuple(missing)

    @property
    def modellable(self) -> bool:
        return not self.unresolved()

    def to_json(self) -> Dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "employer_ticker": self.employer_ticker,
            "vest_date": self.vest_date,
            "gross_shares": self.gross_shares,
            "vest_price_source": self.vest_price_source,
            "withholding_method": self.withholding_method.value,
            "withholding_rate": self.withholding_rate,
            "shares_withheld": self.shares_withheld,
            "shares_delivered": self.shares_delivered,
            "blackout_window": [list(w) for w in self.blackout_window],
            "disposition_policy": self.disposition_policy.value,
            "allocation_rule": self.allocation_rule,
            "account_destination": self.account_destination,
            "tax_runtime_ref": self.tax_runtime_ref,
            "market_data_ref": self.market_data_ref,
            "corporate_action_ref": self.corporate_action_ref,
            "unresolved": list(self.unresolved()),
            "modellable": self.modellable,
        }


class UnpinnedVest(ValueError):
    """A vest that cannot be modelled from what it carries."""


# --- mechanisms ------------------------------------------------------------
#
# These are what `realized_by` names below. Each delegates to the engine that
# already performs the work rather than restating it: a second implementation of
# the withholding split would be a second answer to one question.


def apply_supplemental_wage_threshold(vest_value: float, *, rate: float,
                                      cumulative_supplemental: float = 0.0
                                      ) -> float:
    """Withholding on one vest, split where it straddles the annual threshold.

    The threshold is cumulative across the calendar year. Applying one rate to
    the whole vest is wrong in the direction that flatters the result.
    """
    from ..mission.templates.rsu import withholding_for

    return withholding_for(vest_value, rate=rate,
                           cumulative_supplemental=cumulative_supplemental)


def apply_share_withholding(gross_shares: float, vest_price: float, *,
                            rate: float, method: WithholdingMethod,
                            cumulative_supplemental: float = 0.0
                            ) -> Dict[str, float]:
    """Split a gross vest into withheld and delivered shares.

    Returns both counts rather than only the net. "78 arrived" and "100 vested,
    22 withheld" answer different questions, and the second is the one a user
    checks against their statement.
    """
    if method is WithholdingMethod.UNSPECIFIED:
        raise UnpinnedVest(
            "withholding method is unspecified. Share withholding and "
            "sell-to-cover deliver different share counts and different cost "
            "bases, so neither can stand in for the other")
    if vest_price <= 0:
        raise UnpinnedVest(
            "vest price is missing or non-positive, so the vest has no value "
            "to withhold against. This is a data gap, not a free vest")

    value = gross_shares * vest_price
    withheld_value = apply_supplemental_wage_threshold(
        value, rate=rate, cumulative_supplemental=cumulative_supplemental)
    withheld_shares = withheld_value / vest_price
    return {"gross_shares": gross_shares,
            "withheld_shares": withheld_shares,
            "delivered_shares": gross_shares - withheld_shares,
            "vest_value": value,
            "withheld_value": withheld_value}


def apply_vest_delivery(vest: VestEvent, *, vest_price: float,
                        cumulative_supplemental: float = 0.0):
    """The in-kind delivery a vest produces.

    Never "cash arrives, then buy employer stock". That model puts a session of
    slippage between the vest and the holding and credits the plan with a
    trading decision nobody made. Withheld shares are simply not delivered —
    they are never granted and then sold, so no intermediate state exists in
    which the full gross count was owned.
    """
    import pandas as pd

    from ..mission.accounting import Grant

    if not vest.modellable:
        raise UnpinnedVest(
            f"vest {vest.grant_id} is missing: {', '.join(vest.unresolved())}")

    rate = vest.withholding_rate
    if rate is None:
        raise UnpinnedVest(
            f"vest {vest.grant_id} states no withholding rate. A statutory "
            "remittance rate is not recoverable from the other fields")

    split = apply_share_withholding(
        vest.gross_shares, vest_price, rate=rate,
        method=vest.withholding_method,
        cumulative_supplemental=cumulative_supplemental)

    return Grant(date=pd.Timestamp(vest.vest_date),
                 ticker=vest.employer_ticker,
                 shares=split["delivered_shares"],
                 reason=(f"vest {vest.grant_id}, net of "
                         f"{split['withheld_shares']:.4f} shares withheld")), split


def next_eligible_disposition_session(session, sessions, blackouts):
    """First tradeable session on or after `session`, outside any blackout.

    Deferral, not cancellation. Dropping a blocked sale would silently convert a
    diversification plan into a hold — a different strategy, with a different
    result, that the user never chose.
    """
    from ..mission.templates.rsu import next_open_session

    return next_open_session(session, sessions, blackouts)


def allocate_disposition_proceeds(proceeds: float, allocation_rule: Optional[str]
                                  ) -> Dict[str, float]:
    """Where the money from a sale goes.

    Refuses rather than defaulting to a house index. Proceeds are the user's
    money and the destination changes the result; choosing one for them would
    be the recommendation this system does not make.
    """
    if not allocation_rule:
        raise UnpinnedVest(
            "no allocation rule was stated, so there is nowhere to put the "
            "proceeds. Selling into an unnamed default would pick an "
            "investment on the user's behalf")
    return {allocation_rule: proceeds}


def compute_employer_concentration(holdings: Mapping[str, float],
                                   prices: Mapping[str, float],
                                   employer_ticker: str) -> Dict[str, float]:
    """How much of the portfolio is the employer.

    The number a vesting plan exists to move, and the one a rising employer
    stock quietly raises while every other figure looks healthy.
    """
    values = {ticker: shares * float(prices.get(ticker, 0.0))
              for ticker, shares in holdings.items()}
    total = sum(values.values())
    employer = values.get(employer_ticker, 0.0)
    return {"employer_value": employer, "portfolio_value": total,
            "employer_fraction": (employer / total) if total > 0 else 0.0}


#: Mechanisms that exist as callables in this module. The realization verifier
#: resolves every name here, so the tuple cannot claim one into existence.
IMPLEMENTED = ("apply_vest_delivery", "apply_share_withholding",
               "apply_supplemental_wage_threshold",
               "next_eligible_disposition_session",
               "allocate_disposition_proceeds",
               "compute_employer_concentration")


@dataclass(frozen=True)
class RSUVestingRuntime(RuntimeArtifact):
    """What a vest does, declared so it can be checked."""

    kind: ClassVar[str] = "rsu_vesting"

    name: str
    version: int
    employer_ticker: str = ""
    withholding_method: WithholdingMethod = WithholdingMethod.UNSPECIFIED
    supplemental_rate: Optional[float] = None
    supplemental_threshold: Optional[float] = None
    high_rate: Optional[float] = None
    models_blackouts: bool = False
    models_disposition: bool = False
    measures_concentration: bool = False
    title: str = ""

    def declared_form(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "version": self.version,
                "employer_ticker": self.employer_ticker,
                "withholding_method": self.withholding_method.value,
                "supplemental_rate": self.supplemental_rate,
                "supplemental_threshold": self.supplemental_threshold,
                "high_rate": self.high_rate,
                "models_blackouts": self.models_blackouts,
                "models_disposition": self.models_disposition,
                "measures_concentration": self.measures_concentration,
                "title": self.title}

    def comparable_form(self) -> Dict[str, Any]:
        declared = self.declared_form()
        for prose in ("title", "name", "version"):
            declared.pop(prose, None)
        return declared

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        out = [
            RuntimeAssumption(
                name="in-kind-delivery",
                statement=("Vested shares arrive as shares at the vest price. "
                           "No cash is spent and no order is placed."),
                realized_by="apply_vest_delivery",
                risk=("Modelled as a purchase, the vest acquires a session of "
                      "slippage and credits the plan with a trade nobody made."),
            ),
        ]
        if self.withholding_method is not WithholdingMethod.UNSPECIFIED:
            out.append(RuntimeAssumption(
                name="share-withholding",
                statement=(f"Withholding is performed by "
                           f"{self.withholding_method.value.lower().replace('_', ' ')}; "
                           "withheld shares never enter the portfolio."),
                realized_by="apply_share_withholding",
                risk=("Withheld shares granted and then sold would show a "
                      "holding that never existed and a trade that never "
                      "happened."),
            ))
        if self.supplemental_threshold is not None:
            out.append(RuntimeAssumption(
                name="supplemental-threshold-split",
                statement=(f"Vest value above ${self.supplemental_threshold:,.0f} "
                           f"cumulative in a year withholds at "
                           f"{(self.high_rate or 0):.0%} rather than "
                           f"{(self.supplemental_rate or 0):.0%}."),
                realized_by="apply_supplemental_wage_threshold",
                risk=("One rate applied to a straddling vest is wrong in the "
                      "direction that flatters the result."),
            ))
        if self.models_blackouts:
            out.append(RuntimeAssumption(
                name="blackout-deferral",
                statement=("A sale falling inside a blackout window is deferred "
                           "to the first eligible session, not cancelled."),
                realized_by="next_eligible_disposition_session",
                risk=("A dropped sale silently converts a diversification plan "
                      "into a hold."),
            ))
        if self.models_disposition:
            out.append(RuntimeAssumption(
                name="proceeds-allocation",
                statement=("Sale proceeds follow the stated allocation rule."),
                realized_by="allocate_disposition_proceeds",
            ))
        if self.measures_concentration:
            out.append(RuntimeAssumption(
                name="employer-concentration",
                statement=("The employer's share of portfolio value is measured "
                           "at each vest."),
                realized_by="compute_employer_concentration",
            ))
        return tuple(out)

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        """What is not modelled, stated before any figure exists.

        Every entry is something a reader could reasonably assume is included.
        """
        return (
            RuntimeLimitation(
                name="withholding-is-not-tax",
                statement=("Withholding is a statutory remittance rate, not a "
                           "marginal rate. Actual tax owed at filing is not "
                           "modelled, and someone whose rate is higher will owe "
                           "more than is withheld here."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                name="no-capital-gains-lots",
                statement=("Delivered shares are not tracked as tax lots, so "
                           "gain treatment on a later sale is not modelled."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                name="no-state-or-local-tax",
                statement="State and local tax are not modelled.",
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                name="no-wash-sale",
                statement=("Wash-sale interactions between a disposition and "
                           "other holdings are not modelled."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                name="no-83b-or-espp",
                statement=("Section 83(b) elections and ESPP interactions are "
                           "not modelled."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                # The withholding itself *is* modelled; what is out of scope is
                # the rounding. Filing this as "withholding not modelled" would
                # make a correct treatment look like a gap, which is the
                # miscategorisation `Exclusion` exists to prevent.
                name="no-whole-share-rounding",
                statement=("Withheld shares are computed as an exact fraction. "
                           "Real plans withhold whole shares and refund or "
                           "carry the remainder in cash, which is not modelled, "
                           "so a delivered count here can differ from a "
                           "statement by less than one share."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
            RuntimeLimitation(
                name="no-plan-documents",
                statement=("Company-specific plan rules, insider-trading "
                           "policy and estate treatment are not modelled and "
                           "no legal advice is given."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
        )


#: US federal supplemental withholding by share withholding. What the RSU
#: template actually performs.
US_SHARE_WITHHOLDING = RSUVestingRuntime(
    name="rsu-vesting/us-share-withholding", version=1,
    withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
    supplemental_rate=0.22, supplemental_threshold=1_000_000.0, high_rate=0.37,
    models_blackouts=True, models_disposition=True, measures_concentration=True,
    title="US federal supplemental withholding, shares withheld at vest",
)
