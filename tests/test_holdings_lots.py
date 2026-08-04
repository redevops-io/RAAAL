"""A position is a projection, not a stored number.

The permanent fixture is real: NVDA split 10-for-1 on 2024-06-10, and 4-for-1
before that on 2021-07-20. Ten shares bought at $1,209.98 on 2024-06-06 are a
hundred shares today, the statement still says ten, and both are true.

Valuing that holding against a split-adjusted series without converting gives
$1,210 for a position worth $12,100 — off by the split ratio, and entirely
plausible on a page.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from src.holdings.actions import (
    ActionKind,
    ActionSnapshot,
    CorporateAction,
    UnsupportedCorporateAction,
    from_split_table,
)
from src.holdings.lots import (
    AcquisitionKind,
    AcquisitionLot,
    DispositionLotAllocation,
    LotLedgerError,
    OverAllocated,
    remaining_quantity,
    value_position,
)
from src.holdings.prices import PricePurpose, WrongPriceSeries, market, total_return

D = Decimal

NVDA_SPLITS = ActionSnapshot("nvda@2026-08", (
    CorporateAction("NVDA", ActionKind.SPLIT, dt.date(2021, 7, 20), D("4")),
    CorporateAction("NVDA", ActionKind.SPLIT, dt.date(2024, 6, 10), D("10")),
))

# Yahoo's split-adjusted close. As traded on 2024-06-06 it printed 1209.98.
NVDA_MARKET = market("prices-catalog-20260804", {
    "NVDA": {dt.date(2024, 6, 6): D("120.998"), dt.date(2026, 8, 3): D("180.00")},
})


def lot(quantity="10", price="1209.98", acquired=dt.date(2024, 6, 6),
        lot_id="lot-1", instrument="NVDA", cost=None):
    quantity = D(quantity)
    price = D(price)
    return AcquisitionLot(
        owner="pilot", account_id="taxable", lot_id=lot_id,
        instrument_id=instrument, acquired_at=acquired,
        as_traded_quantity=quantity, as_traded_unit_price=price,
        acquisition_cost=D(cost) if cost else quantity * price,
        acquisition_kind=AcquisitionKind.PURCHASE,
        source_ref="statement-2024-06", corporate_action_snapshot=NVDA_SPLITS.snapshot_id)


class TestTheStatementIsNeverRewritten:
    def test_the_lot_still_says_what_the_user_bought(self):
        view = lot().statement_view()
        assert view["quantity"] == D("10")
        assert view["unit_price"] == D("1209.98")

    def test_there_is_no_stored_current_quantity(self):
        """A second authority would disagree with the action snapshot silently."""
        assert not hasattr(lot(), "current_quantity")
        assert "current_quantity" not in AcquisitionLot.__dataclass_fields__


class TestCurrentUnitsAreDerived:
    def test_ten_shares_before_the_split_are_a_hundred_today(self):
        resolved = lot().current(NVDA_SPLITS)
        assert resolved.current_quantity == D("100")
        assert resolved.as_traded_quantity == D("10")

    def test_the_resolution_names_the_actions_and_the_snapshot(self):
        """A number alone cannot be reconciled against a statement."""
        resolved = lot().current(NVDA_SPLITS)
        assert [a.effective_on for a in resolved.actions_applied] == [dt.date(2024, 6, 10)]
        assert resolved.snapshot_id == "nvda@2026-08"

    def test_a_lot_acquired_after_every_split_is_unchanged(self):
        resolved = lot(acquired=dt.date(2025, 1, 2)).current(NVDA_SPLITS)
        assert resolved.current_quantity == D("10")
        assert resolved.unchanged

    def test_splits_compound_rather_than_replacing(self):
        resolved = lot(acquired=dt.date(2020, 1, 2)).current(NVDA_SPLITS)
        assert resolved.current_quantity == D("400")  # 4 x 10

    def test_a_split_on_the_acquisition_date_is_not_applied_twice(self):
        """The price that day is already in post-split units, and so is the buy."""
        assert lot(acquired=dt.date(2024, 6, 10)).current(NVDA_SPLITS).current_quantity == D("10")


class TestValueSurvivesTheSplit:
    def test_the_invariant_holds_across_epochs(self):
        one = lot()
        as_traded = one.as_traded_quantity * one.as_traded_unit_price
        current = one.current(NVDA_SPLITS).current_quantity * D("120.998")
        assert as_traded == current == D("12099.80")

    def test_valuation_uses_current_units(self):
        view = value_position([lot()], snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                              as_of=dt.date(2024, 6, 6))
        assert view.current_quantity == D("100")
        assert view.market_value == D("12099.800")


class TestTheWrongSeriesIsRefusedByName:
    """Both mistakes are plausible; neither announces itself."""

    def test_total_return_prices_cannot_value_a_position(self):
        wrong = total_return("prices-catalog-20260804",
                             {"NVDA": {dt.date(2024, 6, 6): D("120.789")}})
        with pytest.raises(WrongPriceSeries) as caught:
            value_position([lot()], snapshot=NVDA_SPLITS, prices=wrong,
                           as_of=dt.date(2024, 6, 6))
        assert caught.value.wanted is PricePurpose.MARKET
        assert "twice" in str(caught.value)

    def test_market_prices_are_accepted(self):
        assert value_position([lot()], snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                              as_of=dt.date(2024, 6, 6)) is not None


class TestTwoLotsStraddlingASplit:
    def test_each_lot_converts_by_its_own_epoch(self):
        before = lot(quantity="10", acquired=dt.date(2024, 6, 6), lot_id="a")
        after = lot(quantity="10", price="121.79",
                    acquired=dt.date(2024, 6, 11), lot_id="b")

        view = value_position([before, after], snapshot=NVDA_SPLITS,
                              prices=NVDA_MARKET, as_of=dt.date(2026, 8, 3))

        # 100 from the pre-split lot, 10 from the post-split one.
        assert view.current_quantity == D("110")

    def test_cost_basis_is_not_converted(self):
        """Money does not split. Only the units do."""
        before = lot(quantity="10", price="1209.98", lot_id="a")
        after = lot(quantity="10", price="121.79",
                    acquired=dt.date(2024, 6, 11), lot_id="b")
        view = value_position([before, after], snapshot=NVDA_SPLITS,
                              prices=NVDA_MARKET, as_of=dt.date(2026, 8, 3))
        assert view.cost_basis == D("12099.80") + D("1217.90")


class TestDispositionsConsumeLots:
    def test_a_partial_sale_leaves_the_rest_of_the_lot(self):
        one = lot()
        sold = DispositionLotAllocation("d-1", "lot-1", D("40"), D("4839.92"), D("7200"))
        assert remaining_quantity(one, NVDA_SPLITS, [sold]) == D("60")

    def test_the_position_reflects_what_is_left(self):
        sold = DispositionLotAllocation("d-1", "lot-1", D("40"), D("4839.92"), D("7200"))
        view = value_position([lot()], snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                              as_of=dt.date(2026, 8, 3), allocations=[sold])
        assert view.current_quantity == D("60")
        assert view.market_value == D("10800.00")

    def test_basis_follows_the_units_still_held(self):
        sold = DispositionLotAllocation("d-1", "lot-1", D("40"), D("4839.92"), D("7200"))
        view = value_position([lot()], snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                              as_of=dt.date(2026, 8, 3), allocations=[sold])
        assert view.cost_basis == D("12099.80") * D("60") / D("100")

    def test_selling_more_than_the_lot_holds_is_refused(self):
        too_much = DispositionLotAllocation("d-1", "lot-1", D("101"), D("0"), D("0"))
        with pytest.raises(OverAllocated):
            remaining_quantity(lot(), NVDA_SPLITS, [too_much])

    def test_realized_gain_is_proceeds_less_basis_and_fees(self):
        sold = DispositionLotAllocation("d-1", "lot-1", D("40"), D("4839.92"),
                                        D("7200"), fees=D("5"))
        assert sold.realized == D("7200") - D("4839.92") - D("5")


class TestAReverseSplitLeavesCashNotShares:
    SNAPSHOT = ActionSnapshot("rev@1", (
        CorporateAction("XYZ", ActionKind.SPLIT, dt.date(2025, 1, 15),
                        D("0.3333333333")),))

    def test_the_fraction_is_paid_out_rather_than_rounded_into_the_position(self):
        one = lot(quantity="10", price="30", acquired=dt.date(2024, 1, 2),
                  instrument="XYZ")
        resolved = one.current(self.SNAPSHOT, whole_shares_only=True,
                               price_on_action=D("90"))
        assert resolved.current_quantity == D("3")
        assert resolved.residual_fraction > 0
        assert resolved.cash_in_lieu > 0

    def test_by_default_the_fraction_is_held(self):
        """Most brokers hold fractional ETF shares; inventing a cash payment
        nobody received would be its own defect."""
        one = lot(quantity="10", price="30", acquired=dt.date(2024, 1, 2),
                  instrument="XYZ")
        resolved = one.current(self.SNAPSHOT)
        assert resolved.cash_in_lieu == 0
        assert resolved.current_quantity != resolved.current_quantity.to_integral_value()


class TestUnmodelledActionsRefuse:
    def test_a_merger_is_named_rather_than_ignored(self):
        snapshot = ActionSnapshot("m@1", (
            CorporateAction("XYZ", ActionKind.MERGER, dt.date(2025, 3, 1)),))
        with pytest.raises(UnsupportedCorporateAction) as caught:
            lot(acquired=dt.date(2024, 1, 2), instrument="XYZ").current(snapshot)
        assert "merger" in str(caught.value)


class TestTheViewCarriesItsPins:
    def test_every_pin_a_comparison_depends_on_is_recorded(self):
        view = value_position([lot()], snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                              as_of=dt.date(2026, 8, 3),
                              lot_selection_policy="specific-identification",
                              distribution_treatment="excluded")
        assert view.price_snapshot == "prices-catalog-20260804"
        assert view.action_snapshot == "nvda@2026-08"
        assert view.price_purpose == "market"
        assert view.lot_selection_policy == "specific-identification"
        assert view.distribution_treatment == "excluded"


class TestTheBridgeFromTheVendorTable:
    def test_a_split_series_becomes_typed_actions(self):
        import pandas as pd

        series = pd.Series([10.0], index=pd.to_datetime(["2024-06-10"]))
        snapshot = from_split_table("NVDA", series, "vendor@1")
        assert len(snapshot.actions) == 1
        assert snapshot.actions[0].kind is ActionKind.SPLIT
        assert lot().current(snapshot).current_quantity == D("100")


class TestAPositionIsOneInstrument:
    def test_mixing_instruments_is_refused(self):
        with pytest.raises(LotLedgerError):
            value_position([lot(instrument="NVDA"), lot(instrument="VTI", lot_id="b")],
                           snapshot=NVDA_SPLITS, prices=NVDA_MARKET,
                           as_of=dt.date(2026, 8, 3))
