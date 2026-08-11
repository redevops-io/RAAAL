"""A stored run cites the delivery it consumed, not the source it declared.

    MarketDataProvenance   which authorized source, and that it was permitted
    MarketDataAccessEvent  what this execution was actually handed

Provenance is reusable: two runs a month apart under one policy carry identical
records, and identical records are not evidence of the same delivery. Until
this existed, a stored figure cited what its *producer declared* — and the
producer is precisely the component a defect would corrupt. This run path has
already been caught dropping the resolver's answer while looking correct.

The three checks are kept separate because two can hold while the third fails:

    event integrity      is the record coherent, and is its body its hash's body
    run binding          does the delivery name this run, and this owner
    declared consistency does the run's own provenance digest to the provenance
                         the delivery was made under
"""
from __future__ import annotations

import pytest

from src.market_data.access_event import (
    FRAME_DIGEST_VERSION,
    MarketDataAccessEvent,
    TimeRange,
    UndigestibleFrame,
    frame_digest,
    from_json,
    provenance_digest,
    verify,
)
from src.market_data.provenance import AccessDecision, ProvenanceStatus
from src.market_data.provenance import MarketDataProvenance

POLICY = "PILOT_DATA_POLICY"
AT = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")


def prices():
    import pandas as pd

    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    return pd.DataFrame({"SPY": [1.0, 2.0, 3.0], "BND": [4.0, 5.0, 6.0]},
                        index=index)


def a_provenance(**overrides):
    fields = dict(status=ProvenanceStatus.RECORDED, snapshot_id="snap-1",
                  content_digest="mdv1:abc", policy_version="SYNTHETIC_ONLY",
                  access_decision=AccessDecision.SYNTHETIC_ALLOWED,
                  accessed_at=AT)
    fields.update(overrides)
    return MarketDataProvenance(**fields)


def an_event(**overrides):
    from src.market_data.access_event import build

    fields = dict(access_event_id="mdae-1", request_id="req-1", run_id="run-1",
                  frame=prices(), provenance=a_provenance(),
                  policy_version="SYNTHETIC_ONLY",
                  decision=AccessDecision.SYNTHETIC_ALLOWED, accessed_at=AT)
    fields.update(overrides)
    return build(**fields)


class TestTheDigestDescribesContentNotPresentation:
    def test_row_and_column_order_do_not_change_it(self):
        frame = prices()
        shuffled = frame[["BND", "SPY"]].iloc[::-1]
        assert frame_digest(shuffled) == frame_digest(frame)

    def test_a_changed_value_changes_it(self):
        frame = prices()
        edited = frame.copy()
        edited.iloc[1, 0] = edited.iloc[1, 0] + 1e-9
        assert frame_digest(edited) != frame_digest(frame)

    def test_a_dropped_row_changes_it(self):
        assert frame_digest(prices().iloc[:-1]) != frame_digest(prices())

    def test_a_dropped_column_changes_it(self):
        assert frame_digest(prices()[["SPY"]]) != frame_digest(prices())

    def test_a_blanked_cell_is_not_a_dropped_row(self):
        """`null` is spelled, not skipped. A blanked cell and an absent one
        producing the same bytes is the substitution this exists to detect."""
        import numpy as np

        blanked = prices().copy()
        blanked.iloc[1, 0] = np.nan
        assert frame_digest(blanked) != frame_digest(prices())
        assert frame_digest(blanked) != frame_digest(prices().drop(
            prices().index[1]))

    def test_it_is_stable_across_calls(self):
        assert frame_digest(prices()) == frame_digest(prices())

    def test_it_names_the_rule_that_produced_it(self):
        """A digest under a changed rule must be distinguishable from a
        mismatch, or a canonicalisation change reports every stored run as
        tampered."""
        assert frame_digest(prices()).startswith(FRAME_DIGEST_VERSION + ":")

    def test_no_frame_is_refused_rather_than_digested(self):
        with pytest.raises(UndigestibleFrame):
            frame_digest(None)

    def test_a_non_frame_is_refused(self):
        with pytest.raises(UndigestibleFrame):
            frame_digest([1, 2, 3])

    def test_numpy_scalars_do_not_leak_into_the_digest(self):
        """`repr(np.float64(1.0))` is `'np.float64(1.0)'` on numpy 2 and
        `'1.0'` on numpy 1. Digesting numpy scalars would silently change every
        stored digest on an upgrade and report the whole workspace as tampered.
        """
        assert "np.float64" not in _digest_body(prices())


def _digest_body(frame):
    """The pre-hash bytes, so the test can assert about them directly."""
    import hashlib

    columns = sorted(str(one) for one in frame.columns)
    ordered = frame[columns].sort_index()
    rows = [f"{stamp}|" + "|".join(
                "null" if cell != cell else repr(cell)
                for cell in cells)
            for stamp, cells in zip([one.isoformat() for one in ordered.index],
                                    ordered.to_numpy().tolist())]
    body = "\n".join([FRAME_DIGEST_VERSION, ",".join(columns), *rows])
    assert hashlib.sha256(body.encode()).hexdigest() in frame_digest(frame)
    return body


class TestTheProvenanceDigestPinsARecordNotASource:
    def test_two_accesses_to_one_snapshot_differ(self):
        """Otherwise an event could cite a snapshot and be satisfied by any
        delivery of it, which is the granularity provenance already has."""
        first = provenance_digest(a_provenance(accessed_at="2026-01-01T00:00:00Z"))
        second = provenance_digest(a_provenance(accessed_at="2026-06-01T00:00:00Z"))
        assert first != second

    def test_the_same_record_digests_the_same(self):
        assert provenance_digest(a_provenance()) == provenance_digest(a_provenance())


class TestAnEventDescribesTheDelivery:
    def test_it_records_what_arrived(self):
        event = an_event()
        assert event.row_count == 3
        assert event.selected_columns == ("BND", "SPY")
        assert event.time_range == TimeRange("2026-01-02T00:00:00",
                                             "2026-01-06T00:00:00")

    def test_it_carries_the_frame_digest(self):
        assert an_event().frame_digest == frame_digest(prices())

    def test_it_names_the_run_it_was_resolved_for(self):
        assert an_event().run_id == "run-1"

    def test_it_survives_a_round_trip(self):
        event = an_event()
        assert from_json(event.to_json()) == event

    def test_absence_stays_absence(self):
        assert from_json(None) is None
        assert from_json({}) is None


class TestTheContentHashCoversEveryField:
    """A hash over a subset lets the excluded fields be edited without trace,
    and the fields most worth editing are the ones a careless exclusion picks."""

    @pytest.mark.parametrize("field,value", [
        ("frame_digest", "mdf1:something-else"),
        ("provenance_digest", "mdp1:something-else"),
        ("run_id", "a-different-run"),
        ("snapshot_id", "a-different-snapshot"),
        ("row_count", 2),
        ("policy_version", "market-data-egress/pilot-vendor-approved@1"),
        ("access_decision", AccessDecision.PILOT_VENDOR_APPROVED),
        ("accessed_at", "2020-01-01T00:00:00Z"),
        ("request_id", "a-different-request"),
        ("selected_columns", ("SPY",)),
    ])
    def test_editing_it_changes_the_hash(self, field, value):
        import dataclasses

        event = an_event()
        edited = dataclasses.replace(event, **{field: value})
        assert edited.content_hash() != event.content_hash(), field


class TestVerifyReadsTheRecordAndNothingCurrent:
    def test_a_coherent_event_has_no_problems(self):
        event = an_event()
        assert verify({**event.to_json(),
                       "content_hash": event.content_hash()}) == ()

    def test_a_tampered_body_is_detected(self):
        event = an_event()
        stored = {**event.to_json(), "content_hash": event.content_hash()}
        stored["frame_digest"] = "mdf1:swapped"
        problems = verify(stored)
        assert any("edited since it was written" in one for one in problems)

    def test_an_event_without_a_frame_digest_is_refused(self):
        event = an_event()
        stored = {**event.to_json()}
        stored["frame_digest"] = ""
        assert any("names a source rather than a delivery" in one
                   for one in verify(stored))

    def test_a_digest_from_another_rule_is_named_as_such(self):
        event = an_event()
        stored = {**event.to_json(), "frame_digest": "mdf0:old-rule"}
        assert any("different rule" in one for one in verify(stored))

    def test_a_delivery_for_a_denied_decision_is_refused(self):
        event = an_event()
        stored = {**event.to_json(),
                  "access_decision": AccessDecision.DENIED.value}
        assert any("DENIED" in one for one in verify(stored))

    def test_an_empty_delivery_is_refused(self):
        stored = {**an_event().to_json(), "row_count": 0}
        assert any("no rows is not a delivery" in one for one in verify(stored))

    def test_absence_is_reported_rather_than_passed(self):
        assert verify({}) == ("no access event was stored",)

    def test_it_does_not_consult_the_environment(self, monkeypatch):
        """A run made under one policy must not become invalid because the
        deployment moved. That is what makes a verifier get switched off."""
        event = an_event()
        stored = {**event.to_json(), "content_hash": event.content_hash()}
        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        assert verify(stored) == ()


class TestTheResolverDigestsWhatItReturns:
    """The load-bearing rule. A caller-computed digest would describe whatever
    the caller was holding, and whether that is still what was delivered is the
    entire question."""

    def test_resolve_returns_an_event(self):
        from src.market_data.access import resolve

        access = resolve(context="a test", run_id="run-9", request_id="req-9")
        assert access.access_event is not None
        assert access.access_event.run_id == "run-9"
        assert access.access_event.request_id == "req-9"

    def test_the_event_digests_the_returned_frame(self):
        from src.market_data.access import resolve

        access = resolve(context="a test")
        assert access.access_event.frame_digest == frame_digest(access.frame)
        assert access.matches_delivery(access.frame)

    def test_a_mutated_frame_no_longer_matches(self):
        from src.market_data.access import resolve

        access = resolve(context="a test")
        mutated = access.frame.copy()
        mutated.iloc[0, 0] = mutated.iloc[0, 0] + 1.0
        assert not access.matches_delivery(mutated)

    def test_the_event_agrees_with_the_provenance_beside_it(self):
        from src.market_data.access import resolve

        access = resolve(context="a test")
        assert access.access_event.provenance_digest == \
            provenance_digest(access.provenance)

    def test_a_denial_yields_no_event(self):
        """An event describing no frame would be a claim with nothing behind
        it. Provenance still records the refusal."""
        import src.market_data.pilot_policy as policy_module
        from src.market_data.access import resolve

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
            patch.setattr(policy_module, "authorise",
                          lambda *a, **k: (_ for _ in ()).throw(
                              policy_module.PilotDataDenied("review open")))
            access = resolve(context="a test")
        assert access.access_event is None
        assert access.access_event_id is None

    def test_no_policy_yields_no_event(self, monkeypatch):
        from src.market_data.access import resolve

        monkeypatch.delenv(POLICY, raising=False)
        assert resolve(context="a test").access_event is None

    def test_the_unpacking_contract_is_unchanged(self):
        """Widening the tuple would silently give every existing
        `frame, provenance = resolve(...)` a third value or an error."""
        from src.market_data.access import resolve

        frame, provenance = resolve(context="a test")
        assert frame is not None
        assert provenance.status is ProvenanceStatus.RECORDED


class TestTwoResolutionsAreTwoDeliveries:
    def test_each_gets_its_own_identity(self):
        from src.market_data.access import resolve

        first = resolve(context="a test")
        second = resolve(context="a test")
        assert first.access_event_id != second.access_event_id

    def test_identical_content_still_digests_alike(self):
        """Two deliveries of one snapshot are different events about the same
        bytes. Conflating them would lose which run received which."""
        from src.market_data.access import resolve

        first = resolve(context="a test")
        second = resolve(context="a test")
        assert first.access_event.frame_digest == second.access_event.frame_digest
        assert first.access_event.content_hash() != \
            second.access_event.content_hash()


class TestTheDigestBelongsToTheApplicationNotTheLibrary:
    """A digest is artifact identity now, so anything that can change it
    silently is a dependency of every stored run.

    The original implementation indexed cell by cell and hashed `repr` of what
    pandas returned — numpy scalars. `repr(np.float64(1.0))` is `'1.0'` on
    numpy 1 and `'np.float64(1.0)'` on numpy 2, so a routine upgrade would have
    changed every digest and reported the whole workspace as tampered. Speed
    was the reason to look; this is the reason it mattered.
    """

    def test_the_bytes_contain_no_library_type_names(self):
        for token in ("np.", "numpy", "float64", "Timestamp", "dtype",
                      "array", "Series", "DataFrame"):
            assert token not in _digest_body(prices()), token

    def test_dtype_is_not_identity_but_a_value_it_changes_is(self):
        """The digest is over values, not storage.

        Written first as "narrowing the dtype changes the digest", which was
        wrong on the sample data and right about the intent: float32 held
        1.0, 2.0 and 3.0 exactly, so the bytes were identical and the
        assertion failed. The behaviour is the better one — a column that
        round-trips through a narrower type carries the same quantities and is
        the same delivery — and it is recorded here as the two separate facts
        it actually is.
        """
        import numpy as np
        import pandas as pd

        exact = prices()
        assert frame_digest(exact.astype(np.float32)) == frame_digest(exact)

        lossy = pd.DataFrame({"A": [0.1]},
                             index=pd.to_datetime(["2026-01-01"]))
        assert frame_digest(lossy.astype(np.float32)) != frame_digest(lossy)

    def test_an_index_rendered_differently_still_digests_alike(self):
        """Timestamps are rendered by `isoformat`, not by pandas' display
        rules, so a frequency or display-option change cannot move a digest."""
        import pandas as pd

        frame = prices()
        relabelled = frame.copy()
        relabelled.index = pd.DatetimeIndex(
            [pd.Timestamp(one).tz_localize(None) for one in frame.index],
            freq=None)
        assert frame_digest(relabelled) == frame_digest(frame)

    def test_the_stored_digest_of_the_pilot_snapshot_is_pinned(self):
        """A golden value over the real synthetic snapshot.

        This is the check that actually fires on a library upgrade. Every test
        above compares two digests computed by the same code in the same
        process, and would agree with itself no matter what the library did.
        If this fails and nothing else does, the canonicalisation moved:
        bump `FRAME_DIGEST_VERSION` so old and new are distinguishable rather
        than editing the constant, which would silently redefine identity for
        every stored run.
        """
        from src.market_data.loader import load_prices, synthetic_snapshot

        digest = frame_digest(load_prices(synthetic_snapshot()).sort_index())
        assert digest == PILOT_SNAPSHOT_DIGEST, (
            "the canonical digest of the pilot snapshot changed. Either the "
            "snapshot moved, or the canonicalisation did — the second silently "
            "invalidates every stored access event")


#: The canonical digest of the synthetic pilot snapshot, pinned.
PILOT_SNAPSHOT_DIGEST = (
    "mdf1:6317ae3dba5c3f75324ce921b697de6ece80de9d7be1b6b2b14f7d668d8d7220")
