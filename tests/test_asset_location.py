"""Asset location is a relation, and it must not fire on a bare account.

    "keep the REITs in the Roth"        holding -> account. A relation.
    "I have a Roth"                     an account, and nothing placed in it.

The first is the last schema gap the strategy sweep left standing: `account_type`
*was* read from "hold the bonds in the IRA and the stocks in the taxable
account" — it returned TAXABLE — so the family scored as understood while the
mapping, which is the entire request, was gone.

The second is why this needs a negative control. A relation that fired on any
sentence mentioning a Roth would refuse ordinary contribution plans by name, and
a boundary that triggers on the wrong sentences is worse than no boundary: the
refusals read as authoritative and the person is told their plan is unsupported
when it is not.
"""
from __future__ import annotations

import pytest


class TestTheRelationIsDeclaredProperly:
    def test_it_requires_both_ends(self):
        """A holding with no account places nothing; an account with no holding
        is somewhere to put things. Neither alone is asset location."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        spec = {r.kind: r for r in QUANTIFY_SCHEMA.relations}["asset_location"]
        assert set(spec.required_roles) == {"holding", "account"}

    def test_both_ends_repeat(self):
        """"bonds in the IRA and stocks in the taxable account" is two pairs,
        and a relation admitting one of each could only carry half of it."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        spec = {r.kind: r for r in QUANTIFY_SCHEMA.relations}["asset_location"]
        assert {"holding", "account"} <= set(spec.repeatable_roles)

    def test_mission_refuses_it_by_name(self):
        from src.mission.capability import decide

        refusal = decide("asset_location", "REITs -> roth")
        assert refusal is not None
        assert refusal.dimension == "asset_location"
        assert "not modelled" in refusal.detail or refusal.detail

    def test_the_schema_can_say_it_and_the_manifest_cannot_run_it(self):
        """The rule the whole boundary follows, on this one relation."""
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.mission.capability import MANIFEST

        assert "asset_location" in {r.kind for r in QUANTIFY_SCHEMA.relations}
        assert dict(MANIFEST)["asset_location"].support == "NOT_MODELLED"


class TestItDoesNotFireOnABareAccount:
    """The negative control.

    Asserted through the reading rather than by inspecting the prompt, because
    what matters is the relation the reader actually returns.
    """

    @pytest.fixture
    def reader(self):
        from src.discovery.hosted_recording import RecordedHostedReader

        return RecordedHostedReader()

    @pytest.mark.parametrize("text", [
        "I have a Roth",
        "contribute $500 monthly to my Roth IRA",
    ])
    def test_naming_an_account_places_nothing_in_it(self, reader, text):
        from src.discovery.schema import QUANTIFY_SCHEMA

        reading = reader.read(text, QUANTIFY_SCHEMA)
        kinds = {getattr(r, "kind", "") for r in (reading.relations or ())}
        assert "asset_location" not in kinds, (
            f"{text!r} produced an asset_location relation; a bare account "
            "mention would then refuse ordinary contribution plans by name")

    def test_and_an_ordinary_contribution_plan_stays_executable(self, reader):
        """The consequence, end to end. If the relation over-fires, this plan
        is refused for a capability the person never asked for."""
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.witnesses import MODEL_ONLY
        from src.workspace.pilot import read

        reading = read("contribute $500 monthly to my Roth IRA", reader,
                       schema=QUANTIFY_SCHEMA, profile=MODEL_ONLY)
        refused = {getattr(r, "dimension", "") for r in reading.refusals}
        assert "asset_location" not in refused
