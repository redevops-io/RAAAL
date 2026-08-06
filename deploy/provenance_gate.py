"""Deployment gate: structured provenance survives the round trip in production.

Reads the stored row directly. The page is not consulted, because the property
under test is not presentation — a page can render an amendment it reconstructed
from prose and look identical to one that read a structured record.

Run against production after the provenance deploy, and again after the
replacement plan is created through the builder.
"""
import json
import sys

from src.deploy.context import current
from src.workspace.store import WorkspaceStore
from src.mission.spec import provenance_shape_of

OWNER = "pilot"
store = WorkspaceStore(current().database.url)

print("=== physical column type (the defect PostgreSQL caught)")
with store._conn() as conn:
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'plan_migration' ORDER BY column_name").fetchall()
    types = {r["column_name"]: r["data_type"] for r in rows}
print("plan_migration.scenario:", types.get("scenario"))
print("IS_JSONB:", types.get("scenario") == "jsonb")

print("\n=== every plan, by provenance shape")
failures = []
for plan in store.list_plans(OWNER):
    record = store.get_plan(plan["plan_id"], OWNER)
    body = record.get("scenario") or {}
    prov = body.get("provenance") or {}
    shape = provenance_shape_of(prov)
    structured = {
        "amended": len(prov.get("amended") or []),
        "asset_resolutions": len(prov.get("asset_resolutions") or []),
        "excluded": len(prov.get("excluded") or []),
        "time_window": prov.get("time_window") is not None,
    }
    answered_in_prose = [s for s in (prov.get("stated") or [])
                         if "(answered)" in str(s)]
    print(f"{plan['plan_id'][:26]:28} shape={shape:14} {structured}")
    print(f"{'':28} prose answers: {len(answered_in_prose)}")
    if shape == "provenance@2" and answered_in_prose and not structured["amended"]:
        failures.append(
            f"{plan['plan_id']}: current shape, {len(answered_in_prose)} "
            f"answers rendered as prose, none stored as amendments")

print("\n=== verdict")
if failures:
    for one in failures:
        print("FAIL", one)
    sys.exit(1)

# The premise, printed rather than assumed.
#
# The check exempts `@1` bodies, correctly — their answers were never stored
# and that is what the classification says. But a workspace containing only
# `@1` plans therefore passes without the discriminating condition ever being
# evaluated, and "no failures" would read as evidence when nothing was tested.
current = [p for p in store.list_plans(OWNER)
           if provenance_shape_of(
               ((store.get_plan(p["plan_id"], OWNER) or {}).get("scenario")
                or {}).get("provenance") or {}) == "provenance@2"]
if not current:
    print("VACUOUS: no plan on the current shape exists, so the check that "
          "matters — prose answers without structured records — was never "
          "evaluated. This gate becomes meaningful only after a plan is saved "
          "by the new serializer.")
    sys.exit(2)
print(f"{len(current)} plan(s) on the current shape; none carries prose "
      f"answers without the structured record behind them")
