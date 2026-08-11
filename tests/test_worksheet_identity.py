"""Worksheet identity, and the tenant existence oracle it used to be.

The `worksheet` table was keyed on (worksheet_id, revision) with no owner, and
ids were `ws-{plan_id}` — derived from a name the user chose. Reads were
correctly owner-scoped, so nothing leaked on the way out. The leak was on the
way in: a second owner creating a worksheet whose id another tenant already held
was refused, and the refusal answered a question the requester was not entitled
to ask.

Two changes, and the second is what removes the class of problem rather than
this instance of it: owner is part of the key, and the identity is opaque and
server-generated. A human-readable name is a field, not an identity.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.workspace.generate import new_worksheet_id
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

MINE, THEIRS = "owner-a", "owner-b"


def worksheet(identifier, owner, scenario_ref="plan-1"):
    return create(worksheet_id=identifier, owner_id=owner,
                  scenario_ref=scenario_ref, primary_run_ref="run-0",
                  title="My retirement plan", created_at="t0")


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


class TestNoTenantExistenceOracle:

    def test_two_owners_can_hold_the_same_worksheet_id(self, store):
        """The write that used to be refused. The refusal was the oracle."""
        store.save_worksheet(worksheet("ws-shared", MINE))
        store.save_worksheet(worksheet("ws-shared", THEIRS))

        assert store.get_worksheet("ws-shared", MINE) is not None
        assert store.get_worksheet("ws-shared", THEIRS) is not None

    def test_neither_owner_can_read_the_others(self, store):
        store.save_worksheet(worksheet("ws-shared", MINE, scenario_ref="mine"))
        store.save_worksheet(worksheet("ws-shared", THEIRS, scenario_ref="theirs"))

        assert store.get_worksheet("ws-shared", MINE)["payload"]["scenario_ref"] \
            == "mine"
        assert store.get_worksheet("ws-shared", THEIRS)["payload"]["scenario_ref"] \
            == "theirs"

    def test_one_owner_cannot_overwrite_the_others_revision(self, store):
        store.save_worksheet(worksheet("ws-shared", MINE, scenario_ref="mine"))
        store.save_worksheet(worksheet("ws-shared", THEIRS, scenario_ref="theirs"))

        assert store.get_worksheet("ws-shared", MINE)["payload"]["scenario_ref"] \
            == "mine"

    def test_an_owner_still_cannot_rewrite_their_own_revision(self, store):
        """The protection that had to survive: revisions are never edited."""
        from src.workspace.store import NotSaveable

        store.save_worksheet(worksheet("ws-1", MINE, scenario_ref="mine"))
        with pytest.raises(NotSaveable):
            store.save_worksheet(worksheet("ws-1", MINE, scenario_ref="changed"))

    def test_the_same_revision_is_still_idempotent(self, store):
        store.save_worksheet(worksheet("ws-1", MINE))
        store.save_worksheet(worksheet("ws-1", MINE))
        assert len(store.worksheet_revisions("ws-1", MINE)) == 1


class TestIdentityCarriesNoMeaning:

    def test_an_id_is_opaque_and_not_derived_from_the_plan(self):
        """It was `ws-{plan_id}`. Knowing a plan name was enough to name
        another tenant's worksheet."""
        assert "plan" not in new_worksheet_id()
        assert len(new_worksheet_id()) > len("ws-plan-1")

    def test_two_worksheets_never_share_an_id(self):
        assert len({new_worksheet_id() for _ in range(1000)}) == 1000

    def test_the_readable_name_survives_as_a_field(self, store):
        """Removing meaning from the identity must not remove it from the
        product. The title is what a user reads."""
        store.save_worksheet(worksheet(new_worksheet_id(), MINE))
        record = store.worksheet_for_scenario("plan-1", MINE)
        assert record["payload"]["title"] == "My retirement plan"


class TestLookupByWhatItCites:

    def test_an_owners_worksheet_is_found_by_its_scenario(self, store):
        identifier = new_worksheet_id()
        store.save_worksheet(worksheet(identifier, MINE))
        assert store.worksheet_for_scenario("plan-1", MINE)["worksheet_id"] \
            == identifier

    def test_another_owners_worksheet_is_not_found(self, store):
        """Two tenants may hold worksheets for identically-named scenarios and
        neither may observe the other's."""
        store.save_worksheet(worksheet(new_worksheet_id(), THEIRS))
        assert store.worksheet_for_scenario("plan-1", MINE) is None

    def test_an_unknown_scenario_returns_nothing(self, store):
        assert store.worksheet_for_scenario("no-such-plan", MINE) is None


class TestTheMigration:

    def test_an_old_database_is_rekeyed_and_keeps_its_rows(self, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript("""
        CREATE TABLE worksheet (
            worksheet_id TEXT NOT NULL, revision INTEGER NOT NULL,
            owner TEXT NOT NULL, payload TEXT NOT NULL,
            canonical_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (worksheet_id, revision));
        INSERT INTO worksheet VALUES
          ('ws-plan-1',1,'pilot','{"scenario_ref":"plan-1"}','h','t0');
        """)
        conn.commit()
        conn.close()

        store = WorkspaceStore(path)
        key = [c[1] for c in sorted(
            (c for c in sqlite3.connect(path).execute(
                "PRAGMA table_info(worksheet)") if c[5]), key=lambda c: c[5])]

        assert key == ["owner", "worksheet_id", "revision"]
        assert store.get_worksheet("ws-plan-1", "pilot") is not None

    def test_the_migrated_database_no_longer_refuses_the_second_owner(
            self, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript("""
        CREATE TABLE worksheet (
            worksheet_id TEXT NOT NULL, revision INTEGER NOT NULL,
            owner TEXT NOT NULL, payload TEXT NOT NULL,
            canonical_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (worksheet_id, revision));
        INSERT INTO worksheet VALUES
          ('ws-plan-1',1,'owner-a','{"scenario_ref":"plan-1"}','h','t0');
        """)
        conn.commit()
        conn.close()

        store = WorkspaceStore(path)
        store.save_worksheet(worksheet("ws-plan-1", THEIRS))
        assert store.get_worksheet("ws-plan-1", THEIRS) is not None
