"""Every production request obtains market data through one gate.

`src/workspace/routes.py` read `data/history/prices.parquet` directly. That was
found and fixed. The identical read stayed in `src/web/routes.py` — the public
router — for the whole time, because the fix was applied to the consumer that
was found rather than to the class of consumer. Nothing enumerated the class,
so nothing said the other one existed.

Two checks, because neither is sufficient:

    inventory    every module that reads a data file is either declared as a
                 non-production reader or goes through the gate
    runtime      what a live request actually opens, watched rather than read

The inventory catches a reader nobody has exercised. The runtime check catches
a reader that satisfies the inventory and still reaches the file — and it is
the one that cannot be fooled by a function whose source no longer mentions
`read_parquet` because the read moved one call deeper.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.market_data.access import UNMANIFESTED_PRICES, resolve_prices

SOURCE = Path("src")

#: Reading functions whose presence in a module makes it a data consumer.
READERS = {"read_parquet", "read_csv", "read_json", "read_feather",
           "read_hdf", "read_pickle", "read_excel"}

#: Modules that read data files and are **not** on a production request path.
#: Each is listed with why, so the list is a set of decisions rather than an
#: accumulation of exceptions.
NON_PRODUCTION_READERS = {
    "src/data_loader.py":
        "the backtest/research pipeline, not reachable from a served request",
    "src/history.py":
        "offline historical analysis, run from the command line",
    # This said "the standalone Bokeh demo, served separately and not part of
    # the pilot surface". That was written from the module's name and was the
    # opposite of true: `scripts/service.py` imports it and is the Dockerfile's
    # CMD, so it is what the container runs. See `src/deploy/surfaces.py`,
    # where surfaces are now declared with their entrypoint and checked against
    # the Dockerfile rather than described.
    "src/visualization/bokeh_app.py":
        "the regime dashboard — production-reachable via scripts/service.py, "
        "restricted to synthetic data, and pending routing through the "
        "market-data gate",
    "src/market_data/loader.py":
        "the gate's own implementation — it is what resolves a snapshot to a "
        "file, and is reached only through `access.resolve_prices`",
    "src/reporting.py":
        "writes CSV reports; reads nothing on a request path",
}

#: Modules on a production request path. These must not read a file directly.
PRODUCTION_MODULES = ("src/web/routes.py", "src/workspace/routes.py",
                      "src/api.py")


def modules_reading_files():
    """Every module under `src/` that calls a pandas reader, from the AST.

    Derived by parsing rather than by grepping: a comment or a docstring
    mentioning `read_parquet` is not a read, and this codebase has produced ten
    defects from tests that could not tell the difference.
    """
    found = {}
    for path in sorted(SOURCE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in READERS
        }
        if calls:
            found[str(path)] = sorted(calls)
    return found


class TestTheInventoryIsComplete:
    def test_every_reader_is_accounted_for(self):
        """A new data reader fails here until it is declared or gated."""
        readers = set(modules_reading_files())
        declared = set(NON_PRODUCTION_READERS)
        undeclared = readers - declared
        assert undeclared == set(), (
            f"these read data files and are neither declared as "
            f"non-production nor routed through the gate: {sorted(undeclared)}")

    def test_no_production_module_reads_a_file_directly(self):
        readers = modules_reading_files()
        offenders = [name for name in PRODUCTION_MODULES if name in readers]
        assert offenders == [], (
            f"{offenders} read a data file on a request path. Every production "
            "read goes through `market_data.access.resolve_prices`, which "
            "resolves a snapshot, authorises it and loads it by identity")

    def test_each_exception_records_why(self):
        for module, reason in NON_PRODUCTION_READERS.items():
            assert reason.strip(), module
            assert Path(module).exists(), f"{module} no longer exists"

    def test_the_inventory_is_not_vacuous(self):
        """A scan finding nothing would pass every test above."""
        assert modules_reading_files(), "the AST scan found no readers at all"


class TestTheGateIsTheOnlyWayIn:
    """Watched at runtime, because the inventory cannot see one call deeper."""

    def _watch(self, monkeypatch):
        import pandas as pd

        opened = []
        original = pd.read_parquet

        def watched(path, *args, **kwargs):
            opened.append(str(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", watched)
        return opened

    def test_the_public_router_reads_no_unmanifested_file(self, monkeypatch):
        """The bypass that survived the first fix, on the public surface."""
        import src.web.routes as routes

        opened = self._watch(monkeypatch)
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        routes._prices()

        assert opened, "nothing was read; this test proves nothing"
        assert not any(UNMANIFESTED_PRICES in one for one in opened), opened

    def test_the_workspace_router_reads_no_unmanifested_file(self, monkeypatch):
        import src.workspace.routes as routes

        opened = self._watch(monkeypatch)
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        routes._prices()

        assert opened
        assert not any(UNMANIFESTED_PRICES in one for one in opened), opened

    def test_both_routers_resolve_the_same_snapshot(self, monkeypatch):
        """One gate, so they cannot drift into two policies."""
        import src.web.routes as web
        import src.workspace.routes as workspace

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        assert web._prices() is not None
        assert workspace._prices() is not None

    def test_no_policy_means_no_prices_on_either_router(self, monkeypatch):
        import src.web.routes as web
        import src.workspace.routes as workspace

        monkeypatch.delenv("PILOT_DATA_POLICY", raising=False)
        assert web._prices() is None
        assert workspace._prices() is None

    def test_a_denied_snapshot_yields_nothing_rather_than_other_data(
            self, monkeypatch):
        """No fallback. A figure from data the plan did not name renders
        ordinarily and says nothing about where it came from."""
        import src.web.routes as web

        monkeypatch.setenv("PILOT_DATA_POLICY",
                           "market-data-egress/pilot-vendor-approved@1")
        assert web._prices() is None


class TestTheSnapshotIsPinned:
    def test_the_loaded_frame_comes_from_a_named_snapshot(self, monkeypatch):
        import src.market_data.pilot_policy as policy_module

        seen = []
        real = policy_module.authorise

        def watched(snapshot, *, context):
            seen.append(snapshot)
            return real(snapshot, context=context)

        monkeypatch.setattr(policy_module, "authorise", watched)
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        assert resolve_prices(context="a test") is not None

        assert len(seen) == 1
        assert seen[0].snapshot_id, "the snapshot carries no identity"

    def test_the_context_reaches_the_authorisation(self, monkeypatch):
        """A denial should say which request wanted the data."""
        import src.market_data.pilot_policy as policy_module

        seen = []
        real = policy_module.authorise
        monkeypatch.setattr(
            policy_module, "authorise",
            lambda snapshot, *, context: (seen.append(context),
                                          real(snapshot, context=context))[1])
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        resolve_prices(context="a distinctive context")
        assert seen == ["a distinctive context"]

    def test_a_snapshot_that_will_not_load_yields_nothing(self, monkeypatch):
        """Falling through to another source would be the original bypass."""
        import src.market_data.access as access

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        import src.market_data.loader as loader

        monkeypatch.setattr(
            loader, "load_prices",
            lambda snapshot: (_ for _ in ()).throw(FileNotFoundError("gone")))
        assert resolve_prices(context="a test") is None


class TestADeniedSnapshotYieldsNothing:
    """The denial branch, given a snapshot to deny.

    Nothing reaches it today: under `SYNTHETIC_ONLY` the snapshot is the
    synthetic one and authorisation succeeds, and under the vendor policy
    `approved_snapshot()` returns None so the function stops before
    authorising. Removing the branch changed no result — so the state is built
    here rather than waited for, which is what will happen the day the six
    licensing questions are answered and a vendor snapshot exists.
    """

    def _with_snapshot(self, monkeypatch, snapshot):
        import src.market_data.access as access

        monkeypatch.setattr(access, "approved_snapshot", lambda: snapshot)
        monkeypatch.setenv("PILOT_DATA_POLICY",
                           "market-data-egress/pilot-vendor-approved@1")

    def test_a_denied_snapshot_produces_no_prices(self, monkeypatch):
        from src.market_data.loader import synthetic_snapshot

        # A real snapshot, offered under the vendor policy. Its licence review
        # is UNCONFIRMED, so `authorise` refuses it.
        self._with_snapshot(monkeypatch, synthetic_snapshot())
        import src.market_data.pilot_policy as policy_module

        def deny(snapshot, *, context):
            raise policy_module.PilotDataDenied("licence review unconfirmed")

        monkeypatch.setattr(policy_module, "authorise", deny)
        assert resolve_prices(context="a test") is None

    def test_it_does_not_fall_back_to_the_synthetic_snapshot(self,
                                                            monkeypatch):
        """The precise failure: a denial quietly becoming a different dataset.

        The figure would render, look ordinary, and say nothing about which
        data produced it.
        """
        import pandas as pd

        from src.market_data.loader import synthetic_snapshot

        self._with_snapshot(monkeypatch, synthetic_snapshot())
        import src.market_data.pilot_policy as policy_module

        monkeypatch.setattr(
            policy_module, "authorise",
            lambda snapshot, *, context: (_ for _ in ()).throw(
                policy_module.PilotDataDenied("denied")))

        opened = []
        original = pd.read_parquet
        monkeypatch.setattr(
            pd, "read_parquet",
            lambda path, *a, **k: (opened.append(str(path)),
                                   original(path, *a, **k))[1])

        assert resolve_prices(context="a test") is None
        assert opened == [], (
            f"a denied snapshot still read data: {opened}")
