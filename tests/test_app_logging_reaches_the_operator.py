"""What the application logs must reach the log an operator reads.

Uvicorn installs its own logging configuration and leaves the root logger
alone, so a module calling `getLogger(__name__)` writes into a logger with no
handler and no level. Nothing fails; the line is simply never emitted.

That was already known for one line — the deployment proof borrows
`uvicorn.error` by hand, with a comment explaining why. Every other module
kept its own logger, including `parse_model`'s

    logger.info("stage1 provider call: model=%s mode=%s", model, mode)

which carries a comment saying it exists so provider calls "can be measured
against a live server rather than a fixture". It was never emitted by one.

The consequence was worse than a missing line. Three reopens of a saved plan
reported zero provider calls, which is the correct answer; a fresh draft that
certainly made a call reported zero too. A measurement that returns the
expected answer for every input is not evidence, and the only reason it was
caught is that the producer was checked before the count was believed.

So the test below is not "the logger has a handler". It is: emit the real
line from the real module and require it to arrive.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def bootstrapped(monkeypatch):
    """The app configured as it is in production, with uvicorn's logger
    carrying a handler this test can read."""
    import src.api as api

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")

    uvicorn = logging.getLogger("uvicorn.error")
    package = logging.getLogger("src")
    before = (list(uvicorn.handlers), list(package.handlers),
              package.level, package.propagate)

    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    uvicorn.handlers = [Collect()]
    package.handlers = []
    try:
        # `create_app`, not `_bootstrap`. The wiring belongs to the production
        # entrypoint — uvicorn is pointed at this function, and it is where
        # the deployment proof is written — and a test that called the other
        # one would exercise nothing and report a handler that was never set.
        api.create_app()
        yield records
    finally:
        uvicorn.handlers, package.handlers = before[0], before[1]
        package.level, package.propagate = before[2], before[3]


class TestThePremise:
    def test_uvicorn_does_not_configure_the_package_logger(self):
        """If it did, none of this would be needed and the test would pass
        for a reason that has nothing to do with the fix."""
        fresh = logging.getLogger("src.some.module.that.does.not.exist")
        assert not fresh.handlers


class TestALineFromADeepModuleArrives:
    def test_the_provider_call_line_is_emitted(self, bootstrapped):
        """The actual instrumentation, from the actual module, not a stand-in
        logger created by this test."""
        logging.getLogger("src.mission.parse_model").info(
            "stage1 provider call: model=%s mode=%s", "test-model", "TEST")
        assert any("stage1 provider call" in one.getMessage()
                   for one in bootstrapped), (
            "the line a live server is supposed to be counted by did not "
            "reach the operator's log")

    def test_a_warning_from_the_same_module_arrives(self, bootstrapped):
        """The fallback line distinguishes a degraded parse from a pinned
        replay. Losing it would let a journey that fell back look like one
        that did not."""
        logging.getLogger("src.mission.parse_model").warning(
            "stage1 fallback to deterministic: %s", "boom")
        assert any("stage1 fallback" in one.getMessage()
                   for one in bootstrapped)

    def test_an_unrelated_module_arrives_too(self, bootstrapped):
        """Attached at the package, so a module does nothing to be heard. A
        per-module fix would pass the two tests above and fail this one."""
        logging.getLogger("src.workspace.routes").info("a routes line")
        assert any("a routes line" in one.getMessage()
                   for one in bootstrapped)

    def test_info_is_not_filtered_out(self, bootstrapped):
        assert logging.getLogger("src.mission.parse_model").isEnabledFor(
            logging.INFO)

    def test_the_self_test_is_emitted_at_startup(self, bootstrapped):
        """An operator can confirm the stream works before trusting a count
        taken from it — without having to make a request that produces one."""
        assert any("instrumentation self-test" in one.getMessage()
                   for one in bootstrapped)

    def test_the_self_test_comes_from_a_deep_module(self, bootstrapped):
        """From `src.mission.parse_model`, not from `src.api`. A line logged
        by the module that does the wiring would prove only that the wiring
        module can log, which was never in doubt."""
        emitted = [one for one in bootstrapped
                   if "instrumentation self-test" in one.getMessage()]
        assert emitted and all(one.name == "src.mission.parse_model"
                               for one in emitted), (
            [one.name for one in emitted])

    def test_the_deployment_proof_still_arrives(self, bootstrapped):
        """It was already working by borrowing uvicorn's logger directly.
        Redirecting the package must not take it away."""
        assert any("deployment proof" in one.getMessage()
                   for one in bootstrapped), (
            "the proof that was working before this change no longer arrives")

    def test_nothing_is_logged_twice(self, bootstrapped):
        """`propagate` is off deliberately: the package's handlers are
        uvicorn's own, so leaving propagation on would deliver every line to
        them a second time through the root."""
        logging.getLogger("src.mission.parse_model").info("exactly once")
        assert sum(1 for one in bootstrapped
                   if "exactly once" in one.getMessage()) == 1
