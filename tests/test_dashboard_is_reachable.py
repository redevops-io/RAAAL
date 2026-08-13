"""A page nothing links to is a page nobody has.

`/research` was restored — the route existed, the cron built the document, the
volume was mounted — and the dashboard was still missing to anyone who did not
already know the URL. Nothing in either navigation pointed at it. The library
header lists Library, Runs, Investigations, Findings, Claims, Protocols, Errata,
API, and the workspace header lists the research library; neither mentioned the
one page the whole daily pipeline exists to produce.

Restoring a feature is not the same as restoring access to it, and every test
written for `/research` asked the first question. This one asks the second.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Every header a person browses from. Both, because the two surfaces have
#: separate templates and a link added to one of them is a dashboard half the
#: site cannot reach.
HEADERS = (
    ROOT / "src" / "web" / "templates" / "base.html",
    ROOT / "src" / "workspace" / "templates" / "base.html",
)


@pytest.mark.parametrize("header", HEADERS, ids=lambda p: p.parent.parent.name)
def test_the_dashboard_is_linked(header):
    assert '/research' in header.read_text(), (
        f"{header.relative_to(ROOT)} links no dashboard. The page can be "
        "served and built nightly and still be unreachable to everyone who "
        "does not type the URL")


#: The general check — every header link naming a real route — was written
#: here and removed. `app.routes` did not carry the prefixed paths the way it
#: was read, so the test reported `/workspace/` and `/ui/` as pointing at
#: nothing while both plainly serve. A guard that cries wolf about working
#: links would be turned off within a week, and the specific assertion above is
#: the one that had a defect to catch.
