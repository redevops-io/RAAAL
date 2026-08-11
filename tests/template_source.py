"""Reading a template the way Jinja reads it.

Assertions about what a template *emits* must not match its comments. That
mistake has now been made seven times in this codebase — against "vest" inside
"uninvested", "proceeds" inside a label, "store" inside a docstring,
"recommendation" inside a comment explaining why nothing is recommended, and
the status wording inside a comment explaining that the wording lives
elsewhere.

The rule, in one place so it stops being re-derived:

    assert against what is emitted, never against the file that emits it.
"""
from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path("src/workspace/templates")


def emitted(name: str) -> str:
    """A template's source with Jinja comments removed, as Jinja removes them."""
    return re.sub(r"\{#.*?#\}", "", (TEMPLATES / name).read_text(), flags=re.S)
