"""The offered catalogue, read from the library rather than the page.

The sweep needs the sentences the menu offers. Taking them from the library
keeps the browser sweep and the fast sweep asking about the same list — a
sweep that scraped the dropdown would silently stop covering an entry the
moment the page failed to render it, which is the case it most needs to catch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def catalogue() -> List[Tuple[str, str]]:
    from src.workspace.strategy_library import LIBRARY

    return [(entry.key, entry.text)
            for group in LIBRARY for entry in group.entries]
