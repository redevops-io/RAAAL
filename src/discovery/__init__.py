"""Discovery: turning language into a verified statement of what was meant.

Written to move to `agentic_os/discovery/`, which is where the architecture
puts it — beside `agentic_os/mission/`, sharing the event store and the
human-task machinery and sharing none of the authority. `reader.py` and
`shadow.py` import nothing from Quantify and are ready to move as they are;
`schema.py` and `readers_quantify.py` are the domain half and stay here.

It lives here for now because this is where the comparator lives: Phase 3
measures the new reader against the old one, and the old one is Quantify's
compiler.
"""
from .reader import Dimension, DiscoveryReader, Reading, ReadingSet, Schema  # noqa: F401
from .schema import QUANTIFY_SCHEMA  # noqa: F401
from .shadow import (  # noqa: F401
    AGREED,
    CONTESTED,
    ONE_SIDED,
    UNREAD,
    Comparison,
    FieldComparison,
    compare,
    evidence_for,
)
