"""The market-data wire contract, frozen before the service is written.

    POST /snapshots                     resolve + fetch + store -> hashes
    GET  /snapshots/{hash}/observations immutable bytes
    GET  /descriptors/{hash}            what is believed about those bytes
    GET  /health, /version

**Every failure carries a `kind`, and no two collapse.** An ambiguous symbol, an
untradable instrument, a missing payload and a corrupted one are four different
repairs, and a status code alone cannot hold four meanings. The kind is the
contract; the status is a courtesy to anything that only reads statuses.

**Observations cross as bytes, descriptors as JSON.** The bytes are the thing the
hash addresses, so re-encoding them into a JSON field would put a transport's
choices between the data and its digest — and the digest is the only reason to
trust either.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping

#: The shape of this request/response pair. Separate from the snapshot contract
#: version: a transport can change how it frames a call without the snapshot
#: contract moving, and one number meaning two things is how a version stops
#: being checkable.
MARKET_DATA_CONTRACT_VERSION = "quantify-market-data-contract@1"


class Failure(str, Enum):
    """Why a request could not be answered, specifically enough to act on.

    Named rather than inferred from a status. Four of these arrive as 422 and
    they are not the same problem: a caller retrying an ambiguous symbol will
    retry forever, and one treating a corrupted payload as an unknown one will
    look for a snapshot that is right there.
    """

    AMBIGUOUS_SYMBOL = "AMBIGUOUS_SYMBOL"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    NOT_TRADABLE = "NOT_TRADABLE"
    NO_COVERAGE = "NO_COVERAGE"
    DESCRIPTOR_MISSING = "DESCRIPTOR_MISSING"
    DESCRIPTOR_MISMATCH = "DESCRIPTOR_MISMATCH"
    PAYLOAD_MISSING = "PAYLOAD_MISSING"
    PAYLOAD_CORRUPT = "PAYLOAD_CORRUPT"
    EMPTY_OBSERVATIONS = "EMPTY_OBSERVATIONS"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"


def create_request(symbols, *, reinvested: bool) -> Dict[str, Any]:
    """What is asked for. Symbols as written; the service resolves them.

    Deliberately not resolved by the caller. Resolution is the market-data
    service's job and refusing an ambiguous name is the point of it — a caller
    that resolved first would be deciding what it meant and then asking for
    confirmation.
    """
    return {"contract_version": MARKET_DATA_CONTRACT_VERSION,
            "symbols": list(symbols), "reinvested": bool(reinvested)}


def create_response(snapshot) -> Dict[str, Any]:
    """The two addresses, and nothing that would let a caller skip them.

    No bytes, no URI, no adapter internals. What comes back is how to ask for
    the data again, which is the only thing anything downstream needs.
    """
    return {"contract_version": MARKET_DATA_CONTRACT_VERSION,
            "snapshot_hash": snapshot.snapshot_hash,
            "descriptor_hash": snapshot.descriptor_hash}


def failure(kind: Failure, detail: str, **extra) -> Dict[str, Any]:
    return {"contract_version": MARKET_DATA_CONTRACT_VERSION,
            "kind": kind.value, "detail": detail, **extra}


#: Which status each failure travels as. Grouped by what a caller should *do*,
#: not by where the fault is: everything in the 422 group means "your request
#: cannot be answered as asked", and everything in the 409 group means "the
#: store is inconsistent and a person should look".
STATUS = {
    Failure.AMBIGUOUS_SYMBOL: 422,
    Failure.UNKNOWN_SYMBOL: 422,
    Failure.NOT_TRADABLE: 422,
    Failure.NO_COVERAGE: 422,
    Failure.CONTRACT_MISMATCH: 400,
    Failure.DESCRIPTOR_MISSING: 404,
    Failure.PAYLOAD_MISSING: 409,
    Failure.DESCRIPTOR_MISMATCH: 409,
    Failure.PAYLOAD_CORRUPT: 409,
    Failure.EMPTY_OBSERVATIONS: 409,
}
