"""Determinism controls and the run manifest.

Release 0 exit criterion #1 is that two clean environments running the same
commit against the same data snapshot produce identical output digests. That
needs two things this module provides:

* :func:`seed_everything` — one call that pins every RNG in the stack. Previously
  only three seeds existed (sklearn's ``random_state`` in two places and a
  NetworkX layout), while torch DataLoaders shuffled unseeded, so LSTM and
  Transformer runs were not reproducible even on one machine.
* :func:`build_run_manifest` — the record that makes a result checkable. A number
  without a manifest cannot be reproduced or superseded, only re-asserted.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Pin every RNG we can reach. Returns the seed, for recording in the manifest.

    Torch and CUDA are seeded only if torch is installed; the import is deliberately
    lazy so the core pipeline keeps running without the ML extras.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN picks algorithms by benchmarking unless told not to; that choice is
    # nondeterministic and silently changes results between runs on the same host.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _git_state() -> Dict[str, Optional[str]]:
    """Commit SHA plus whether the tree was dirty.

    A pinned commit with uncommitted changes is the most common way a
    reproducibility claim quietly becomes false, so record the diff hash too.
    """
    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    diff = _run(["git", "diff", "HEAD"])
    return {
        "commit": sha,
        "dirty": bool(diff) if diff is not None else None,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def frame_digest(frame: pd.DataFrame) -> str:
    """Content hash of a DataFrame, stable across runs and machines.

    Uses pandas' row hashing over a column-sorted copy so column ordering can't
    change the digest, then hashes the resulting array.
    """
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(ordered, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to decide whether two runs are the same run."""

    run_id: str
    created_at: str
    git: Dict[str, Optional[str]]
    seed: int
    python: str
    platform: str
    packages: Dict[str, str]
    params: Dict[str, Any]
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @property
    def digest(self) -> str:
        """Hash of the reproducibility-relevant fields.

        `created_at` and `run_id` are excluded — two identical runs at different
        times must produce the same digest, or the exit criterion is untestable.
        """
        payload = {
            "git": self.git,
            "seed": self.seed,
            "packages": self.packages,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


def _package_versions() -> Dict[str, str]:
    """Resolved versions of the packages that can move a number."""
    from importlib.metadata import PackageNotFoundError, version

    watched = [
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
        "torch", "yfinance", "networkx", "bokeh",
    ]
    versions: Dict[str, str] = {}
    for name in watched:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_run_manifest(
    *,
    run_id: str,
    seed: int = DEFAULT_SEED,
    params: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
) -> RunManifest:
    """Assemble the manifest for a completed or starting run."""
    return RunManifest(
        run_id=run_id,
        created_at=pd.Timestamp.now("UTC").isoformat(),
        git=_git_state(),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=_package_versions(),
        params=params or {},
        inputs=inputs or {},
        outputs=outputs or {},
    )
