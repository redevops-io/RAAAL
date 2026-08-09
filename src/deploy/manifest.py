"""The build manifest: which code, which schema, which data, which rules.

Deployment facts come from the environment and are never inferred. Reading the
working tree's git state would describe the developer's checkout rather than the
running image, and on a server there is no working tree to read — so the failure
would appear only in the place it matters.

Code versions are imported from the modules that declare them. Restated here
they would be a second list that drifts, and the drift would be invisible
precisely because both lists look authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Facts only the deployment knows. Absent, the manifest is not observable —
#: it does not guess, and it does not fall back to the local checkout.
REQUIRED_DEPLOYMENT_FACTS: Sequence[str] = (
    "QUANTIFY_COMMIT",
    "QUANTIFY_RELEASE_REF",
    "QUANTIFY_IMAGE_DIGEST",
    "QUANTIFY_SNAPSHOT_ID",
)
# `QUANTIFY_MIGRATION_HEAD` was here, and nothing produced it. It is now read
# from the migration scripts in this image — see `code_versions` — because a
# hand-set variable can disagree with the migrations it claims to describe,
# and a schema version that can be wrong is worse than none. What the *database*
# is at is a separate question, checked against this at startup by
# `src.db.migrate.require_migration_head`.

#: Reported alongside, and not required: a deployment can run without the
#: optional integrations and must still be able to describe itself.
OPTIONAL_DEPLOYMENT_FACTS: Sequence[str] = (
    "PILOT_DATA_POLICY",
    "QUANTIFY_RUNTIME_CONTRACTS_VERSION",
    # Whether participants' own sentences are kept. Declared here so a
    # deployment that retains prose says so in the same place it says
    # everything else about itself — a retention decision nobody can read off
    # the running system is a decision the cohort cannot be told about
    # accurately.
    "QUANTIFY_PILOT_TRANSCRIPTS",
)

#: Fields safe to return to a client. Everything else stays private: an image
#: digest and a migration head describe how to attack the deployment, not how to
#: interoperate with it.
PUBLIC_FIELDS: Sequence[str] = (
    "api_version", "compiler_version", "classifier_version",
    "scope_schema_version", "canonicalization_version", "observable",
)


def code_versions() -> Dict[str, str]:
    """Versions the code declares, imported rather than restated."""
    from ..comparison.rsu_profile import PROFILE_VERSION
    from ..db.migrate import code_head
    from ..db.schema import SCHEMA_VERSION
    from ..market_data.integrity import DIGEST_VERSION
    from ..mission.comparability import CLASSIFIER_VERSION
    from ..mission.evolution import COMPILER_VERSION
    from ..mission.parse_model import PARSER_VERSION
    from ..mission.rsu_reconcile import MATCHING_POLICY_VERSION
    from ..mission.rsu_result import CONTEXT_VERSION
    from ..workspace.intent_history import PLANNER_VERSION
    from ..workspace.retention import RETENTION_POLICY_VERSION
    from ..workspace.scope_disclosure import SCOPE_SCHEMA_VERSION

    from ..api import API_VERSION

    return {
        "api_version": API_VERSION,
        # From the migration scripts in this image, not from a variable someone
        # set by hand. The database is checked against it at startup.
        "migration_head": code_head(),
        "schema_version": SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "parser_version": PARSER_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "comparison_profile_version": PROFILE_VERSION,
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "result_context_version": CONTEXT_VERSION,
        "matching_policy_version": MATCHING_POLICY_VERSION,
        "planner_version": PLANNER_VERSION,
        "retention_policy_version": RETENTION_POLICY_VERSION,
        "canonicalization_version": DIGEST_VERSION,
    }


@dataclass(frozen=True)
class BuildManifest:
    """What this deployment is, or a statement that it cannot say."""

    deployment: Mapping[str, str]
    versions: Mapping[str, str]
    missing: Sequence[str] = ()

    @property
    def observable(self) -> bool:
        """Whether every required deployment fact is present.

        False is a useful answer. A manifest that filled the gaps would describe
        a build that does not exist, and would be indistinguishable from a
        correct one.
        """
        return not self.missing

    def private(self) -> Dict[str, Any]:
        """The operator's view. Never returned to a client."""
        return {"observable": self.observable, "missing": list(self.missing),
                **{k: v for k, v in self.deployment.items()},
                **dict(self.versions)}

    def public(self) -> Dict[str, Any]:
        """What a client needs to know it is compatible, and nothing more."""
        payload = self.private()
        return {name: payload[name] for name in PUBLIC_FIELDS
                if name in payload}

    def to_json(self) -> Dict[str, Any]:
        return self.private()


def read_manifest(environ: Mapping[str, str]) -> BuildManifest:
    """Assemble the manifest from the environment and the code.

    Deployment facts are read only from `environ`. There is deliberately no
    fallback to `git rev-parse`: on a server there is no working tree, and in
    development it would report the checkout rather than the running image.
    """
    source = environ

    deployment: Dict[str, str] = {}
    missing: List[str] = []
    for name in REQUIRED_DEPLOYMENT_FACTS:
        value = source.get(name)
        key = name.replace("QUANTIFY_", "").lower()
        if value:
            deployment[key] = value
        else:
            missing.append(name)

    for name in OPTIONAL_DEPLOYMENT_FACTS:
        value = source.get(name)
        if value:
            deployment[name.replace("QUANTIFY_", "").lower()] = value

    return BuildManifest(deployment=deployment, versions=code_versions(),
                         missing=tuple(missing))
