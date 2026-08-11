"""What this deployment actually is, and whether it can say.

The build manifest answers one question a running service must be able to answer
about itself: *which code, which schema, which data, which rules*.

**An unanswerable question is answered as unanswerable.** A manifest that
substituted `"unknown"`, `"dev"` or the working tree's git state for a missing
deployment variable would report a build that does not exist, and the report
would look exactly like a correct one. `observable` is False when any required
fact is absent, and the missing names are listed.

**Two views, deliberately.** The private view carries the image digest, the
migration head and the snapshot id; the public view carries only what a client
needs to know it is compatible. A diagnostic endpoint that returned the whole
manifest would publish the deployment's internals to anyone who asked.
"""
from .manifest import (
    BuildManifest,
    REQUIRED_DEPLOYMENT_FACTS,
    code_versions,
    read_manifest,
)

__all__ = ["BuildManifest", "REQUIRED_DEPLOYMENT_FACTS", "code_versions",
           "read_manifest"]
