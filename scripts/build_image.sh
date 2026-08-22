#!/usr/bin/env bash
#
# The one way this project's serving image is built.
#
# Inline in a workflow, a build procedure is invisible to everything else —
# and the moment a second consumer needs one, it gets a second implementation.
# That is how `requirements-core.txt` and `requirements.txt` came to answer
# "which Discovery Runtime?" differently for a month. The rule this exists to
# hold:
#
#     A dependency is not pinned if different consumers can resolve it from
#     different authorities.
#
# So the gitlink is the authority. The submodule is checked out at exactly the
# commit this repository records, both requirements files install that tree,
# and this script refuses to build if any of that is untrue.
#
#     scripts/build_image.sh                 build and verify, no registry
#     scripts/build_image.sh --push          also push and resolve the digest
#
# Emits, on the last line of stdout, the digest-pinned reference a deployment
# may consume. Never a tag: a tag is a mutable pointer, and Terraform refuses
# one for the same reason a model alias under a fixed reader id is refused.
set -euo pipefail

PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n=== %s\n' "$*" >&2; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# --- 1. the source is a commit, and it is clean --------------------------
say "1. source identity"
COMMIT="$(git rev-parse HEAD)"
git diff --quiet && git diff --cached --quiet \
  || die "the working tree is dirty. The commit is the deployment's identity,
        so the image must be built from exactly what is committed."
printf '   commit           %s\n' "$COMMIT" >&2

# --- 2. the submodule is at the gitlink this commit records ---------------
#
# `git submodule status` prefixes the sha with `+` when the checked-out commit
# differs from the recorded gitlink and `-` when it is not initialised. Either
# means the tree about to be copied into the image is not the tree this
# repository pins — which is the whole defect this script exists to prevent,
# and it is invisible from inside the container afterwards.
say "2. submodule identity"
STATUS="$(git submodule status --recursive vendor/discovery-runtime)"
case "$STATUS" in
  \+*) die "the submodule is not at the recorded gitlink: $STATUS" ;;
  -*)  die "the submodule is not initialised: $STATUS
        run: git submodule update --init --recursive" ;;
esac
GITLINK="$(git rev-parse HEAD:vendor/discovery-runtime)"
ACTUAL="$(git -C vendor/discovery-runtime rev-parse HEAD)"
[[ "$GITLINK" == "$ACTUAL" ]] \
  || die "gitlink $GITLINK != checkout $ACTUAL"
DESCRIBED="$(git -C vendor/discovery-runtime describe --tags --exact-match 2>/dev/null || true)"
[[ -n "$DESCRIBED" ]] \
  || die "the submodule is at $ACTUAL, which is not a tagged release.
        The serving path depends on releases, not on loose commits."
printf '   gitlink          %s (%s)\n' "${GITLINK:0:12}" "$DESCRIBED" >&2

# --- 3. build ------------------------------------------------------------
say "3. docker build"
TAG="quantify-web:${COMMIT:0:7}"
docker build --no-cache -t "$TAG" . >&2
IMAGE_ID="$(docker image inspect "$TAG" --format '{{.Id}}')"
SIZE="$(docker image inspect "$TAG" --format '{{.Size}}')"
printf '   image            %s\n   size             %s bytes\n' "$IMAGE_ID" "$SIZE" >&2

# --- 4. the image holds the runtime this commit pins ----------------------
#
# Asked of the image, not of the build context. A `COPY` that silently missed
# the directory, a cached layer, a stale wheel — none are visible in the source
# and all are visible here.
say "4. installed versions"
# The expected version is what the pinned submodule *declares*, not its tag name.
# discovery-runtime's own convention (and its test_contract_compatibility) is that
# the installed version equals its pyproject version; its tags can run ahead of
# that string (v0.1.11 ships 0.1.10). Checking against the tag conflated two
# things and refused a legitimate release for a versioning-string lag in a
# dependency this repo does not own. Step 2 already required the submodule to be
# at a *tagged* release; this checks the image installed the version that release
# declares. The gitlink is still the authority for *which* code — this only
# confirms the image built from it.
EXPECTED_RUNTIME="$(sed -nE 's/^version = "([^"]+)".*/\1/p' vendor/discovery-runtime/pyproject.toml | head -1)"
[[ -n "$EXPECTED_RUNTIME" ]] || die "could not read the version from vendor/discovery-runtime/pyproject.toml"
INSTALLED="$(docker run --rm --network=none --entrypoint python "$TAG" -c \
  "import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in ('discovery-runtime','runtime-contracts','stanza')}))")"
printf '   %s (submodule %s declares %s)\n' "$INSTALLED" "$DESCRIBED" "$EXPECTED_RUNTIME" >&2
python3 - "$INSTALLED" "$EXPECTED_RUNTIME" <<'PY' || die "the image does not hold the pinned runtime"
import json, sys
got, expected = json.loads(sys.argv[1]), sys.argv[2]
assert got["discovery-runtime"] == expected, (
    f"image has discovery-runtime {got['discovery-runtime']}, the pinned "
    f"submodule declares {expected}")
assert got["stanza"] == "1.14.0", f"image has stanza {got['stanza']}"
# Freeze plan §7: the serving image must hold the FROZEN runtime-contracts floor,
# and nothing in the build (a vendored submodule pinning an older rc, a cached
# wheel) may downgrade below it. A bare `count(".")==2` accepted 0.2.4 — the exact
# version the stale-digest incident shipped — so it is not a freeze guard.
FLOOR = (0, 3, 0)
def as_tuple(raw):
    return tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in raw.split("."))
rc = got["runtime-contracts"]
assert as_tuple(rc) >= FLOOR, (
    f"image has runtime-contracts {rc}, below the frozen floor "
    f"{'.'.join(map(str, FLOOR))} — a downgrade the freeze forbids")
PY

# --- 5. the serving-image contract ---------------------------------------
#
# The ten gates: the model is present, the parser runs with no network, the
# cases that earned the syntax witness parse, and a container without the
# model refuses to serve rather than quietly becoming a MODEL_ONLY server.
say "5. serving-image contract"
PYTEST="${PYTEST:-python3 -m pytest}"
QUANTIFY_IMAGE="$TAG" $PYTEST tests/test_serving_image_contract.py -q >&2 \
  || die "the built image does not satisfy the serving contract"

if [[ "$PUSH" -eq 0 ]]; then
  say "built and verified; not pushed"
  printf '%s\n' "$TAG"
  exit 0
fi

# --- 6. push, and resolve the digest the registry assigned ---------------
say "6. push"
: "${AWS_REGION:?AWS_REGION is required to push}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY is required to push}"
REGISTRY="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY" >&2

REMOTE="${REGISTRY}/${ECR_REPOSITORY}:${COMMIT:0:7}"
docker tag "$TAG" "$REMOTE"
docker push "$REMOTE" >&2

# --- 7. emit the digest, never the tag -----------------------------------
say "7. digest"
DIGEST="$(aws ecr describe-images --region "$AWS_REGION" \
  --repository-name "$ECR_REPOSITORY" --image-ids imageTag="${COMMIT:0:7}" \
  --query 'imageDetails[0].imageDigest' --output text)"
[[ "$DIGEST" == sha256:* ]] || die "could not resolve a digest for ${COMMIT:0:7}"
printf '   %s\n' "$DIGEST" >&2

# --- 7b. the immutable release manifest (freeze plan §5) -----------------
#
# Bind the code, the image, and the exact dependency versions into one manifest,
# so canary validation and production promotion reference the same release
# identity rather than a mutable tag or a stale digest file (the incident).
say "7b. release manifest"
CANON="$(docker run --rm --network=none --entrypoint python "$TAG" -c \
  'import runtime_contracts as rc; print(rc.CANONICALIZATION_VERSION)')"
RC_VERSION="$(printf '%s' "$INSTALLED" | python3 -c 'import json,sys; print(json.load(sys.stdin)["runtime-contracts"])')"
python3 -m deploy.release.manifest build \
  --app-commit "$COMMIT" --image-digest "$DIGEST" \
  --runtime-contracts "$RC_VERSION" --discovery-runtime "$EXPECTED_RUNTIME" \
  --canonicalization "$CANON" --payload-schema "redevops/strategy-selection" \
  --build-timestamp "${SOURCE_DATE_EPOCH:-}" --out release-manifest.json >&2
printf '   wrote release-manifest.json (commit=%s rc=%s canon=%s)\n' \
  "${COMMIT:0:7}" "$RC_VERSION" "$CANON" >&2

printf '%s\n' "${REGISTRY}/${ECR_REPOSITORY}@${DIGEST}"
