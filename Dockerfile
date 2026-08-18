# The production image. It serves the pilot API and nothing else.
#
# This ran `scripts/service.py` — the Bokeh dashboard — until Gate 3, while
# nothing anywhere served `src/api.py`. The pilot application, its gated
# routers and its entire startup preflight had no deployment entrypoint: every
# control was on a path the container never took. The dashboard now lives in
# `Dockerfile.dashboard` and is a development surface.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# The API surface does not need the ML/RL stack.
COPY requirements-core.txt ./

# `git` is a *build* dependency, not a runtime one. Two requirements are
# `git+https://` references — `runtime-contracts` and `agentic-os`, both pinned
# by tag — and pip shells out to git to fetch them. `python:3.13-slim` has no
# git, so this layer failed with "Cannot find command 'git'" the first time the
# image was built from a commit that had those pins.
#
# It had never been built from one before: the deployed image was `3eaa5eb`,
# which had no git dependencies at all. The pins arrived afterwards and nothing
# rebuilt until now, so a broken Dockerfile sat behind a running service.
#
# Installed and removed in one layer so the served image carries neither git
# nor the package lists. A runtime image holding build tools is a larger attack
# surface for no benefit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir -r requirements-core.txt \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

# The English model, baked in.
#
# `StanzaReader` loads its pipeline with `download_method=None`, which is
# deliberate: a reader that fetches a 500MB model on first request turns a
# parse into a network call and a cold start into a timeout. So the model has
# to be in the image, and a build that cannot fetch it must fail here rather
# than at the first sentence somebody types.
#
# Only the four processors the reader asks for — `tokenize,pos,lemma,depparse`.
# `ner` and `constituency` are absent for the same reason they are absent from
# `PROCESSORS`: they would cost size and time for evidence this layer does not
# score.
ENV STANZA_RESOURCES_DIR=/opt/stanza
RUN python -c "import stanza; stanza.download('en', model_dir='/opt/stanza',         processors='tokenize,pos,lemma,depparse', verbose=False)"     && python -c "import stanza; stanza.Pipeline(lang='en', dir='/opt/stanza',         processors='tokenize,pos,lemma,depparse', download_method=None,         verbose=False)('a smoke sentence')"

COPY . .

RUN mkdir -p data

# Deliberately unset. `QUANTIFY_DATABASE_URL` has no production default — the
# preflight refuses rather than falling back to a local SQLite file, which
# would be a live path reading a database nobody authorised. The build stamps
# are supplied at deploy time and the preflight refuses without them.
# The two-witness profile, declared and not inferred.
#
# `_declared_profile` reads this rather than checking whether a syntax reader
# was constructed, because MODEL_ONLY and BOTH are claims a plan carries for
# its whole life and inferring one from an import is how an artifact comes to
# say it had two witnesses because a package happened to be installed.
#
# The preflight refuses to serve if this says `yes` and the parser cannot be
# loaded, so a declaration and a capability cannot drift apart silently.
ENV QUANTIFY_DEPLOYMENT_PROFILE=production \
    QUANTIFY_SYNTAX_WITNESS=yes \
    PORT=8000

EXPOSE 8000

# Readiness, not liveness. A process answering on a port is not evidence that
# it has a migrated database, a current schema or an observable build.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# The factory, not the module-level app. `create_app` runs the deployment
# preflight while the process is still starting, so a refusal prevents the
# server binding at all rather than being noticed by a lifespan hook after the
# socket is already open.
CMD ["uvicorn", "src.api:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
