# License

RAAAL / Quantify is licensed under the **GNU Affero General Public License v3.0
or later (AGPL-3.0-or-later)**. The full license text is in [LICENSE](LICENSE).

Copyright (c) 2025-2026 RedevOps.

## Relicensing note

This project was previously distributed under the MIT License. It was relicensed to
AGPL-3.0-or-later on **2026-07-30** to align with the ReDevOps open-core split: the
OSS runtime engine (`redevops-io/context-runtime`, `redevops-io/redevops-rag`) is
AGPL, and this project builds on it. MIT is compatible with AGPL, so relicensing the
existing code is permitted; `git log` records a single author to date.

Contributions published before 2026-07-30 under MIT remain available under MIT from
the commits in which they were published.

## What AGPL §13 means here

AGPL extends copyleft to **network use**. If you run a modified version of this
software and let users interact with it over a network, you must offer those users
the corresponding source of your modified version.

Any deployment of this project — including the hosted dashboard — must therefore
present a visible offer of source, not merely keep the repository public.

## What must not be combined with this code

`redevops-io/CR-enterprise` is **proprietary** and must not be linked into or
distributed with this AGPL work. Where an enterprise pattern is wanted here
(Policy-Constrained Planning, the TrustLedger), it must be reimplemented rather than
imported.
