# License

RAAAL / Quantify is **source-available**, not open source. It is licensed under
the **GNU Affero General Public License v3.0 or later**, as modified by the
**Commons Clause License Condition v1.0**.

- Full AGPL text: [LICENSE](LICENSE)
- The added condition: [LICENSE-COMMONS-CLAUSE](LICENSE-COMMONS-CLAUSE)

Copyright (c) 2025-2026 RedevOps.

    SPDX-License-Identifier: LicenseRef-AGPL-3.0-or-later-with-Commons-Clause

There is no registered SPDX identifier for this combination — Commons Clause is
a licence *condition*, not an SPDX exception — so a `LicenseRef-` name is used.
Tooling that expects a standard identifier will report this project as
unrecognised, and GitHub will stop labelling it AGPL. That is accurate: it is
not AGPL any more.

## What this permits, and what it does not

You may read, run, modify, self-host and redistribute this software, including
inside a company for its own purposes. Every AGPL obligation still applies —
most importantly §13, below.

You may **not sell it**. In the words of the condition, you may not exercise
the licence rights "to provide to third parties, for a fee or other
consideration (including without limitation fees for hosting or
consulting/support services related to the Software), a product or service
whose value derives, entirely or substantially, from the functionality of the
Software."

Running a paid service built on Quantify is the case this exists to prevent.

## Why the Commons Clause and not AGPL alone

AGPL does **not** restrict commercial use. It explicitly permits charging a fee
(§4) and permits commercial deployment. What it restricts is *proprietary*
use: §13 forces anyone offering a modified version over a network to give its
users the corresponding source.

That deters a closed fork. It does not stop a company running this as a
commercial service, provided they publish their changes. Barring that requires
a term AGPL does not contain, which is what the Commons Clause adds — at the
cost of the project no longer being open source under the OSI definition.

## What AGPL §13 means here

AGPL extends copyleft to **network use**. If you run a modified version of this
software and let users interact with it over a network, you must offer those
users the corresponding source of your modified version.

Any deployment of this project — including the hosted pilot at
`quantify.club` — must therefore present a visible offer of source, not merely
keep the repository public. The served pages carry that offer in the footer.

## Relicensing history

- Originally distributed under the **MIT License**.
- Relicensed to **AGPL-3.0-or-later** on 2026-07-30, to align with the ReDevOps
  open-core split.
- The **Commons Clause** condition was added on 2026-08-07.

MIT is compatible with AGPL, so relicensing the existing code was permitted.
Adding the Commons Clause is permitted here for a narrower reason: RedevOps
holds copyright in this work and no AGPL-licensed dependency is linked into it.
Were one linked, this condition could not lawfully be added to the combined
work, because AGPL §7 forbids imposing further restrictions on code received
under it.

Contributions published before 2026-07-30 under MIT remain available under MIT
from the commits in which they were published. Code published between
2026-07-30 and 2026-08-07 remains available under plain AGPL-3.0-or-later from
those commits.

## Dependencies

No dependency is under GPL or AGPL. `certifi` is MPL-2.0, which is file-level
copyleft and imposes no condition on this project's own terms.

`redevops-io/context-runtime` and `redevops-io/redevops-rag` are AGPL and are
**planned** dependencies; no linkage exists in the current tree. If and when
they are linked, the Commons Clause condition can no longer be applied to the
combined work, and this file must be revisited before that merge lands.

## What must not be combined with this code

`redevops-io/CR-enterprise` is **proprietary** and must not be linked into or
distributed with this work. Where an enterprise pattern is wanted here
(Policy-Constrained Planning, the TrustLedger), it must be reimplemented rather
than imported.

## Not legal advice

This file records the intent and the mechanics of the choice. It is not legal
advice, and the Commons Clause has not been tested in court.
