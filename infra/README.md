# Deploying the closed pilot to AWS

Terraform owns the infrastructure. Ansible owns the deployment onto the
cluster. Neither holds a secret value. Both run from CI —
`.github/workflows/deploy-aws.yml` — not from a laptop.

```text
Cloudflare (DNS + TLS)
        │  terraform-managed tunnel (outbound-only; no public inbound)
        ▼
   EKS  quantify-test-eks  (us-east-1, ns quantify)
        │  per-service ALB Ingresses (web, identity)
        ▼
   data · identity (Zitadel OIDC) · evaluate · web · tunnel
        │
        ▼
   RDS PostgreSQL 16   (users + plans)     S3 + REST catalog (market-data lake)
```

Point `kubectl` at the cluster with:

```bash
aws eks update-kubeconfig --name quantify-test-eks --region us-east-1
```

## Edge and access control

The Cloudflare tunnel (`config_src="cloudflare"`,
`cloudflare_zero_trust_tunnel_cloudflared_config.pilot` in `cloudflare.tf`) is
the only path in. TLS terminates at Cloudflare; there is no public ALB on :443
and no CIDR allowlist to maintain — an unrouted hostname simply resolves to
nothing serving.

Access control is **Zitadel OIDC** (`auth.quantify.club`, the `quantify-identity`
service; PKCE at `/auth/login`), not a reverse-proxy basic-auth credential.
Hostnames, all via the tunnel: `quantify.club` (redirects to `/research`),
`www.quantify.club`, `auth.quantify.club`, and `workspace.quantify.club` — the
last served by the separate **quantify-workspace** repo (the asset-managers
demo), wired by the terraform vars `workspace_domain_name` + `cluster_albs_ready`,
already set in `environments/test.tfvars`.

## What each layer refuses

Both layers refuse rather than defaulting, and each refusal names a condition
under which deploying would be worse than not deploying.

| Refusal | Where |
|---|---|
| an image not pinned by digest | `terraform plan`, and again at deploy |
| a data policy other than the approved pilot vendor policy | `terraform plan`, and again at deploy |
| backup retention of zero | `terraform plan` |
| an empty model key under `MODEL_ASSISTED` | the cluster, before the service starts |
| a database not at the code's migration head | the playbook, before starting |
| a deployment whose startup proof names a different commit | the playbook, after starting |

## First deployment

The apply runs in CI (`deploy-aws.yml`), which materializes the gitignored
`backend.hcl` and `environments/*.tfvars` from repository secrets and pins the
terraform version that wrote the state. State lives in S3:

```text
bucket = quantify-tfstate-388062344663
key    = quantify/test/terraform.tfstate
```

### Two-phase apply

The tunnel origins wire to the ALBs the `web` and `identity` Ingresses create,
and those ALBs do not exist until the workloads are deployed. `enable_kubernetes`
and `cluster_albs_ready` both default to **false** so a routine apply cannot
create a cluster as a side effect or fail on a load balancer that is not there
yet. A from-scratch bring-up is therefore two phases:

1. `terraform apply` with `enable_kubernetes=true`, `cluster_albs_ready=false` —
   creates the cluster and everything under it.
2. `ansible-playbook services.yml` — deploys the services so the Ingresses
   provision their ALBs.
3. `terraform apply` with `cluster_albs_ready=true` — terraform's `data.aws_lb`
   lookups (`cloudflare.tf`) now find those ALBs and wire the tunnel origins and
   DNS records to them.

On a steady-state redeploy the ALBs already exist, so `environments/test.tfvars`
keeps both true and CI applies in one pass.

### Services

`infra/ansible/services.yml` (`known_services`) deploys, in order:

```bash
ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/services.yml                 # every service in order
ansible-playbook ansible/services.yml -e service=data # or one at a time
```

    data       ingest, provenance, licensing        (Iceberg lake)
    identity   Zitadel OIDC provider
    evaluate   QuantLib + the plan simulator         (stateless)
    web        pages, sessions, plans, /research     (users Postgres/RDS)
    tunnel     the Cloudflare tunnel origin config

The last task asks the service what it is and refuses a pod that is running but
reports the wrong build: a green `kubectl rollout status` is a different claim
from the service answering correctly.

## Where the secrets go

The database password is created **empty** by terraform, on purpose: `terraform
apply` must not be the thing that knows it, or it ends up in the plan file, in
state, and in shell history. The model key is put into Secrets Manager directly:

```bash
aws secretsmanager put-secret-value \
  --secret-id quantify-test/model-api-key \
  --secret-string '{"api_key":"sk-..."}'
```

At deploy time `services.yml` reads the password from a **sensitive terraform
output with `no_log`**, hands it to a cluster Secret, and never prints it, never
writes it to a variables file, and never places it in `ansible_variables` — even
under `-vvv`. The cluster Secret and Terraform state (encrypted in S3, which is
why the backend sets `encrypt = true`) are the only copies.

## Reviewing the plan before applying

```bash
terraform plan -var-file=environments/test.tfvars -out=test.tfplan
terraform show -no-color test.tfplan > test.tfplan.txt
```

Scan the **text** view for a credential assigned a literal value:

```bash
grep -Ein 'sk-[A-Za-z0-9_-]|(password|api[_-]?key|secret_string)[[:space:]]*=[[:space:]]*"|postgresql://[^"<]*:[^"<@]*@' \
  test.tfplan.txt
```

Silence is the pass. The obvious version of this pattern — matching the *word*
`password` anywhere — fires on a completely clean plan: Terraform's own
`password = (sensitive value)` redaction, the resource *name* `model_api_key`,
and the deliberately elided `database_url_template` output. An operator who sees
benign hits on every deploy stops reading them, which is how a check becomes
decoration. The pattern above was tested against a clean plan and against
planted leaks.

Also read the **per-resource list, never the summary count**: a
`Plan: 4 to add, 2 to change, 3 to destroy` line does not tell you that a
Cloudflare tunnel or a Secrets Manager version is among the replacements —
taking `quantify.club` offline — but the per-resource list does. Treat a
Cloudflare or Secrets Manager resource appearing in a plan you did not intend
to touch as a stop signal.

### Keep `test.tfplan.txt`, not `test.tfplan`

The saved binary plan is a zip archive containing `tfplan`, `tfstate` and
`tfstate-prev`. On a **first** plan those are empty and the archive is clean.
On **every plan after that** all three members carry the database password in
plaintext — which a `grep` of the compressed file cannot see: it cannot tell
absent from compressed.

So the artifact that belongs in the evidence set is `test.tfplan.txt`, the
redacted text view, which stays clean in both cases. If you keep the binary
plan for a re-apply, treat it exactly like state: encrypted, access-controlled,
deleted afterwards.

## The deploy-time journeys, and when they decline

The playbook runs both supported journeys against the real image, the real
configuration and RDS — then deletes the plans it made, so the first pilot
user opens an empty workspace.

**It only does this when the workspace is empty.** The pilot has one owner and
no per-plan deletion, so on a redeploy a smoke journey would write into a real
user's list, and the only cleanup available — whole-workspace erasure — would
take their data with it. Once plans exist the playbook says so and skips. A
check that cannot run safely should decline rather than run anyway.

After that, verify by hand in a browser: `docs/Runbook.md`, deployment
sequence step 8.

## What this does not do

- **No rate limits or cost caps** (Gate 8). The AWS budget alarm bounds
  infrastructure spend only. Nothing here caps model spend, and that bill
  comes from the provider, not AWS. Set a provider-side budget alert before the
  first invitation.
- **No egress allowlist** (Gate 10). The application pods have open outbound
  access; they need the model provider and the registry. Scope the credentials
  narrowly and watch outbound traffic.
- **Single-AZ RDS by default.** `db_multi_az` is false; set it if that changes.
- **One NAT gateway**, an expensive component here. Losing it stops egress,
  which means the parser stops answering and the application returns 503 — the
  refusal it is designed to give, not a wrong answer.
- **Node-pool sizing is explicit**, not automatic guesswork: `managed_node_pools`
  (`["system"]` moves app pods onto the custom `general-large` pool) and the
  `large_node_pool_*` levers decide instance shape. EKS refuses an empty pool
  list while a node role is set.

## Tearing down

RDS carries `deletion_protection` and `prevent_destroy`, and takes a final
snapshot. `terraform destroy` will not remove it. Overriding means editing
`rds.tf` in a commit — which leaves a record, unlike a `-target` typed at
three in the morning. The same guard is why a bare `terraform apply` with
reconstructed variables is dangerous: see `docs/Runbook.md` §
"Terraform: do not run a bare `apply`".
