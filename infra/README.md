# Deploying the closed pilot to AWS

Terraform owns the infrastructure. Ansible owns the host, the configuration
and the deployment. Neither holds a secret value.

```text
Route 53 ── ACM ── ALB :443
                     │
                     ▼  private subnet
              EC2 :80 ── Caddy (basic auth) ── api :8000
                     │
                     ▼
              RDS PostgreSQL 16
```

## Why Caddy is still here

The ALB terminates TLS, so Caddy is no longer the TLS endpoint. It stays for
one job: **basic auth on `/workspace/*`**. An ALB cannot do HTTP basic auth,
and that rule is the entire access control for the closed pilot — there are no
user accounts, and one `pilot` owner holds every plan. `deploy/acceptance.py`
checks for it. Removing the proxy would delete a control while leaving every
other part of the deployment looking correct.

## What each layer refuses

Both layers refuse rather than defaulting, and each refusal names a condition
under which deploying would be worse than not deploying.

| Refusal | Where |
|---|---|
| an image not pinned by digest | `terraform plan`, and again at deploy |
| a data policy other than `SYNTHETIC_ONLY` | `terraform plan`, and again at deploy |
| backup retention of zero | `terraform plan` |
| an empty model key under `MODEL_ASSISTED` | the host, before the service starts |
| a basic-auth secret with no hash | the host, before the service starts |
| a password that corrupts the connection string | the host, after rendering |
| a database not at the code's migration head | the playbook, before starting |
| a deployment whose startup proof names a different commit | the playbook, after starting |

## First deployment

Bootstrap the state bucket once (see `terraform/backend.hcl.example`), then:

```bash
cd infra/terraform
cp backend.hcl.example backend.hcl                       # edit
cp environments/test.tfvars.example environments/test.tfvars   # edit
terraform init -backend-config=backend.hcl
terraform apply -var-file=environments/test.tfvars
```

Terraform creates the model key and workspace credential **empty**, on
purpose: `terraform apply` must not be the thing that knows them, or they end
up in the plan file, in state, and in shell history. Put them in directly:

```bash
aws secretsmanager put-secret-value \
  --secret-id quantify-test/model-api-key \
  --secret-string '{"api_key":"sk-ant-..."}'

caddy hash-password --plaintext 'the-pilot-password'      # copy the output
aws secretsmanager put-secret-value \
  --secret-id quantify-test/workspace-basic-auth \
  --secret-string '{"username":"pilot","password_hash":"$2a$14$..."}'
```

Then deploy:

```bash
cd ../ansible
ansible-galaxy install -r requirements.yml
terraform -chdir=../terraform output -json ansible_variables > /tmp/quantify.json
ansible-playbook services.yml -e service=data,evaluate,web
```

Finally, from your machine — against the public URL, not the container:

```bash
cd ../..
mkdir -p evidence
python deploy/acceptance.py https://YOUR-HOST --record evidence/acceptance.json
aws logs tail /quantify/test/application --since 15m \
  | grep "deployment proof" > evidence/startup-proof.txt
```

Two files, kept together. See `docs/Runbook.md` § Deployment evidence for why
they are two and not one.

## Where the secrets go

They are fetched **on the host**, under its instance profile, by
`render-env.sh`. They do not pass through the Ansible controller, its fact
cache, or a `-vvv` run. The controller renders a template containing
`${QUANTIFY_DB_PASSWORD}` and `${QUANTIFY_MODEL_KEY}`; the host expands it.

`envsubst` is called with an explicit variable list rather than bare. A bcrypt
hash is full of `$`, and a bare `envsubst` would eat the credential it was
installing.

## Where the database password is, and is not

Audited before the first apply, by exercising each path rather than reading
it. The URL-substitution defect was real, so "it should be fine" was not good
enough for the places the value could surface.

| Surface | Result |
|---|---|
| `terraform output` | absent — outputs carry secret *names*, never values |
| `/tmp/quantify.json` | absent — `ansible_variables` names the secret only |
| generated compose YAML | absent — `env_file`, no inline environment |
| Ansible output, render task | `no_log`, and the script prints only CHANGED/UNCHANGED |
| Ansible output, migration task | verified absent: a failed connect raises `OperationalError` with no URL, and SQLAlchemy renders the engine as `postgresql+psycopg://quantify:***@…` |
| Alembic | present in the config object in memory — unavoidable, it must connect — but absent from upgrade output at root DEBUG |
| CloudWatch startup proof | absent — the proof reports `"database": {"engine": "postgresql"}` and nothing else: no host, no user, no credential |
| acceptance evidence | absent |

`DatabaseTarget.display` renders `postgresql://***@host:5432/quantify`, which
is what any operator-facing surface gets.

The host-rendered `/opt/quantify/.env` is the only plaintext copy, at `0600`
in a `0750` directory, and Terraform state — encrypted in S3, which is why
`backend.hcl.example` sets `encrypt = true`.

One wrinkle worth knowing before a scanner surprises you: `acceptance.json`
contains the literal string `postgresql://` — inside the *name* of the check
`"an error does not leak 'postgresql://'"`. A naive secret-scanner flags the
evidence file for carrying the string it exists to prove absent.

## Reviewing the plan before applying

```bash
terraform plan -var-file=environments/test.tfvars -out=test.tfplan
terraform show -no-color test.tfplan > test.tfplan.txt
```

Scan the **text** view for a credential assigned a literal value:

```bash
grep -Ein 'sk-ant-[A-Za-z0-9_-]|(password|api[_-]?key|secret_string)[[:space:]]*=[[:space:]]*"|postgresql://[^"<]*:[^"<@]*@' \
  test.tfplan.txt
```

Silence is the pass. The obvious version of this pattern — matching the *word*
`password` anywhere — fires four times on a completely clean plan: Terraform's
own `password = (sensitive value)` redaction, the resource *name*
`model_api_key` twice, and the deliberately elided `database_url_template`
output. An operator who sees four benign hits on every deploy stops reading
them by the third, which is how a check becomes decoration. The pattern above
was tested against a clean plan and against six planted leaks.

### Keep `test.tfplan.txt`, not `test.tfplan`

The saved binary plan is a zip archive containing `tfplan`, `tfstate` and
`tfstate-prev`. On a **first** plan those are empty and the archive is clean.
On **every plan after that** all three members carry the database password in
plaintext — verified by unpacking the archive and searching its members, which
a `grep` of the compressed file cannot do: it cannot tell absent from
compressed.

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
  comes from Anthropic, not AWS. Set a provider-side budget alert before the
  first invitation.
- **No egress allowlist** (Gate 10). The application host has open outbound
  access; it needs the model provider and the registry. Scope the credentials
  narrowly and watch outbound traffic.
- **One instance, one AZ, no autoscaling.** A test deployment. RDS is
  single-AZ by default; set `db_multi_az` if that changes.
- **One NAT gateway**, the most expensive component here. Losing it stops
  egress, which means the parser stops answering and the application returns
  503 — the refusal it is designed to give, not a wrong answer.
- **`operator_cidrs` defaults to empty**, so a fresh apply serves nobody.
  That is the safe default, not a mistake to work around.

## Tearing down

RDS carries `deletion_protection` and `prevent_destroy`, and takes a final
snapshot. `terraform destroy` will not remove it. Overriding means editing
`rds.tf` in a commit — which leaves a record, unlike a `-target` typed at
three in the morning.
