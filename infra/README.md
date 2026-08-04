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
| an image not pinned by digest | `terraform plan`, and again in `site.yml` |
| a data policy other than `SYNTHETIC_ONLY` | `terraform plan`, and again in `site.yml` |
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
ansible-playbook -i inventory.aws_ec2.yml site.yml -e @/tmp/quantify.json
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
