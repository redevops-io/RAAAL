# Deploying the Agentic Investment OS to quantify.club

quantify.club runs on **Amazon EKS**. The cluster is `quantify-test-eks` (us-east-1;
its terraform name is `${project}-${environment}-eks`), namespace `quantify`, reached
only through a **terraform-managed Cloudflare tunnel** (outbound-only; no inbound port,
no public ALB on :443). Point `kubectl` at it with:

```bash
aws eks update-kubeconfig --name quantify-test-eks --region us-east-1
```

**Paper-trading-first / governance boundary:** the image ships no broker client and no
order router, so there is no code path to a real venue. The only state-mutating endpoint
is mission approval, which writes a *paper* order to the local state store. `mode: paper`
is enforced at manifest validation.

## Services

`infra/ansible/services.yml` (`known_services`) knows five services, deployed in order:

    data       ingest, provenance, licensing        (Iceberg lake: S3 + REST catalog)
    identity   Zitadel OIDC provider (auth.quantify.club)
    evaluate   QuantLib + the plan simulator        (stateless)
    web        pages, sessions, plans, /research     (users Postgres/RDS)
    tunnel     the Cloudflare tunnel origin config

## How deployment happens

Terraform owns the infrastructure; Ansible owns the deployment onto the cluster. Both
run from **CI** — `.github/workflows/deploy-aws.yml` — not by hand. Terraform state is
in S3 (`quantify-tfstate-388062344663`, key `quantify/test/terraform.tfstate`; the
`backend.hcl` is materialized by the workflow). Images are pulled from
`388062344663.dkr.ecr.us-east-1.amazonaws.com`.

The deploy step runs `ansible-playbook services.yml` (kubernetes.core.k8s), which points
`kubectl` at the cluster, applies each service's manifests, and refuses a pod that is
running but reports the wrong build.

### Two-phase apply (from-scratch bring-up)

The Cloudflare tunnel origins wire to the ALBs the `web` and `identity` Ingresses create,
and those ALBs do not exist until the workloads are deployed. So a from-scratch bring-up
is two phases:

1. `terraform apply` with `enable_kubernetes=true` and `cluster_albs_ready=false` — creates
   the cluster and everything under it.
2. Deploy the services (`ansible-playbook services.yml`) so the Ingresses provision their ALBs.
3. `terraform apply` again with `cluster_albs_ready=true` — terraform's `data.aws_lb` lookups
   (see `infra/terraform/cloudflare.tf`) now find those ALBs and wire the tunnel origins and
   DNS records to them.

On a steady-state redeploy the ALBs already exist, so `test.tfvars` keeps both true.

## Edge and auth

- **Edge = Cloudflare tunnel**, terraform-managed (`config_src="cloudflare"`,
  `cloudflare_zero_trust_tunnel_cloudflared_config.pilot` in `cloudflare.tf`). TLS
  terminates at Cloudflare. Hostnames, all via the tunnel: `quantify.club` (redirects to
  `/research`), `www.quantify.club`, `auth.quantify.club`, and `workspace.quantify.club`
  (the asset-managers demo, served by the separate **quantify-workspace** repo; wired by
  the terraform vars `workspace_domain_name` + `cluster_albs_ready`, already set in
  `infra/terraform/environments/test.tfvars`).
- **Auth = Zitadel OIDC** (`auth.quantify.club`, the `quantify-identity` service; PKCE at
  `/auth/login`).

## Smoke test after deploy

```bash
kubectl -n quantify rollout status deploy/quantify-web
kubectl -n quantify get pods
curl -s https://quantify.club/health
# open https://quantify.club/  -> redirects to /research (nightly Bokeh dashboard)
# open https://auth.quantify.club/  -> Zitadel login (PKCE)
```

## Retired paths

The EC2 + Caddy + ALB:443 + docker-compose-on-one-host path, and the earlier proxmox
`investment-agent` single-container path, are retired. `infra/ansible/services.yml`'s
header states this is now the only deployment path; the EC2 deployment is preserved under
`archive/ec2-deployment/`.

For the full terraform + ansible detail — variables, the plan-review discipline, secret
handling and teardown guards — see [`infra/README.md`](../infra/README.md).
