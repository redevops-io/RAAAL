# The EC2 deployment, retired 2026-08-16

Quantify ran here from the start of the pilot until the move to EKS. It is kept
because it is the record of how several things were solved, not because it can
be redeployed — the terraform that built the instance, its load balancer, its
security groups and its SSM endpoints was deleted in the same change, so
nothing here has anything to run against.

## What lived here

`site.yml` provisioned one Ubuntu host over SSM and ran a docker-compose stack:
Caddy as the reverse proxy, the application, Zitadel, and the `cloudflared`
connector. `roles/quantify/templates/` holds the templates, and two of them are
worth reading before changing the cluster equivalents.

**`Caddyfile.j2`** is the one that mattered on the way out. Caddy was doing four
jobs that were invisible until it was gone:

- setting `X-Forwarded-Proto: https` on the identity host, without which Zitadel
  builds its issuer from the request it actually received and publishes
  `http://auth.quantify.club`. `ZITADEL_EXTERNALSECURE=true` does not fix this
  and the file says so.
- `redir / /workspace/ 302`, because the application root returns service
  metadata and a pilot invitee typing the hostname should see the product.
- redirecting `www` to the apex, because `PUBLIC_BASE_URL` and the registered
  OIDC redirect URI are both the apex.
- four security headers.

All four are reproduced in the cluster — the first in a proxy container in the
identity pod, the middle two as ALB redirect actions on the web Ingress, the
last as application middleware. The last one is middleware rather than
infrastructure precisely because this migration proved that a header living in
the proxy disappears with the proxy and nothing reports worse for it.

**`docker-compose.yml.j2`** documents the connector: token-only, pinned by
digest, `config_src = "cloudflare"` so the routing is remote. The cluster
manifest at `deploy/kubernetes/tunnel.yaml` keeps all three properties and adds
a second replica.

## What did not move

The `basic_auth` block, which was already inert — it is skipped whenever an
identity provider is configured, and the guard is `require_a_signed_in_viewer`
in `src/api.py`. Nothing was lost by dropping it.
