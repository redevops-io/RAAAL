# Cloudflare owns DNS and TLS. The pilot has no public inbound path at all.
#
#   browser ──TLS──> Cloudflare edge ──outbound tunnel──> cloudflared (on EC2)
#                                                       └─> internal ALB :80
#                                                           └─> Caddy (basic auth)
#                                                               └─> api :8000
#
# `cloudflared` dials *out* to Cloudflare and holds the connection open, so
# there is no listener on the internet, no ACM certificate, no Route 53 record
# and no security-group rule admitting a public CIDR. The load balancer is
# internal; nothing outside the VPC can reach it even if a rule were wrong.
#
# This also removes the CIDR-allowlist problem entirely: pilot users on mobile
# networks and changing home addresses are authenticated by the credential
# Caddy enforces, not by where they happen to be.

resource "cloudflare_zero_trust_tunnel_cloudflared" "pilot" {
  account_id = var.cloudflare_account_id
  name       = local.name
  # The tunnel's own credential. Generated here, read by the host from
  # Secrets Manager, and never placed in a variable or an output.
  secret     = random_password.tunnel.result
  config_src = "cloudflare"
}

# Ingress rules live in Cloudflare rather than in a config file on the host,
# so the routing is one declarative thing rather than two that can disagree.
# Where the tunnel sends traffic.
#
# One origin per hostname, and no choice left in it. This was a variable during
# the migration so the cutover and the rollback were the same command; EC2 is
# gone, so a variable would now describe a decision nobody can make. It is also
# the variable whose default silently moved production back to the old
# deployment, which is an argument for having no default rather than a better
# one.
#
# Two ALBs, not one: EKS Auto Mode's controller does not implement
# `alb.ingress.kubernetes.io/group.name`, so each Ingress gets its own.
# Observed, not assumed — both reconciled and reported different hostnames.
# Gated on `cluster_albs_ready` (see variables.tf). On a from-scratch bring-up
# these ALBs do not exist until `services.yml` has deployed the Ingresses, and a
# data source that resolves nothing fails the whole plan — so the first apply
# runs with this false, and the second (after the workloads are up) with it true.
data "aws_lb" "cluster_web" {
  count = var.cluster_albs_ready ? 1 : 0
  tags  = { "ingress.eks.amazonaws.com/stack" = "quantify/quantify-web" }
}

data "aws_lb" "cluster_identity" {
  count = var.cluster_albs_ready ? 1 : 0
  tags  = { "ingress.eks.amazonaws.com/stack" = "quantify/quantify-identity" }
}

# The Portfolio Operations workspace's own internal ALB — provisioned by the
# quantify-workspace Ingress (its own namespace + name), found by the same stack tag
# the controller writes. Guarded on the domain being set, so this reconciles only once
# the workspace is deployed.
data "aws_lb" "cluster_workspace" {
  count = (var.cluster_albs_ready && var.workspace_domain_name != "") ? 1 : 0
  tags  = { "ingress.eks.amazonaws.com/stack" = "quantify-workspace/quantify-workspace" }
}

locals {
  tunnel_web_origin       = var.cluster_albs_ready ? "http://${data.aws_lb.cluster_web[0].dns_name}:80" : ""
  tunnel_identity_origin  = var.cluster_albs_ready ? "http://${data.aws_lb.cluster_identity[0].dns_name}:80" : ""
  tunnel_workspace_origin = (var.cluster_albs_ready && var.workspace_domain_name != "") ? "http://${data.aws_lb.cluster_workspace[0].dns_name}:80" : ""
}

resource "cloudflare_zero_trust_tunnel_cloudflared_config" "pilot" {
  # The tunnel exists in phase one so its token can be stored; the routing it
  # carries names the ALBs, so it waits for them — see `cluster_albs_ready`.
  count      = var.cluster_albs_ready ? 1 : 0
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.pilot.id

  config {
    ingress_rule {
      hostname = var.domain_name
      # The internal load balancer. Keeping it in the path preserves the
      # target-group health check and the 5xx metric that the CloudWatch
      # alarms read — the tunnel itself reports nothing to AWS.
      service = local.tunnel_web_origin

      origin_request {
        connect_timeout = "30s"
        # The origin is plaintext HTTP inside the VPC. TLS terminates at
        # Cloudflare's edge; between the edge and here the traffic is inside
        # the tunnel, and from cloudflared to the ALB it never leaves the
        # private subnets.
        no_tls_verify = false
      }
    }

    # The identity provider, on its own hostname through the same tunnel and
    # the same load balancer. Caddy on the host routes by Host header, so this
    # adds a name rather than a second ingress path — and Zitadel's issuer must
    # be a hostname it is reached by, because OIDC discovery publishes it and
    # every token carries it as `iss`.
    dynamic "ingress_rule" {
      for_each = var.identity_domain_name == "" ? [] : [1]
      content {
        hostname = var.identity_domain_name
        service  = local.tunnel_identity_origin

        origin_request {
          connect_timeout = "30s"
          no_tls_verify   = false
        }
      }
    }

    # The Portfolio Operations workspace, on its own hostname through the same tunnel,
    # to its own internal ALB. Its nginx serves the SPA and proxies the wealth-manager
    # API in-cluster, so this is one more name, not a second origin path.
    dynamic "ingress_rule" {
      for_each = var.workspace_domain_name == "" ? [] : [1]
      content {
        hostname = var.workspace_domain_name
        service  = local.tunnel_workspace_origin

        origin_request {
          connect_timeout = "30s"
          no_tls_verify   = false
        }
      }
    }

    # Required terminal rule.
    # `www`, so the proxy can redirect it to the apex.
    #
    # A redirect rule at the Cloudflare edge would be better — it never
    # reaches the origin — and it needs a Dynamic Redirect permission this
    # deployment's API token does not carry. Widening a token to save one hop
    # for a hostname nobody should be using is the wrong trade, so the
    # redirect lives in the proxy, which this token already manages.
    ingress_rule {
      hostname = "www.${var.domain_name}"
      # The apex origin, because whatever serves the apex is what redirects
      # `www` to it. On EC2 that is Caddy; in the cluster it is a redirect
      # action on the web Ingress, verified to answer 301 with the path and
      # query preserved. Pointing this at the identity origin instead would
      # send `www` to the login page of a hostname it is not registered for.
      service = local.tunnel_web_origin
    }

    ingress_rule {
      service = "http_status:404"
    }
  }
}

resource "cloudflare_record" "pilot" {
  # Published in phase two, with the tunnel config — a record pointing at a
  # tunnel that carries no ingress rule for the host resolves to nothing serving.
  count   = var.cluster_albs_ready ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = var.domain_name
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pilot.id}.cfargotunnel.com"
  type    = "CNAME"
  # Proxied is mandatory, not a preference: an unproxied record would expose
  # the tunnel address directly and skip the edge that terminates TLS.
  proxied = true
  comment = "Quantify closed pilot — synthetic data only"

  # The apex already carried a CNAME to the `vibexgen-proxmox` tunnel
  # (1711f014-…), created 2025-11-21 and returning 502: that tunnel is healthy
  # but has no ingress rule for this hostname, so the record pointed at
  # nothing serving. Terraform takes ownership of the name rather than failing
  # on the collision.
  #
  # This flag is a footgun in general — it silently replaces whatever is
  # there — and it is used here deliberately, once, with the prior value
  # recorded in evidence/dns-record-replaced.json. It only affects
  # quantify.club; the other hostnames on that tunnel are untouched.
  allow_overwrite = true
}


# The identity provider's own name. Separate from the application's because a
# token's issuer is part of its identity: moving the provider under a path on
# the main hostname would make every issued token invalid the day it moved.
resource "cloudflare_record" "identity" {
  count = (var.cluster_albs_ready && var.identity_domain_name != "") ? 1 : 0

  zone_id = var.cloudflare_zone_id
  name    = var.identity_domain_name
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pilot.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}


# The Portfolio Operations workspace's own name, resolving through the tunnel exactly
# as the apex and identity do. Published with the tunnel rule above (a record pointing
# at a tunnel that carries no ingress rule for the host resolves to nothing serving).
resource "cloudflare_record" "workspace" {
  count = (var.cluster_albs_ready && var.workspace_domain_name != "") ? 1 : 0

  zone_id = var.cloudflare_zone_id
  name    = var.workspace_domain_name
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pilot.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
  comment = "Quantify workspace — Portfolio Operations"
}


# `www`, redirected to the apex rather than served.
#
# It carried a CNAME to the `vibexgen-proxmox` tunnel — the same leftover the
# apex had, from the same 2025 deployment — and answered 404, because that
# tunnel is healthy and has no ingress rule for this hostname. Anybody
# reaching the site by habit got nothing.
#
# Redirected rather than pointed at our tunnel, and the reason is specific to
# this deployment rather than tidiness: `PUBLIC_BASE_URL` and the OIDC
# redirect URI are both `https://quantify.club`. A person who signed in from
# `www` would be sent back to the apex with a session cookie scoped to a
# hostname they are no longer on, and the failure would look like a login that
# silently does not take. One canonical hostname removes the class.
resource "cloudflare_record" "www" {
  count   = var.cluster_albs_ready ? 1 : 0
  zone_id = var.cloudflare_zone_id
  name    = "www.${var.domain_name}"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pilot.id}.cfargotunnel.com"
  type    = "CNAME"

  # Proxied so the redirect below can happen at the edge. Unproxied, the rule
  # never runs and the name resolves to a tunnel with no ingress for it —
  # which is exactly the state being fixed.
  proxied = true
  comment = "Redirects to the apex — see cloudflare_ruleset.www_to_apex"

  # No `allow_overwrite`. It is used once in this file, on the apex, and
  # deliberately; here it failed to claim a hand-edited record ("attempted to
  # override existing record however didn't find an exact match") because the
  # provider only overwrites what it can match exactly. Deleting the stale
  # record and letting Terraform create this one is a smaller act than a flag
  # that silently replaces whatever it finds.
}

