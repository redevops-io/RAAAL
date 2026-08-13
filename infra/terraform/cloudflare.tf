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
resource "cloudflare_zero_trust_tunnel_cloudflared_config" "pilot" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.pilot.id

  config {
    ingress_rule {
      hostname = var.domain_name
      # The internal load balancer. Keeping it in the path preserves the
      # target-group health check and the 5xx metric that the CloudWatch
      # alarms read — the tunnel itself reports nothing to AWS.
      service = "http://${aws_lb.main.dns_name}:80"

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
        service  = "http://${aws_lb.main.dns_name}:80"

        origin_request {
          connect_timeout = "30s"
          no_tls_verify   = false
        }
      }
    }

    # Required terminal rule.
    ingress_rule {
      service = "http_status:404"
    }
  }
}

resource "cloudflare_record" "pilot" {
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
  count = var.identity_domain_name == "" ? 0 : 1

  zone_id = var.cloudflare_zone_id
  name    = var.identity_domain_name
  content = "${cloudflare_zero_trust_tunnel_cloudflared.pilot.id}.cfargotunnel.com"
  type    = "CNAME"
  proxied = true
}
