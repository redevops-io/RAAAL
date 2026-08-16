# Three security groups in a chain, and an instance profile that can read
# exactly three secrets.
#
# The chain is the whole isolation story, and with a Cloudflare tunnel it has
# no public end at all: nothing on the internet can reach any of these. The
# tunnel client on the application host reaches the internal load balancer,
# the load balancer reaches the application, and the application is the only
# thing that reaches the database. Each rule names a security group rather
# than a CIDR, so widening one does not silently widen the next.


# The only client is `cloudflared`, which runs on the application host. There
# is no CIDR here and no public rule: with a tunnel there is nothing to
# allowlist, which also removes the problem that a CIDR allowlist would have
# created for pilot users on mobile networks and changing home addresses.



resource "aws_security_group" "database" {
  name        = "${local.name}-database"
  description = "PostgreSQL, reachable only from the application"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-database" }
}


# --- instance identity -----------------------------------------------------



# Named secrets, not a wildcard. The host can read the three values it needs
# to run and nothing else in the account — a compromised host should not be a
# key to every secret the organisation owns.

