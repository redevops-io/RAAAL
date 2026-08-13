# Three security groups in a chain, and an instance profile that can read
# exactly three secrets.
#
# The chain is the whole isolation story, and with a Cloudflare tunnel it has
# no public end at all: nothing on the internet can reach any of these. The
# tunnel client on the application host reaches the internal load balancer,
# the load balancer reaches the application, and the application is the only
# thing that reaches the database. Each rule names a security group rather
# than a CIDR, so widening one does not silently widen the next.

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Internal load balancer. Reachable only from the tunnel client."
  vpc_id      = aws_vpc.main.id

  egress {
    description     = "To the application host"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  tags = { Name = "${local.name}-alb" }
}

# The only client is `cloudflared`, which runs on the application host. There
# is no CIDR here and no public rule: with a tunnel there is nothing to
# allowlist, which also removes the problem that a CIDR allowlist would have
# created for pilot users on mobile networks and changing home addresses.
resource "aws_vpc_security_group_ingress_rule" "alb_from_tunnel" {
  security_group_id            = aws_security_group.alb.id
  description                  = "HTTP from cloudflared on the application host"
  referenced_security_group_id = aws_security_group.app.id
  from_port                    = 80
  to_port                      = 80
  ip_protocol                  = "tcp"
}

resource "aws_security_group" "app" {
  name        = "${local.name}-app"
  description = "Application host. No public address, no inbound SSH."
  vpc_id      = aws_vpc.main.id

  # Deliberately no port 22 rule. Access is SSM Session Manager, which leaves
  # an auditable record and needs no key distribution. An SSH rule added here
  # is a key somebody now has to rotate.

  egress {
    description = "Outbound: image registry, the model provider, package mirrors"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-app" }
}

resource "aws_vpc_security_group_ingress_rule" "app_from_alb" {
  security_group_id            = aws_security_group.app.id
  description                  = "HTTP from the load balancer only"
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = 80
  to_port                      = 80
  ip_protocol                  = "tcp"
}

resource "aws_security_group" "database" {
  name        = "${local.name}-database"
  description = "PostgreSQL, reachable only from the application"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-database" }
}

resource "aws_vpc_security_group_ingress_rule" "database_from_app" {
  security_group_id            = aws_security_group.database.id
  description                  = "PostgreSQL from the application host only"
  referenced_security_group_id = aws_security_group.app.id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}

# --- instance identity -----------------------------------------------------

resource "aws_iam_role" "app" {
  name = "${local.name}-app"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Named secrets, not a wildcard. The host can read the three values it needs
# to run and nothing else in the account — a compromised host should not be a
# key to every secret the organisation owns.
resource "aws_iam_role_policy" "secrets" {
  name = "read-pilot-secrets"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        # The identity masterkey joins by `concat` over the resource's own
        # instances rather than by a conditional ARN: a `""` in a policy names
        # nothing, which reads at a glance like a grant and denies at runtime.
        Resource = concat([
          aws_secretsmanager_secret.database_password.arn,
          aws_secretsmanager_secret.model_api_key.arn,
          aws_secretsmanager_secret.workspace_basic_auth.arn,
          aws_secretsmanager_secret.tunnel_token.arn,
          ], [
          for secret in aws_secretsmanager_secret.identity_masterkey :
          secret.arn
        ])
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = "${aws_cloudwatch_log_group.application.arn}:*"
      },
      # Pull the application image.
      #
      # This was missing, and the deployment reached the point of logging in
      # to the registry before anything noticed: the role could read every
      # secret and write every log, and could not fetch the one artifact it
      # exists to run. The token call is account-wide by API design; the layer
      # reads are scoped to this repository.
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = "arn:aws:ecr:${var.region}:${data.aws_caller_identity.current.account_id}:repository/quantify"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "${local.name}-app"
  role = aws_iam_role.app.name
}
