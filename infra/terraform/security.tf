# Three security groups in a chain, and an instance profile that can read
# exactly three secrets.
#
# The chain is the whole isolation story: the internet reaches only the load
# balancer, the load balancer reaches only the application, and the
# application is the only thing that reaches the database. Each rule names a
# security group rather than a CIDR, so widening one does not silently widen
# the next.

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Public entry point"
  vpc_id      = aws_vpc.main.id

  # No ingress rules here. They are separate resources below so that an empty
  # `operator_cidrs` produces a load balancer nothing can reach, rather than
  # an inline block that a default would quietly fill in.

  egress {
    description     = "To the application host"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  tags = { Name = "${local.name}-alb" }
}

resource "aws_vpc_security_group_ingress_rule" "alb_https" {
  for_each = toset(var.operator_cidrs)

  security_group_id = aws_security_group.alb.id
  description       = "HTTPS from an allowlisted operator or pilot user"
  cidr_ipv4         = each.value
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "alb_http_redirect" {
  for_each = toset(var.operator_cidrs)

  security_group_id = aws_security_group.alb.id
  description       = "HTTP, redirected to HTTPS"
  cidr_ipv4         = each.value
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
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
        Resource = [
          aws_secretsmanager_secret.database_password.arn,
          aws_secretsmanager_secret.model_api_key.arn,
          aws_secretsmanager_secret.workspace_basic_auth.arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = "${aws_cloudwatch_log_group.application.arn}:*"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "${local.name}-app"
  role = aws_iam_role.app.name
}
