# An *internal* load balancer. Nothing outside the VPC can reach it.
#
# TLS terminates at Cloudflare's edge and traffic arrives through the tunnel,
# so there is no ACM certificate and no public listener here. The ALB stays in
# the path for what it uniquely provides: a target-group health check on
# /health/ready and a 5xx metric, both of which the CloudWatch alarms read.
# The tunnel reports nothing to AWS.
#
# **Caddy is still in the stack too**, and for a different reason: it holds
# the basic-auth rule on /workspace/*, which is the entire access control for
# a pilot with no user accounts. Neither an ALB nor a Cloudflare tunnel
# provides it, and `deploy/acceptance.py` checks that it is there.
#
#   browser -> Cloudflare (TLS) -> cloudflared -> ALB :80 -> Caddy -> api :8000

resource "aws_lb" "main" {
  name               = substr("${local.name}-alb", 0, 32)
  load_balancer_type = "application"
  internal           = true
  subnets            = aws_subnet.private[*].id
  security_groups    = [aws_security_group.alb.id]

  drop_invalid_header_fields = true
  enable_deletion_protection = false

  # A pilot user describing a retirement plan can sit on a slow form for a
  # while; the default 60s is short enough to cut a legitimate save.
  idle_timeout = 120

  tags = { Name = local.name }
}

resource "aws_lb_target_group" "app" {
  name        = substr("${local.name}-app", 0, 32)
  port        = 80
  protocol    = "HTTP"
  target_type = "instance"
  vpc_id      = aws_vpc.main.id

  health_check {
    enabled = true
    # Readiness, not liveness. `/health/live` reports a process that exists;
    # this instance may be refusing to serve because its database is at the
    # wrong migration head, and the load balancer must not send a user to it.
    path                = "/health/ready"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  # The pilot has no user accounts and no session state worth pinning, but a
  # confirmation flow spans several requests against one instance today.
  stickiness {
    type    = "lb_cookie"
    enabled = false
  }

  deregistration_delay = 30
}

resource "aws_lb_target_group_attachment" "app" {
  target_group_arn = aws_lb_target_group.app.arn
  target_id        = aws_instance.app.id
  port             = 80
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # Plaintext, deliberately. This listener is not reachable from the internet:
  # the load balancer is internal, and the only client is cloudflared inside
  # the same VPC. TLS is terminated at Cloudflare's edge, and the hop from the
  # edge to here runs inside the tunnel.
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
