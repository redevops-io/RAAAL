# The public entry point. TLS terminates here with an ACM certificate.
#
# **Caddy is still in the stack.** The load balancer replaces it as the TLS
# endpoint, not as the access control: an ALB has no HTTP basic auth, and the
# basic-auth rule on `/workspace/*` is the entire thing standing between the
# pilot workspace and anyone who reaches the hostname. Removing Caddy here
# would silently delete a control that `deploy/acceptance.py` checks for.
#
#   internet -> ALB :443 (ACM) -> EC2 :80 (Caddy, basic auth) -> api :8000

resource "aws_lb" "main" {
  name               = substr("${local.name}-alb", 0, 32)
  load_balancer_type = "application"
  internal           = false
  subnets            = aws_subnet.public[*].id
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

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.main.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}
