# Alarms on the conditions the runbook has procedures for, and nothing else.
#
# An alarm nobody acts on is noise, and noise is how a monitoring system stops
# being read at all. Each of these maps to a section of docs/Runbook.md.

resource "aws_cloudwatch_log_group" "application" {
  name              = "/${var.project}/${var.environment}/application"
  retention_in_days = 30
}

resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- the deployment is not serving ----------------------------------------

# The most important alarm here. The target group is unhealthy whenever
# `/health/ready` is not 200, which is exactly the set of conditions the
# application refuses to serve under: migration mismatch, schema drift,
# unreachable database, unobservable build, undeclared parser. A refusal is
# correct behaviour and still needs somebody told.
resource "aws_cloudwatch_metric_alarm" "unhealthy_target" {
  alarm_name          = "${local.name}-not-ready"
  alarm_description   = "The application is not answering /health/ready. See docs/Runbook.md - Startup refusals."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "breaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "target_5xx" {
  alarm_name          = "${local.name}-5xx"
  alarm_description   = "The application returned 5xx. Every one carries a correlation id; grep the log for it."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# --- the host and the database --------------------------------------------

resource "aws_cloudwatch_metric_alarm" "instance_cpu" {
  alarm_name          = "${local.name}-instance-cpu"
  alarm_description   = "Sustained CPU on the application host."
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions    = { InstanceId = aws_instance.app.id }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${local.name}-database-cpu"
  alarm_description   = "Sustained CPU on the pilot database."
  namespace           = "AWS/RDS"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions    = { DBInstanceIdentifier = aws_db_instance.main.id }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

# Storage, in bytes, at 20% of what was allocated. Storage exhaustion on the
# pilot database is the failure that stops writes while reads keep working —
# so the application looks fine until somebody tries to save a plan.
resource "aws_cloudwatch_metric_alarm" "database_storage" {
  alarm_name          = "${local.name}-database-storage"
  alarm_description   = "Free storage on the pilot database is below 20% of allocated."
  namespace           = "AWS/RDS"
  metric_name         = "FreeStorageSpace"
  statistic           = "Minimum"
  period              = 300
  evaluation_periods  = 2
  threshold           = var.db_allocated_storage_gb * 1024 * 1024 * 1024 * 0.2
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions    = { DBInstanceIdentifier = aws_db_instance.main.id }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

# --- spend -----------------------------------------------------------------

# Infrastructure only. This does not and cannot bound model spend: Gate 8 is
# open, nothing in the application caps what a pilot user can cause the parser
# to bill, and that bill arrives from Anthropic rather than AWS. Set the
# provider-side budget alert as well — the runbook requires it before the
# first invitation.
resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$${var.project}"]
  }

  dynamic "notification" {
    for_each = [80, 100]

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
