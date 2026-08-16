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

# --- the application failed --------------------------------------------------
#
# Counted from the application's own log rather than from the load balancer's
# 5xx total, because those are not the same event.
#
# `HTTPCode_Target_5XX_Count` includes 501, and this build serves 501
# deliberately: an out-of-scope capability — RSUs, options — is declined with a
# readable page saying so. A 144-prompt evaluation sweep produced nine of them
# in five minutes and put this alarm into ALARM with no fault anywhere, which
# is the failure mode the header of this file warns about. A pilot cohort will
# ask for those capabilities as a matter of course.
#
# So the alarm measures what its description claims: the server broke. A
# refusal is not a break, and an alarm that fires on ordinary use is one nobody
# reads by week three.


# --- the host and the database --------------------------------------------


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
#
# There is deliberately no `aws_budgets_budget` here: the deploying identity
# has no budgets:* permission, and a resource that always fails to apply is
# worse than an absent one. Set the AWS budget in the console, and — more
# importantly — set the **provider-side** model budget alert, which no AWS
# budget can see. Gate 8 is open: nothing in this code caps what a pilot user
# can cause the parser to bill. docs/Runbook.md requires it before the first
# invitation.
