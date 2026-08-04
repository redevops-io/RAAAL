# Outputs Ansible and the acceptance run need. No secret values: the database
# password and the model key are read from Secrets Manager by the host at
# deploy time, and an output would copy them into state readers' terminals.

output "application_url" {
  description = "Where the pilot is served. Point deploy/acceptance.py here."
  value       = "https://${var.domain_name}"
}

output "instance_id" {
  description = "EC2 instance. `aws ssm start-session --target <id>`."
  value       = aws_instance.app.id
}

output "database_endpoint" {
  description = "RDS endpoint. Not publicly reachable."
  value       = aws_db_instance.main.address
}

output "database_url_template" {
  description = <<-EOT
    The connection string with the password elided. Ansible assembles the real
    one on the host from Secrets Manager; this exists so an operator can see
    the shape without the value.
  EOT
  value       = "postgresql://quantify:<password>@${aws_db_instance.main.address}:5432/quantify"
}

output "secret_ids" {
  description = "Secrets the host reads at deploy time."
  value = {
    database_password    = aws_secretsmanager_secret.database_password.name
    model_api_key        = aws_secretsmanager_secret.model_api_key.name
    workspace_basic_auth = aws_secretsmanager_secret.workspace_basic_auth.name
  }
}

output "log_group" {
  description = "Where the application's operator channel lands."
  value       = aws_cloudwatch_log_group.application.name
}

output "alerts_topic" {
  description = "Confirm the email subscription before relying on any alarm."
  value       = aws_sns_topic.alerts.arn
}

# The Ansible playbook reads these rather than being told them, so the
# deployed configuration cannot disagree with the planned one.
output "ansible_variables" {
  description = "Feed to ansible-playbook with -e @<file>."
  value = {
    quantify_region                = var.region
    quantify_instance_id           = aws_instance.app.id
    quantify_database_host         = aws_db_instance.main.address
    quantify_image                 = var.application_image
    quantify_registry_host         = var.registry_host
    quantify_domain                = var.domain_name
    quantify_data_policy           = var.pilot_data_policy
    quantify_parser_mode           = var.parser_mode
    quantify_parser_model          = var.parser_model
    quantify_parser_prompt_version = var.parser_prompt_version
    quantify_parser_fallback       = var.parser_fallback
    quantify_trace_retention_days  = var.trace_retention_days
    quantify_build_commit          = var.build_commit
    quantify_build_release_ref     = var.build_release_ref
    quantify_build_snapshot_id     = var.build_snapshot_id
    quantify_log_group             = aws_cloudwatch_log_group.application.name
    quantify_secret_database       = aws_secretsmanager_secret.database_password.name
    quantify_secret_model_key      = aws_secretsmanager_secret.model_api_key.name
    quantify_secret_basic_auth     = aws_secretsmanager_secret.workspace_basic_auth.name
  }
}
