# Inputs. Anything a deployment must decide is here; anything the pilot has
# already decided is a validated default, so a wrong value fails `terraform
# plan` rather than the application's startup preflight.

variable "project" {
  description = "Name prefix for every resource."
  type        = string

  # Not "quantify-test". Every name is built as "${project}-${environment}",
  # so that default produced `quantify-test-test/database-password` and a log
  # group of `/quantify-test/test/application` for anyone who did not override
  # it. The example tfvars happened to set it correctly, which is exactly how
  # a bad default survives review.
  default = "quantify"
}

variable "environment" {
  description = "Environment name. The pilot is a test deployment."
  type        = string
  default     = "test"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

# --- network ---------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR for the VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "operator_cidrs" {
  description = <<-EOT
    CIDRs permitted to reach the load balancer. Defaults to nobody.

    A closed pilot is closed. The workspace is behind basic auth, but an
    allowlist here is the difference between "a credential stands between the
    internet and the pilot" and "the internet cannot reach the pilot at all".
    Set it to your own address plus the pilot users'; set it to
    ["0.0.0.0/0"] deliberately, and know that you did.
  EOT
  type        = list(string)
  default     = []
}

# --- dns and tls -----------------------------------------------------------

variable "domain_name" {
  description = "Fully-qualified hostname the pilot is served on."
  type        = string
}

variable "hosted_zone_id" {
  description = "Route 53 zone ID that owns domain_name."
  type        = string
}

# --- compute ---------------------------------------------------------------

variable "instance_type" {
  description = "EC2 instance type for the application host."
  type        = string
  default     = "t3.small"
}

variable "root_volume_gb" {
  description = "Root EBS volume size. Holds the image and the trace database."
  type        = number
  default     = 30
}

variable "application_image" {
  description = <<-EOT
    The application image, pinned by digest.

    A tag is a moving target: `:latest` resolves to whatever was pushed last,
    so the acceptance record would name a configuration that no longer
    describes what is running. The build identity the preflight refuses to
    start without is only meaningful if the image it describes is fixed.
  EOT
  type        = string

  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.application_image))
    error_message = "Pin the image by digest: registry/name@sha256:<64 hex>. A tag is not a deployment identity."
  }
}

variable "registry_host" {
  description = "Registry to authenticate against. Empty for a public image."
  type        = string
  default     = ""
}

# --- database --------------------------------------------------------------

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage_gb" {
  description = "RDS allocated storage."
  type        = number
  default     = 20
}

variable "db_backup_retention_days" {
  description = <<-EOT
    Automated backup retention. Must be at least 1.

    Zero disables automated backups entirely, which is a valid RDS setting and
    an invalid pilot: the runbook's restore drill has nothing to restore from,
    and the first person to discover that is whoever needed it.
  EOT
  type        = number
  default     = 7

  validation {
    condition     = var.db_backup_retention_days >= 1
    error_message = "Backups must be retained. The restore drill is not optional for the pilot."
  }
}

variable "db_multi_az" {
  description = "Single-AZ is adequate for the test deployment."
  type        = bool
  default     = false
}

# --- application configuration --------------------------------------------

variable "pilot_data_policy" {
  description = <<-EOT
    The market-data boundary, enforced at runtime by the application.

    SYNTHETIC_ONLY until all six vendor licensing questions are resolved and
    recorded. Declared here as well as in the environment file so that
    changing it is a reviewed infrastructure change rather than an edit to a
    file on a host.
  EOT
  type        = string
  default     = "SYNTHETIC_ONLY"

  validation {
    condition     = var.pilot_data_policy == "SYNTHETIC_ONLY"
    error_message = "The pilot is synthetic-only until the six licensing questions are resolved and recorded. Change this deliberately, in a commit."
  }
}

variable "parser_mode" {
  description = "Declared, never inferred. Production refuses an unset mode."
  type        = string
  default     = "MODEL_ASSISTED"

  validation {
    condition     = contains(["MODEL_ASSISTED", "DETERMINISTIC"], var.parser_mode)
    error_message = "parser_mode must be MODEL_ASSISTED or DETERMINISTIC."
  }
}

variable "parser_model" {
  description = "Pinned exactly. An unpinned model changes what a description means."
  type        = string
  default     = "claude-sonnet-5"
}

variable "parser_prompt_version" {
  description = "Prompt revision pinned independently of the build. Empty uses the image default."
  type        = string
  default     = ""
}

variable "parser_fallback" {
  description = <<-EOT
    REFUSE or EXPLICIT_DETERMINISTIC.

    REFUSE for the pilot: a silent fallback hands two users different products
    under one deployment and tells neither.
  EOT
  type        = string
  default     = "REFUSE"

  validation {
    condition     = contains(["REFUSE", "EXPLICIT_DETERMINISTIC"], var.parser_fallback)
    error_message = "parser_fallback must be REFUSE or EXPLICIT_DETERMINISTIC."
  }
}

variable "trace_retention_days" {
  description = "Telemetry retention, applied by the purge cron job."
  type        = number
  default     = 90
}

# --- build identity --------------------------------------------------------

variable "build_commit" {
  description = "Git commit the image was built from. The preflight refuses without it."
  type        = string
}

variable "build_release_ref" {
  description = "Release ref or branch name."
  type        = string
}

variable "build_snapshot_id" {
  description = "Identifier of the synthetic market-data snapshot."
  type        = string
}

# --- monitoring ------------------------------------------------------------

variable "alert_email" {
  description = "Address that receives CloudWatch alarms and budget notices."
  type        = string
}

variable "monthly_budget_usd" {
  description = <<-EOT
    Monthly AWS budget. Alarms at 80% and 100% of it.

    This bounds infrastructure spend only. It does not bound model spend —
    Gate 8 is open, so nothing in this code caps what a pilot user can cause
    the parser to bill. Set a provider-side budget alert as well; the runbook
    lists it as required before the first invitation.
  EOT
  type        = number
  default     = 150
}
