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

# --- dns, tls and ingress ---------------------------------------------------
# Cloudflare owns both. There is no ACM certificate and no Route 53 record:
# `cloudflared` dials out from the host, so the pilot has no public listener.

variable "domain_name" {
  description = "Hostname the pilot is served on, e.g. quantify.club."
  type        = string
}

variable "workspace_domain_name" {
  description = <<-EOT
    Hostname the Portfolio Operations workspace SPA is served on, e.g.
    workspace.quantify.club. It has its own internal ALB (the quantify-workspace
    Ingress), reached through the same tunnel.

    Empty disables it: no DNS record and no tunnel rule, the state before the
    workspace was deployed. Set it only once the quantify-workspace Ingress exists,
    so the ALB the tunnel rule names has been provisioned.
  EOT
  type        = string
  default     = ""
}

variable "identity_domain_name" {
  description = <<-EOT
    Hostname the identity provider is served on, e.g. auth.quantify.club.

    Empty disables it: no DNS record, no tunnel rule, and the application
    declares no OIDC issuer, which is the configuration the pilot ran under for
    months. That has to stay expressible — a deployment with no accounts serves
    everybody, and one with a half-configured provider must serve nobody.

    Its own hostname rather than a path on the application's, because a token's
    issuer is part of the token: every JWT carries it as `iss` and the
    application checks it, so moving the provider later invalidates everything
    already issued.
  EOT
  type        = string
  default     = ""
}

variable "cloudflare_account_id" {
  description = "Cloudflare account that owns the tunnel."
  type        = string
}

variable "cloudflare_zone_id" {
  description = <<-EOT
    Zone ID for domain_name.

    Passed explicitly rather than read from CLOUDFLARE_ZONE_ID: that variable
    is commonly set to whichever zone was last worked on, and a DNS record
    written into the wrong zone is a mistake nobody notices until the name
    fails to resolve.
  EOT
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
    Optional override for the application image, pinned by digest.

    Normally empty. The image is resolved from `build_commit` — see
    `image.tf` — because supplying it separately is how a deployment comes to
    run one revision while declaring another.

    Left here for the case the resolution cannot serve: an image pushed under
    no commit tag, or a rollback to something the tag no longer points at.
    Overriding means asserting by hand that this image is that commit, and
    nothing can check it for you.
  EOT
  type        = string
  default     = ""

  validation {
    condition     = var.application_image == "" || can(regex("@sha256:[0-9a-f]{64}$", var.application_image))
    error_message = "Pin the image by digest: registry/name@sha256:<64 hex>. A tag is not a deployment identity."
  }
}

variable "ecr_repository" {
  description = "ECR repository holding the application image."
  type        = string
  default     = "quantify"
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

variable "market_data_bucket" {
  description = <<-EOT
    The private S3 bucket holding licensed vendor market-data snapshots.

    Created out of band by scripts/provision_market_data.py (which owns its
    versioning, encryption and TLS-only policy); named here only so the
    quantify-data pod's IAM role can be granted read-only, version-pinned access
    to it. Not the manifest's bucket reference — that stays env-provided so the
    name is not committed to an AGPL repository; this is the access grant.
  EOT
  type        = string
  default     = "quantify-club-market-data"
}

variable "pilot_data_policy" {
  description = <<-EOT
    The market-data boundary, enforced at runtime by the application.

    Was pinned to SYNTHETIC_ONLY until all six vendor licensing questions were
    resolved and recorded. They are, in data/licensing/market-data-licensing@1.yaml,
    so the approved policy is now a permitted value.

    Terraform cannot check that record — it cannot read the repository, and a
    condition here that tried would be checking a file on whoever's laptop ran
    the plan. So this list says which values are spellable, and the gate that
    actually decides is the application's: `approved_snapshot()` re-reads the
    record on every resolve and returns nothing if a single field is missing.
    Setting this variable does not grant access to vendor data; it declares
    which boundary the deployment is asking for.

    Declared here as well as in the environment file so that changing it is a
    reviewed infrastructure change rather than an edit to a file on a host.
  EOT
  type        = string
  default     = "SYNTHETIC_ONLY"

  validation {
    condition = contains([
      "SYNTHETIC_ONLY",
      "market-data-egress/pilot-vendor-approved@1",
    ], var.pilot_data_policy)
    error_message = "pilot_data_policy must be SYNTHETIC_ONLY or market-data-egress/pilot-vendor-approved@1. A value the application does not recognise resolves to no policy at all, which serves no figures."
  }
}

variable "parser_mode" {
  description = <<-EOT
    Declared, never inferred. Production refuses an unset mode.

    RUNTIME is the pilot interpreter — hosted reader, fusion, VerifiedIntent,
    `compile_intent` — and it was missing from this list, which meant the
    infrastructure could not express the mode the product was being built for.
    `/workspace/new` branches on it, so quantify.club served the legacy
    `compile_scenario` path and every surface behind RUNTIME was unreachable:
    the strategy selector, the clarification-convergence work, refusal by name
    from the capability manifest. Nothing failed. The deployment served a
    different program from the one under development and reported success.
  EOT
  type        = string
  default     = "MODEL_ASSISTED"

  validation {
    condition     = contains(["MODEL_ASSISTED", "DETERMINISTIC", "RUNTIME"], var.parser_mode)
    error_message = "parser_mode must be MODEL_ASSISTED, DETERMINISTIC or RUNTIME."
  }
}

variable "parser_provider" {
  description = <<-EOT
    Which hosted reader this deployment serves: ANTHROPIC or OPENAI.

    Declared rather than inferred from the model name. The secret in
    `model_api_key` is injected under whichever environment variable that
    provider reads, so the two must agree — a deployment naming an OpenAI model
    with an Anthropic key in the secret refuses at startup, which is the right
    outcome and a confusing one to debug from the model name alone.
  EOT
  type        = string
  default     = "ANTHROPIC"

  validation {
    condition     = contains(["ANTHROPIC", "OPENAI"], var.parser_provider)
    error_message = "parser_provider must be ANTHROPIC or OPENAI."
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

variable "enable_kubernetes" {
  description = <<-EOT
    Whether this environment runs the EKS Auto Mode cluster.

    Defaults to false so a routine application deploy cannot create a cluster
    as a side effect. Set it in an environment's tfvars rather than passing it
    on the command line: passed as a flag, the first apply that forgot it would
    destroy the cluster, and "remember the flag" is not a control.
  EOT
  type        = bool
  default     = false
}

variable "cluster_albs_ready" {
  description = <<-EOT
    Whether the EKS ingresses have provisioned their ALBs yet.

    The Cloudflare tunnel config in `cloudflare.tf` wires to the two ALBs the
    web and identity Ingresses create, and those do not exist until
    `services.yml` has deployed the workloads. On a from-scratch bring-up this
    is a second phase: apply once with this false to create the cluster and
    everything under it, deploy the workloads so the Ingresses provision their
    ALBs, then apply again with `-var cluster_albs_ready=true` to wire the
    tunnel to those ALBs and publish the DNS records.

    Defaults to false so a first apply cannot fail on a load balancer that does
    not exist yet. On a steady-state redeploy the ALBs already exist; set it
    true in the environment's tfvars once the cluster is standing.
  EOT
  type        = bool
  default     = false
}

