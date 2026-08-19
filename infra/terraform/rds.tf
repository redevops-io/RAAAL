# PostgreSQL 16. The major version is not a preference: the application's
# preflight refuses anything else, because 16 is what the test lane has been
# proven against.

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = local.name }
}

resource "aws_db_parameter_group" "main" {
  name   = "${local.name}-pg16"
  family = "postgres16"

  # `apply_method` is stated on both. Left off, AWS reports back the method it
  # chose, Terraform sees a value its configuration does not have, and every
  # future plan shows this group being modified in place. A diff that appears
  # on every plan and never means anything is how a team learns to skim plans.
  parameter {
    name         = "log_min_duration_statement"
    value        = "1000"
    apply_method = "immediate"
  }

  # Refuse unencrypted connections. The application reaches the database
  # across a private subnet, which is not the same as reaching it privately.
  # Static: it takes effect on reboot, not on apply.
  parameter {
    name         = "rds.force_ssl"
    value        = "1"
    apply_method = "pending-reboot"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_db_instance" "main" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage_gb
  max_allocated_storage = var.db_allocated_storage_gb * 4
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "quantify"
  username = "quantify"
  password = random_password.database.result
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.database.id]
  parameter_group_name   = aws_db_parameter_group.main.name

  # Not publicly accessible, in a private subnet, reachable only from the
  # application security group. Three independent reasons, because this is the
  # one that leaks personal financial data if it is wrong.
  publicly_accessible = false

  multi_az                = var.db_multi_az
  backup_retention_period = var.db_backup_retention_days
  backup_window           = "04:00-05:00"
  maintenance_window      = "sun:05:30-sun:06:30"
  copy_tags_to_snapshot   = true

  auto_minor_version_upgrade = true
  apply_immediately          = false

  performance_insights_enabled    = true
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  # Restored for the 2026-08-18 recreation on EKS.
  #
  # The decommission commit (fcdee25) turned this off so `terraform destroy`
  # could remove the instance, and said in as many words to put both guards
  # back when the environment is next created. This is that creation. The
  # instance holds pilot users' financial descriptions, so a `terraform
  # destroy` must refuse at the plan rather than proceed by default.
  deletion_protection = true

  # A test deployment still holds pilot users' financial descriptions. Taking
  # a final snapshot costs nothing and is the difference between a mistaken
  # destroy being recoverable and being final.
  #
  # Kept through the decommission on purpose. The instance goes; the data
  # remains restorable from `quantify-test-final-<timestamp>` until somebody
  # deletes that snapshot deliberately, which is a separate decision from
  # tearing down the infrastructure around it.
  skip_final_snapshot       = false
  final_snapshot_identifier = "${local.name}-final-${formatdate("YYYYMMDDhhmmss", timestamp())}"

  lifecycle {
    # Restored with this environment (see `deletion_protection` above).
    # `terraform destroy` will not remove this. Overriding is deliberate:
    # delete this block in a commit, or detach the instance from state. Both
    # leave a record; a `-target` flag typed at 3am does not.
    prevent_destroy = true

    # The snapshot name embeds a timestamp, which would otherwise show as a
    # diff on every plan and train everyone to ignore diffs on this resource.
    ignore_changes = [final_snapshot_identifier]
  }
}
