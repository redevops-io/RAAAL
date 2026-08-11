# A bucket the Ansible SSM connection uses to move files to the host.
#
# `community.aws.aws_ssm` does not send file content down the Session Manager
# channel: it stages each file in S3 and has the host fetch it. Without a
# bucket the connection fails during fact gathering with a null-pointer error
# that names nothing about S3, which is a long way from the cause.
#
# It is declared here rather than reused from the state bucket. State and
# deploy scratch have different lifetimes, different blast radii and different
# readers, and putting transient transfers in the bucket that holds state is
# how a cleanup rule eventually deletes something that mattered.

resource "aws_s3_bucket" "deploy_transfer" {
  bucket        = "${local.name}-deploy-transfer-${data.aws_caller_identity.current.account_id}"
  force_destroy = true

  tags = { Name = "${local.name}-deploy-transfer" }
}

resource "aws_s3_bucket_public_access_block" "deploy_transfer" {
  bucket                  = aws_s3_bucket.deploy_transfer.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "deploy_transfer" {
  bucket = aws_s3_bucket.deploy_transfer.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Transfers are scratch. Anything still here tomorrow is debris from a failed
# run, and debris in a bucket that briefly holds rendered configuration is
# worth expiring rather than accumulating.
resource "aws_s3_bucket_lifecycle_configuration" "deploy_transfer" {
  bucket = aws_s3_bucket.deploy_transfer.id

  rule {
    id     = "expire-transfers"
    status = "Enabled"

    filter {}

    expiration {
      days = 1
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

resource "aws_iam_role_policy" "deploy_transfer" {
  name = "deploy-transfer"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.deploy_transfer.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = aws_s3_bucket.deploy_transfer.arn
      },
    ]
  })
}
