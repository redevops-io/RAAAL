provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "redevops-io/RAAAL"
      # The pilot boundary, on every resource. An operator reading the console
      # rather than the runbook should still learn that nothing here is
      # licensed market data.
      DataPolicy = "SYNTHETIC_ONLY"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# Reads CLOUDFLARE_API_TOKEN from the environment. Deliberately not a
# variable: a variable is recorded in the plan file and in state, and this
# token can create DNS records for every zone the account owns.
provider "cloudflare" {}
