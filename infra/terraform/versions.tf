terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Bootstrapped separately — see backend.hcl.example. State holds the RDS
  # endpoint and the secret ARNs, so it is not a repository artifact.
  backend "s3" {}
}
