# Which image this deployment runs, resolved from the commit it deploys.
#
# `build_commit` and `application_image` used to be two independent variables.
# Nothing tied them together, and on 2026-08-12 they drifted: the run declared
# 909264c and `application_image` still held a digest from an earlier build, so
# the host pulled the older image and the process stamped it with the newer
# commit. The container then reported a revision whose code it was not running,
# and served the licence notice of a commit two months behind the one it named.
#
# Every check in the deploy path passed. The Ansible assertion compared the
# startup proof against `quantify_build_commit` — and the proof takes its commit
# from the same variable, so it was the deployment agreeing with itself. What
# caught it was `verify_deployment_identity.py`, which asks the *running*
# service for its source offer: the offer resolved to the repository root
# instead of a revision, because the code serving it predated the change that
# made the offer name one.
#
# So the fix is not another assertion. It is removing the second variable: the
# digest is now looked up from the tag the build pushed, and there is one fact —
# the commit — from which the image follows.

data "aws_ecr_image" "app" {
  count = var.application_image == "" ? 1 : 0

  repository_name = var.ecr_repository
  # The short commit, which is what the build tags the image with. A lookup
  # that finds nothing fails the plan, which is the right outcome: deploying a
  # commit whose image was never pushed should stop here rather than silently
  # resolve to whatever else is in the repository.
  image_tag = substr(var.build_commit, 0, 7)
}

locals {
  # Pinned by digest either way. The digest is what identifies the bytes; the
  # tag is only how this configuration found them, and is not what is deployed.
  application_image = (
    var.application_image != ""
    ? var.application_image
    : "${var.registry_host}/${var.ecr_repository}@${data.aws_ecr_image.app[0].image_digest}"
  )
}
