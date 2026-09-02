# The cluster the service split will run on. EKS Auto Mode, in the VPC that
# already exists.
#
# **Auto Mode rather than EKS plus self-managed Karpenter.** It owns node
# provisioning and consolidation, the load-balancer and storage integrations,
# and node repair. This project has one operator; a control plane whose node
# lifecycle is somebody's ongoing job is a control plane that will drift while
# they are doing something else.
#
# **The existing VPC, not a new one.** Networking here works, has a NAT
# gateway, two private and two public subnets across two availability zones,
# and an RDS instance the application already reaches. Redesigning it as part
# of a migration would mean diagnosing a new cluster and new networking at the
# same time, and the first failure would belong to both.
#
# **Opt-in per environment.** `enable_kubernetes` defaults to false, so a
# routine application deploy cannot create a cluster as a side effect — and,
# once it is set in an environment's tfvars, every later apply keeps it rather
# than destroying it because somebody forgot a flag. Turning it on is a diff in
# a reviewed file.
#
# Nothing here deploys a workload. The cluster is the first step and its own
# claim: it exists once it has been provisioned and probed, not once this file
# has been written.

data "aws_caller_identity" "deployer" {}

locals {
  eks_name = "${local.name}-eks"

  # Auto Mode splits what used to be one broad policy. Each is the managed
  # policy AWS names for the capability, attached rather than inlined so an
  # audit reads them by name.
  eks_cluster_policies = var.enable_kubernetes ? {
    cluster       = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
    compute       = "arn:aws:iam::aws:policy/AmazonEKSComputePolicy"
    block_storage = "arn:aws:iam::aws:policy/AmazonEKSBlockStoragePolicy"
    load_balancing = "arn:aws:iam::aws:policy/AmazonEKSLoadBalancingPolicy"
    networking    = "arn:aws:iam::aws:policy/AmazonEKSNetworkingPolicy"
  } : {}

  # Deliberately minimal. Nodes pull images and join; they are not how a
  # workload reaches AWS. Each service gets its own role through Pod Identity,
  # so a pod that should not reach S3 cannot do so by borrowing the node's
  # permissions — which is the same boundary the import graph already draws,
  # expressed where it cannot be bypassed by writing different Python.
  eks_node_policies = var.enable_kubernetes ? {
    worker   = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodeMinimalPolicy"
    registry = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly"
  } : {}
}

resource "aws_iam_role" "eks_cluster" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "${local.eks_name}-cluster"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = ["eks.amazonaws.com"] }
      Action = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster" {
  for_each   = local.eks_cluster_policies
  role       = aws_iam_role.eks_cluster[0].name
  policy_arn = each.value
}

resource "aws_iam_role" "eks_node" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "${local.eks_name}-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node" {
  for_each   = local.eks_node_policies
  role       = aws_iam_role.eks_node[0].name
  policy_arn = each.value
}

# The load-balancer discovery tags are on the subnets themselves, in
# `networking.tf`. They were `aws_ec2_tag` resources here first, which fought
# the subnets' own `tags` attribute: one resource added the tag and the other
# removed it as drift, so every plan showed the same four updates forever.

resource "aws_eks_cluster" "main" {
  count    = var.enable_kubernetes ? 1 : 0
  name     = local.eks_name
  role_arn = aws_iam_role.eks_cluster[0].arn

  # API, not the aws-auth ConfigMap. Access is then IAM and terraform rather
  # than a map somebody edits in the cluster, which is the version that drifts
  # from what the account actually permits.
  access_config {
    authentication_mode = "API"
  }

  # Auto Mode supplies its own. The self-managed set would be a second,
  # conflicting answer to the same question.
  bootstrap_self_managed_addons = false

  compute_config {
    enabled       = true
    # Which AWS-managed pools provision nodes. ["system"] removes the small
    # general-purpose app pool (keeping a valid managed pool for kube-system
    # addons) so app workloads land on the custom `general-large` NodePool
    # (applied by ansible/services.yml, referencing this Auto Mode `default`
    # NodeClass). The API refuses [] while node_role_arn is set — the list must
    # be non-empty (enforced by the variable's validation).
    node_pools    = var.managed_node_pools
    node_role_arn = aws_iam_role.eks_node[0].arn
  }

  kubernetes_network_config {
    elastic_load_balancing {
      enabled = true
    }
  }

  storage_config {
    block_storage {
      enabled = true
    }
  }

  vpc_config {
    # Private only, and this was wrong in the first version.
    #
    # It listed both tiers on the reasoning that load balancers need the public
    # subnets. They do — but they are discovered by the
    # `kubernetes.io/role/elb` tags above, not from this list. What this list
    # decides is where *nodes* are placed, and Auto Mode duly placed one in a
    # public subnet, where it received no public IP because these subnets do
    # not auto-assign one. A public subnet routes 0.0.0.0/0 to the internet
    # gateway, and an instance with no public address cannot use it: the node
    # came up Ready with no egress whatsoever, and every image pull timed out.
    #
    # Nothing in `describe-cluster` showed it. Status ACTIVE, Auto Mode
    # enabled, node pool READY, a node provisioned and Ready, the pod
    # scheduled — and ImagePullBackOff. That gap is the whole reason a cluster
    # is not a claim until a workload has run on it.
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    # Public endpoint so an operator can probe the cluster without a bastion.
    # The *workloads* are not public — that is the ingress's business, and the
    # acceptance evidence has to show external traffic cannot reach evaluate or
    # data regardless of this setting.
    endpoint_public_access = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster,
    aws_iam_role_policy_attachment.eks_node,
  ]
}

# The deploying identity, as a cluster administrator. Without this the cluster
# exists and nobody can talk to it — the creator is no longer an implicit admin
# under API authentication mode.
resource "aws_eks_access_entry" "deployer" {
  count         = var.enable_kubernetes ? 1 : 0
  cluster_name  = aws_eks_cluster.main[0].name
  principal_arn = data.aws_caller_identity.deployer.arn
  type          = "STANDARD"
}

resource "aws_eks_access_policy_association" "deployer_admin" {
  count         = var.enable_kubernetes ? 1 : 0
  cluster_name  = aws_eks_cluster.main[0].name
  principal_arn = data.aws_caller_identity.deployer.arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  access_scope {
    type = "cluster"
  }

  depends_on = [aws_eks_access_entry.deployer]
}

output "eks_cluster_name" {
  description = "Empty until the cluster is provisioned, which is the point."
  value       = var.enable_kubernetes ? aws_eks_cluster.main[0].name : ""
}

output "eks_cluster_endpoint" {
  value = var.enable_kubernetes ? aws_eks_cluster.main[0].endpoint : ""
}

output "eks_kubeconfig_command" {
  description = "How to reach it, so the probe step needs no guessing."
  value = var.enable_kubernetes ? join(" ", [
    "aws eks update-kubeconfig --region", var.region,
    "--name", aws_eks_cluster.main[0].name,
  ]) : ""
}


# --- Pod Identity -----------------------------------------------------------
#
# Each service assumes its own role. The node role grants nothing beyond
# pulling images and joining, so a pod cannot reach AWS by borrowing the
# node's permissions — which is what makes this an enforcement boundary rather
# than a second copy of the import-graph rule.
#
# `quantify-data` owns the market-data store: it reads the database password
# and, later, the S3 bucket holding payloads. `quantify-evaluate` gets no role
# at all, and that absence is the point. Somebody who later writes code in the
# evaluator that reaches for a bucket will find no credentials to reach it
# with, which is a stronger guarantee than a test that reads imports.

resource "aws_iam_role" "pod_data" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "${local.eks_name}-pod-data"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy" "pod_data" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "market-data-store"
  role  = aws_iam_role.pod_data[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # The database password, by ARN rather than by wildcard. A service
        # permitted to read every secret in the account is one whose blast
        # radius is the account.
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.database_password.arn]
      },
      {
        # Read-only, and by object version. The market-data service fetches the
        # licensed vendor snapshot from S3 under the approved-vendor policy,
        # pinned to the object version the manifest records. GetObjectVersion is
        # the one that matters — the whole design refuses an unversioned read, so
        # a role that could only GetObject (latest) would defeat the pin. No
        # PutObject: this service reads snapshots, it does not write them
        # (provisioning is an operator action, not a pod capability).
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = [
          "arn:aws:s3:::${var.market_data_bucket}",
          "arn:aws:s3:::${var.market_data_bucket}/*",
        ]
      },
    ]
  })
}

resource "aws_eks_pod_identity_association" "data" {
  count           = var.enable_kubernetes ? 1 : 0
  cluster_name    = aws_eks_cluster.main[0].name
  namespace       = "quantify"
  service_account = "quantify-data"
  role_arn        = aws_iam_role.pod_data[0].arn
}

# `quantify-web` reads the licensed vendor snapshot from S3 too, because the
# pilot's evaluation runs inline in the web pod (run_boundary resolves the
# approved snapshot per request) rather than calling the data service for it.
# The original boundary — web/evaluate reach market data *through* quantify-data
# and hold no credentials of their own — did not survive that move: an inline
# read needs the pod's own credentials, and without them every figure came back
# "no market data". So the grant is deliberately the narrowest that restores it:
# read-only, versioned, this one bucket, and *not* the database-password secret
# `pod_data` also carries — the web pod gets its database URL from its env
# secret, not from this role.
#
# Created out of band first (an `aws eks create-pod-identity-association` while a
# full `terraform apply` was gated on a Cloudflare-tunnel credential); these
# resources describe what exists and must be `terraform import`ed before the
# next apply, or the apply will try to create a role and association that are
# already there.
resource "aws_iam_role" "pod_web" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "${local.eks_name}-pod-web"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy" "pod_web" {
  count = var.enable_kubernetes ? 1 : 0
  name  = "market-data-read"
  role  = aws_iam_role.pod_web[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      # Read-only and by object version, exactly as pod_data reads it — the pin
      # refuses an unversioned read, so GetObjectVersion is the one that matters.
      # No secret access: the web pod's role is market data and nothing else.
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:GetObjectVersion"]
      Resource = [
        "arn:aws:s3:::${var.market_data_bucket}",
        "arn:aws:s3:::${var.market_data_bucket}/*",
      ]
    }]
  })
}

resource "aws_eks_pod_identity_association" "web" {
  count           = var.enable_kubernetes ? 1 : 0
  cluster_name    = aws_eks_cluster.main[0].name
  namespace       = "quantify"
  service_account = "quantify-web"
  role_arn        = aws_iam_role.pod_web[0].arn
}

# The connection string, assembled where both halves are already known.
#
# Marked sensitive so terraform will not print it, and fetched by the
# deployment playbook with `no_log` so it reaches the cluster Secret without
# passing through a log, a variables file or a shell history. It is
# deliberately *not* part of `ansible_variables`: that output is written to a
# file on disk, and a password in it would be a password in a file nobody
# remembered to delete.
output "database_url" {
  description = "Postgres URL including the password. Never written to a file."
  sensitive   = true
  value = join("", [
    "postgresql://", aws_db_instance.main.username, ":",
    random_password.database.result, "@", aws_db_instance.main.endpoint,
    "/", aws_db_instance.main.db_name,
  ])
}

output "eks_pod_identity_roles" {
  description = "Which role each service assumes. Evaluate has none, on purpose."
  value = var.enable_kubernetes ? {
    "quantify-data"     = aws_iam_role.pod_data[0].arn
    "quantify-web"      = aws_iam_role.pod_web[0].arn
    "quantify-evaluate" = "none — it reaches market data through quantify-data"
  } : {}
}


# The database, reachable from the cluster as well as from the EC2 host.
#
# Found by probing rather than by reading: `quantify-data` came up 1/1 Ready,
# answered /health and /version, and hung on the first request that touched
# PostgreSQL — because the pod's traffic arrives from the cluster security
# group and the database admits only the application host's. A pod that is
# Ready and cannot reach its database is exactly the shape `readinessProbe`
# cannot see: /health asks whether the process is serving, not whether it can
# do the thing it exists for.
#
# Added alongside the existing rule rather than replacing it. Both paths run
# during the migration, and removing the EC2 host's access to make a point
# about the new one would take production down to prove the cluster works.
resource "aws_vpc_security_group_ingress_rule" "database_from_cluster" {
  count                        = var.enable_kubernetes ? 1 : 0
  security_group_id            = aws_security_group.database.id
  description                  = "PostgreSQL from the EKS cluster"
  referenced_security_group_id = aws_eks_cluster.main[0].vpc_config[0].cluster_security_group_id
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
}
