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

# Subnet tags the load-balancer integration discovers by. Applied as their own
# resources rather than by editing the subnets, so this migration adds tags
# without taking ownership of networking it did not create.
resource "aws_ec2_tag" "public_elb" {
  count       = var.enable_kubernetes ? length(aws_subnet.public) : 0
  resource_id = aws_subnet.public[count.index].id
  key         = "kubernetes.io/role/elb"
  value       = "1"
}

resource "aws_ec2_tag" "private_elb" {
  count       = var.enable_kubernetes ? length(aws_subnet.private) : 0
  resource_id = aws_subnet.private[count.index].id
  key         = "kubernetes.io/role/internal-elb"
  value       = "1"
}

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
    node_pools    = ["general-purpose"]
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
    # Both tiers: pods run private, load balancers need the public ones.
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
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
