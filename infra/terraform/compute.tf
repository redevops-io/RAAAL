# One EC2 instance in a private subnet, running the Docker Compose stack.
#
# `user_data` installs Docker and the SSM agent and stops. It does not deploy
# the application: cloud-init runs once, silently, at boot, and a deployment
# step that lives there cannot be re-run, cannot report what it did, and
# cannot be tested. Ansible owns everything after the host exists.

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.private[0].id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name

  # No key pair. Access is SSM Session Manager; a key pair here is a private
  # key somebody has to hold, distribute and eventually rotate.
  associate_public_ip_address = false

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 2 # containers reach the metadata service
  }

  user_data_replace_on_change = true
  user_data                   = <<-EOT
    #!/bin/bash
    set -euxo pipefail

    # Only what Ansible needs in order to connect and take over: Docker, the
    # SSM agent, and Python. Application configuration is a deploy-time
    # concern, and cloud-init is a poor place to keep anything you may need to
    # run twice.
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y ca-certificates curl gnupg python3 python3-pip unzip

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker

    snap install amazon-ssm-agent --classic || true
    systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service || true
  EOT

  tags = {
    Name = "${local.name}-app"
    # The Ansible dynamic inventory selects on this tag.
    AnsibleGroup = "quantify"
  }

  depends_on = [aws_nat_gateway.main]
}
