# Three secrets. Terraform creates the containers and generates the database
# password; the model key and the workspace credential are placed by hand.
#
# `ANTHROPIC_API_KEY` is deliberately not a Terraform variable. A variable
# ends up in the plan file, in state, and in whatever shell history typed it.
# Terraform makes the empty secret and the runbook says to put the value in
# with the CLI, so the key exists in exactly one place that is designed to
# hold it.

resource "random_password" "database" {
  length = 40

  # Restricted to characters that survive being placed in a URL userinfo
  # field unencoded, because the deployed connection string is built by string
  # substitution on the host.
  #
  # RDS itself only rejects '/', '@', '"' and space. That is not the binding
  # constraint: '#', '%', '?' and ':' are all legal RDS passwords and all
  # corrupt `postgresql://user:PASSWORD@host/db` — '#' starts a fragment, '%'
  # begins a percent-escape, '?' starts a query. A generated password is
  # different on every apply, so the wider set fails on *some* deployments and
  # not others, which is the shape of defect that gets diagnosed as
  # "networking" for a day.
  #
  # These remain: sub-delims and unreserved characters, which RFC 3986 permits
  # in userinfo verbatim.
  override_special = "!$&*()-_=+"
}

resource "aws_secretsmanager_secret" "database_password" {
  name        = "${local.name}/database-password"
  description = "RDS master password for the pilot database"

  # A test deployment gets torn down and rebuilt; a 30-day deletion window
  # means the next apply collides with the name of a secret that still exists.
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "database_password" {
  secret_id     = aws_secretsmanager_secret.database_password.id
  secret_string = random_password.database.result
}

resource "aws_secretsmanager_secret" "model_api_key" {
  name        = "${local.name}/model-api-key"
  description = "Hosted parser API key for whichever provider `parser_provider` declares. Value set out of band — see infra/README.md."

  recovery_window_in_days = 0
}

# Created empty on purpose. `terraform apply` must not be the thing that knows
# this value. Ansible fails with a readable message if it is still empty, so
# the omission surfaces at deploy time rather than as a startup refusal.
resource "aws_secretsmanager_secret_version" "model_api_key" {
  secret_id     = aws_secretsmanager_secret.model_api_key.id
  secret_string = jsonencode({ api_key = "" })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "workspace_basic_auth" {
  name        = "${local.name}/workspace-basic-auth"
  description = "Caddy basic-auth credential for /workspace/*"

  recovery_window_in_days = 0
}

# The username, and a bcrypt hash of the password. The plaintext is never
# stored: Caddy needs only the hash, so the deployment never holds a value
# that could be replayed against anything else.
#
#   caddy hash-password --plaintext 'the-pilot-password'
resource "aws_secretsmanager_secret_version" "workspace_basic_auth" {
  secret_id     = aws_secretsmanager_secret.workspace_basic_auth.id
  secret_string = jsonencode({ username = "pilot", password_hash = "" })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# --- cloudflare tunnel -----------------------------------------------------

# The tunnel's shared secret, and the connector token derived from it.
#
# Unlike the model key and the workspace credential, this one *is* generated
# by Terraform: it is not a value a human chose or holds anywhere else, and
# the tunnel resource needs it at create time. It reaches the host through
# Secrets Manager like everything else, never through a variable or an output.
resource "random_password" "tunnel" {
  length  = 64
  special = false
}

resource "aws_secretsmanager_secret" "tunnel_token" {
  name        = "${local.name}/cloudflare-tunnel-token"
  description = "Connector token for the Cloudflare tunnel"

  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "tunnel_token" {
  secret_id = aws_secretsmanager_secret.tunnel_token.id
  secret_string = jsonencode({
    token = cloudflare_zero_trust_tunnel_cloudflared.pilot.tunnel_token
  })
}


# Zitadel's masterkey, which encrypts everything it stores.
#
# Created empty and never known to Terraform, the same rule the model key
# follows. Losing this value is worse than losing a password: it decrypts the
# provider's own secrets, so a rotation is a migration rather than an edit.
# Ansible refuses to deploy the provider while it is empty.
resource "aws_secretsmanager_secret" "identity_masterkey" {
  count = var.identity_domain_name == "" ? 0 : 1

  name        = "${local.name}/identity-masterkey"
  description = "Zitadel masterkey. Exactly 32 bytes. Value set out of band — see infra/README.md."

  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "identity_masterkey" {
  count = var.identity_domain_name == "" ? 0 : 1

  secret_id     = aws_secretsmanager_secret.identity_masterkey[0].id
  secret_string = jsonencode({ masterkey = "" })

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# The identity provider's first administrator.
#
# Zitadel creates one at first boot whether or not it is told to, and its
# default is `zitadel-admin@zitadel.<domain>` with the password `Password1!`
# and a change-required flag. That is a documented default on a login page
# facing the internet: whoever reaches the console first claims the instance,
# and the change-required flag helps them rather than us.
#
# Generated here rather than left empty, unlike the masterkey. The masterkey
# cannot be rotated without a migration, so it is worth the manual step; this
# is an ordinary password on one account, and a generated one that nobody ever
# has to type beats a documented one everybody knows.
resource "random_password" "identity_admin" {
  count = var.identity_domain_name == "" ? 0 : 1

  length = 32
  # Zitadel's default password policy requires all four classes. Restricted to
  # symbols that survive an env file, an envsubst pass and a shell export
  # without quoting games — `$` in particular would be substituted on the way
  # through and the account would be created with a password nobody holds.
  override_special = "!@#%^*-_=+"
  min_upper        = 2
  min_lower        = 2
  min_numeric      = 2
  min_special      = 2
}

resource "aws_secretsmanager_secret" "identity_admin" {
  count = var.identity_domain_name == "" ? 0 : 1

  name        = "${local.name}/identity-admin-password"
  description = "Password for the identity provider's first administrator."

  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "identity_admin" {
  count = var.identity_domain_name == "" ? 0 : 1

  secret_id     = aws_secretsmanager_secret.identity_admin[0].id
  secret_string = jsonencode({ password = random_password.identity_admin[0].result })
}


# Outbound mail for the identity provider.
#
# Not optional the moment registration is open: ZITADEL requires a verified
# email address before an account can sign in, and the only way to verify one
# is to send to it. A provider with no mail sender lets people register and
# then refuses to let them in — which is a worse outcome than closed
# registration, because it looks like it worked.
#
# Created empty, like the model key and the masterkey. The token belongs to a
# third-party account and Terraform has no business knowing it; Ansible refuses
# to configure mail while this is blank, and the provider simply has no sender
# until somebody sets it.
resource "aws_secretsmanager_secret" "smtp" {
  count = var.identity_domain_name == "" ? 0 : 1

  name        = "${local.name}/smtp-credentials"
  description = "SMTP credentials for identity notifications. Value set out of band — see infra/README.md."

  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "smtp" {
  count = var.identity_domain_name == "" ? 0 : 1

  secret_id = aws_secretsmanager_secret.smtp[0].id
  secret_string = jsonencode({
    host   = "", port = 587, user = "", password = "",
    sender = "", sender_name = "Quantify",
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}
