#!/usr/bin/env python3
"""Create the OIDC application the workspace logs in through, and a test user.

The EKS port of `archive/ec2-deployment/roles/quantify/templates/bootstrap-identity.py.j2`.
On EC2 this ran on the host beside the provider, reaching it on 127.0.0.1; on
the cluster the provider is a Service, so this runs on the deploy controller
against a `kubectl port-forward` to it, with the same Host header and the same
`X-Forwarded-Proto: https` the in-pod proxy would add.

It does three things, each idempotent by search so a re-run adopts what exists
rather than making a second of it (a second OIDC client would keep logins
working against a configuration nobody edits):

  * the `Quantify` project,
  * a public PKCE web OIDC application → prints its client id on stdout,
  * a pre-verified human test user for the ui-agent runs (created only; its
    password is not read back or printed).

A public client with PKCE, deliberately: a confidential client's secret is a
third credential to store and rotate for no gain, because the authorization
code is already bound to the request that started it by the code challenge.

Everything but the client id goes to stderr, so `... | tail -1` is the client id.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROJECT_NAME = "Quantify"
APPLICATION_NAME = "Quantify Workspace"


def _client(base_url: str, identity_host: str, pat: str):
    def call(path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {pat}",
                "Content-Type": "application/json",
                # Selects the identity site on the proxy and is the issuer host.
                "Host": identity_host,
                # The provider rejects a management call it considers insecure;
                # TLS genuinely terminates at the edge, and the proxy would add
                # this on the real path.
                "X-Forwarded-Proto": "https",
            },
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise SystemExit(
                f"The provider refused {path}: {error.code} {detail}\n"
                "A 401 means the PAT is not valid for this instance — it is "
                "written once, at first setup, so a token from an earlier "
                "database will not work. A 403 naming a Cloudflare error code is "
                "the edge answering, not the provider: this talks to a local "
                "port-forward, not the public name.")
    return call


def project(call) -> str:
    found = call("/management/v1/projects/_search",
                 {"queries": [{"nameQuery": {"name": PROJECT_NAME,
                                             "method": "TEXT_QUERY_METHOD_EQUALS"}}]})
    for existing in found.get("result") or []:
        return existing["id"]
    return call("/management/v1/projects", {"name": PROJECT_NAME})["id"]


def application(call, project_id: str, app_origin: str) -> str:
    found = call(f"/management/v1/projects/{project_id}/apps/_search",
                 {"queries": [{"nameQuery": {"name": APPLICATION_NAME,
                                             "method": "TEXT_QUERY_METHOD_EQUALS"}}]})
    for existing in found.get("result") or []:
        config = existing.get("oidcConfig") or {}
        if config.get("clientId"):
            return config["clientId"]

    created = call(
        f"/management/v1/projects/{project_id}/apps/oidc",
        {
            "name": APPLICATION_NAME,
            "redirectUris": [f"{app_origin}/auth/callback"],
            "postLogoutRedirectUris": [f"{app_origin}/"],
            "responseTypes": ["OIDC_RESPONSE_TYPE_CODE"],
            "grantTypes": ["OIDC_GRANT_TYPE_AUTHORIZATION_CODE",
                           "OIDC_GRANT_TYPE_REFRESH_TOKEN"],
            "appType": "OIDC_APP_TYPE_WEB",
            "authMethodType": "OIDC_AUTH_METHOD_TYPE_NONE",  # public + PKCE
            "devMode": False,  # on, and any redirect URI is accepted — decoration
            "accessTokenType": "OIDC_TOKEN_TYPE_JWT",
            "idTokenUserinfoAssertion": True,
        })
    client_id = created.get("clientId")
    if not client_id:
        raise SystemExit("The provider created the application but returned no "
                         "client id: " + json.dumps(created)[:500])
    return client_id


def test_user(call, email: str, password: str) -> str:
    """A pre-verified human, created if absent. Its email is marked verified so
    no SMTP sender is needed to let it sign in, which a test instance has none of."""
    found = call("/management/v1/users/_search",
                 {"queries": [{"userNameQuery": {"userName": email,
                                                 "method": "TEXT_QUERY_METHOD_EQUALS"}}]})
    for existing in found.get("result") or []:
        return f"exists ({existing.get('id')})"

    created = call(
        "/management/v1/users/human/_import",
        {
            "userName": email,
            "profile": {"firstName": "Pilot", "lastName": "Tester"},
            "email": {"email": email, "isEmailVerified": True},
            "password": password,
            "passwordChangeRequired": False,
        })
    return f"created ({created.get('userId')})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL of a port-forward to the identity Service")
    ap.add_argument("--identity-host", required=True, help="e.g. auth.quantify.club")
    ap.add_argument("--pat", help="the deploy-bootstrap machine PAT")
    ap.add_argument("--pat-file",
                    help="a file holding the PAT — preferred in-cluster, where "
                         "FIRSTINSTANCE writes it to a mounted path and passing "
                         "it as an argument would put a live credential in the "
                         "process table for anything in the pod to read")
    ap.add_argument("--app-origin", required=True, help="e.g. https://quantify.club")
    ap.add_argument("--test-email", required=True)
    ap.add_argument("--test-password", required=True)
    args = ap.parse_args()

    pat = args.pat
    if args.pat_file:
        with open(args.pat_file, encoding="utf-8") as handle:
            pat = handle.read().strip()
    if not pat:
        raise SystemExit("no PAT: pass --pat or point --pat-file at the file "
                         "FIRSTINSTANCE_PATPATH wrote. An empty PAT is a 401 on "
                         "the first call, which reads as the instance being wrong "
                         "rather than the credential being absent.")

    call = _client(args.url.rstrip("/"), args.identity_host, pat)

    project_id = project(call)
    print(f"project: {project_id}", file=sys.stderr)
    client_id = application(call, project_id, args.app_origin.rstrip("/"))
    print(f"application client id: {client_id}", file=sys.stderr)
    status = test_user(call, args.test_email, args.test_password)
    print(f"test user {args.test_email}: {status}", file=sys.stderr)

    # stdout is the client id and nothing else.
    print(client_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
