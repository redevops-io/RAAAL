#!/usr/bin/env python3
"""Point ZITADEL at an SMTP sender so the hosted register flow can deliver its
verification code — without which a self-registered user is stranded on the
"Activate User" screen (that screen both verifies the address *and* sets the
password, so there is no out-of-band way to finish it).

Runs where the bootstrap PAT already is: the identity pod's bootstrap sidecar,
against `localhost`, with the same Host header the OIDC bootstrap uses. The SMTP
password (the Postmark server token) is read from the environment, never an
argument, so it is not left in the pod's process table; ZITADEL stores it in RDS
encrypted with the master key.

Idempotent by sender address: a re-run adopts the provider already configured
rather than stacking a second, so the bootstrap sidecar can call it on every
start. Adds, then activates, then optionally sends one test message.

The provider is Postmark by default (`--host smtp.postmarkapp.com:587`, STARTTLS
on 587, the server token as both username and password — Postmark's scheme). The
sender address must be on a domain the token is allowed to send from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _client(base_url: str, identity_host: str, pat: str):
    def call(path: str, body: dict | None = None, method: str | None = None):
        m = method or ("POST" if body is not None else "GET")
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {pat}",
                "Content-Type": "application/json",
                "Host": identity_host,
                "X-Forwarded-Proto": "https",
            },
            method=m,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, {"error": error.read().decode(errors="replace")[:600]}
    return call


def _existing_id(call, sender_address: str) -> str | None:
    """The id of a provider already configured for this sender, if any. The
    list item's shape has moved across ZITADEL versions, so this reads the id
    and sender out of whatever nesting it finds rather than a fixed path."""
    status, body = call("/admin/v1/smtp/_search", {})
    if status != 200:
        raise SystemExit(f"could not list SMTP providers: {status} {body}")
    for item in body.get("result") or []:
        config = item.get("config") or item
        sender = (config.get("senderAddress") or config.get("sender_address")
                  or item.get("senderAddress"))
        cid = item.get("id") or config.get("id")
        if sender == sender_address and cid:
            return cid
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--identity-host", default="auth.quantify.club")
    ap.add_argument("--pat-file", required=True)
    ap.add_argument("--host", required=True, help="host:port, e.g. smtp.postmarkapp.com:587")
    ap.add_argument("--sender", required=True, help="sender address on a domain the token may send from")
    ap.add_argument("--sender-name", default="Quantify")
    ap.add_argument("--tls", default="true", help="STARTTLS (true for port 587)")
    ap.add_argument("--test", default="", help="if set, send one test message to this address")
    args = ap.parse_args()

    pat = open(args.pat_file, encoding="utf-8").read().strip()
    token = os.environ.get("POSTMARK_TOKEN", "").strip()
    if not token:
        raise SystemExit("POSTMARK_TOKEN is empty in the environment — the SMTP "
                         "password is read from there, never passed as an argument.")

    call = _client(args.url, args.identity_host, pat)

    cid = _existing_id(call, args.sender)
    if cid:
        print(f"smtp: provider for {args.sender} already present ({cid})", file=sys.stderr)
    else:
        payload = {
            "senderAddress": args.sender,
            "senderName": args.sender_name,
            "tls": args.tls.lower() == "true",
            "host": args.host,
            "user": token,
            "password": token,
        }
        status, body = call("/admin/v1/smtp", payload)
        if status != 200:
            raise SystemExit(f"AddSMTPConfig failed: {status} {json.dumps(body)}")
        cid = body.get("id")
        if not cid:
            raise SystemExit(f"AddSMTPConfig returned no id: {json.dumps(body)}")
        print(f"smtp: added provider {cid} ({args.sender} via {args.host})", file=sys.stderr)

    status, body = call(f"/admin/v1/smtp/{cid}/_activate", {})
    # Already-active is success, not failure, on a re-run.
    if status not in (200, 409) and "already" not in json.dumps(body).lower():
        print(f"smtp: activate returned {status} {json.dumps(body)}", file=sys.stderr)
    else:
        print(f"smtp: provider {cid} active", file=sys.stderr)

    if args.test:
        status, body = call(f"/admin/v1/smtp/{cid}/_test",
                            {"receiverAddress": args.test, "email": args.test})
        print(f"smtp: test to {args.test} -> {status} {json.dumps(body)}", file=sys.stderr)
        if status != 200:
            return 2

    print(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
