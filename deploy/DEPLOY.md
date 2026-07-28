# Deploying the Agentic Investment OS to quantify.club

quantify.club runs as **one containerized app** on proxmox, alongside redevops.io / vibexgen.io /
demo.redevops.io, reached only through the **shared Cloudflare tunnel** (outbound-only; no inbound port).
The container serves the operating console at `/` and the §17 API under `/api/investment/*`; the nightly
Bokeh dashboard is served at `/research` from a mounted volume.

**Paper-trading-first / governance boundary:** the image ships no broker client and no order router, so
there is no code path to a real venue. The only state-mutating endpoint is mission approval, which writes
a *paper* order to the local state store. `mode: paper` is enforced at manifest validation.

## 1. Build + run the service

`agentic_os` (the shared runtime) is **vendored** into the image, not bind-mounted (the proxmox
`/projects/agentic-os-src` is only a partial copy). Vendor it from a full agentic-os-src checkout first:

```bash
ssh proxmox
cd /projects && git clone https://github.com/redevops-io/RAAAL.git   # if not present
cd /projects/RAAAL && git fetch && git checkout feat/agentic-investment-os
scripts/vendor_agentic_os.sh /projects/agentic-os-src   # or rsync a full agentic_os into ./agentic_os
docker build -f Dockerfile.agent -t localhost:5000/investment-agent:stable .
```

Merge `deploy/investment-agent.compose.yml` into the shared `integrated.compose.yml` (it uses the
`*agent` anchor there — networks/security_opt/host-gateway), then:

```bash
cd /projects/agentic-os-stack
docker compose -f integrated.compose.yml up -d --no-deps investment-agent
curl -s http://192.168.40.105:8250/health          # what cloudflared will hit
```

The service mounts RAAAL's `reports/` (ro) + `data/` (rw — an import creates `data/models`) for the
nightly research artifacts; its paper book + mission ledger persist in the `investment_state` volume. If
the parquet is absent the console falls back to a deterministic synthetic snapshot.

## 2. Route quantify.club through the shared tunnel

Add ONE ingress rule to the existing cloudflared `config.yml` (the same tunnel already fronting
demo.redevops.io) — above the catch-all 404 rule:

```yaml
ingress:
  # ... existing rules ...
  - hostname: quantify.club
    service: http://investment-agent:8250
  - service: http_status:404
```

Reload the tunnel (it runs as the `cloudflared` systemd service on the host, config at
`/main/cloudflared/config.yml`):

```bash
cloudflared tunnel --config /main/cloudflared/config.yml ingress validate
systemctl restart cloudflared
```

### Go-live DNS record (the one manual step)

quantify.club is in a **different Cloudflare account** than the tunnel, so
`cloudflared tunnel route dns … quantify.club` fails with `Authentication error` (same cross-account
caveat as demo.redevops.io). Create this record **by hand in quantify.club's own Cloudflare zone**:

```
Type:   CNAME
Name:   quantify.club              (root / @)
Target: 1711f014-dea7-4b4d-a409-9cd38d1c4ee2.cfargotunnel.com
Proxy:  Proxied  (orange cloud)
```

Once that record exists, quantify.club resolves through the tunnel to `investment-agent:8250`. Nothing
else is required — the service + ingress rule are already live on proxmox.

## 3. Daily updates — proxmox systemd timer (the reliable path)

The GitHub Actions self-hosted runner (`ROG-Strix`) is intermittent and off-box, so daily refresh runs
**on proxmox** where the container + volumes live. `scripts/nightly_refresh.sh` runs the backtest +
Bokeh build inside the lean `investment-agent` image and writes fresh `data/history/*.parquet` +
`reports/regime_dashboard.html` into `/projects/RAAAL/{data,reports}`. The console reads the parquet on
every request and `/research` is a file read, so **no restart is needed** — the site updates on the next
page load. If the parquet is ever absent the console falls back to a deterministic synthetic snapshot.

Install the timer (once):

```bash
ssh proxmox
sudo tee /etc/systemd/system/raaal-nightly.service >/dev/null <<'UNIT'
[Unit]
Description=RAAAL nightly research refresh (quantify.club)
After=docker.service
Requires=docker.service
[Service]
Type=oneshot
ExecStart=/projects/RAAAL/scripts/nightly_refresh.sh
UNIT
sudo tee /etc/systemd/system/raaal-nightly.timer >/dev/null <<'UNIT'
[Unit]
Description=Run the RAAAL nightly refresh daily
[Timer]
OnCalendar=*-*-* 07:00:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
UNIT
chmod +x /projects/RAAAL/scripts/nightly_refresh.sh
sudo systemctl daemon-reload
sudo systemctl enable --now raaal-nightly.timer
systemctl start raaal-nightly.service      # run the first refresh now
systemctl list-timers raaal-nightly.timer  # confirm next run
tail -f /projects/RAAAL/reports/nightly.log
```

The GitHub Actions `daily-deploy.yml` is kept as an optional secondary path: when `ROG-Strix` is online
it builds the dashboard and, if `RAAAL_DEPLOY_DIR` + `INTEGRATED_COMPOSE_FILE` secrets are set and the
runner can reach the host, syncs artifacts too. The proxmox timer is authoritative.

## 4. Smoke test after deploy

```bash
curl -s https://quantify.club/health
curl -s -X POST https://quantify.club/api/investment/projects            # manifest (paper-only)
curl -s https://quantify.club/api/investment/projects/default/discoveries
curl -s -X POST https://quantify.club/api/investment/projects/default/missions/objective-compare
# open https://quantify.club/ -> three objective columns + attention queue + governed paper approval
# open https://quantify.club/research -> nightly Bokeh dashboard with the DEMO banner
```
