# Deploying the Agentic Investment OS to quantife.club

quantife.club runs as **one containerized app** on proxmox, alongside redevops.io / vibexgen.io /
demo.redevops.io, reached only through the **shared Cloudflare tunnel** (outbound-only; no inbound port).
The container serves the operating console at `/` and the §17 API under `/api/investment/*`; the nightly
Bokeh dashboard is served at `/research` from a mounted volume.

**Paper-trading-first / governance boundary:** the image ships no broker client and no order router, so
there is no code path to a real venue. The only state-mutating endpoint is mission approval, which writes
a *paper* order to the local state store. `mode: paper` is enforced at manifest validation.

## 1. Build + run the service

Merge `deploy/investment-agent.compose.yml` into the shared `integrated.compose.yml` (or run it stand-alone
on the `agentic` network), then:

```bash
ssh proxmox
cd /projects/agentic-os-stack            # wherever integrated.compose.yml lives
docker compose -f integrated.compose.yml up -d --build investment-agent
docker compose -f integrated.compose.yml logs -f investment-agent   # expect: Uvicorn running on :8250
curl -s http://investment-agent:8250/health
```

The service bind-mounts `/projects/agentic-os-src/agentic_os` (the shared runtime) and RAAAL's
`reports/` + `data/` (the nightly research artifacts). Its paper book + mission ledger persist in the
`investment_state` volume.

## 2. Route quantife.club through the shared tunnel

Add ONE ingress rule to the existing cloudflared `config.yml` (the same tunnel already fronting
demo.redevops.io) — above the catch-all 404 rule:

```yaml
ingress:
  # ... existing rules ...
  - hostname: quantife.club
    service: http://investment-agent:8250
  - service: http_status:404
```

Then publish the DNS route and restart the tunnel:

```bash
cloudflared tunnel route dns <tunnel-name> quantife.club
docker compose -f integrated.compose.yml restart cloudflared
```

Only the tunnel sidecar reaches `investment-agent` (it is `expose`d, not `ports`-published).

## 3. Nightly research artifacts

`.github/workflows/daily-deploy.yml` still runs the historical backtest + `bokeh_app` (with the DEMO
banner + legend) and now, instead of Cloudflare Pages, syncs `reports/regime_dashboard.html` +
`data/history/*.parquet` to `/projects/RAAAL/{reports,data}` on the proxmox host — the volumes the
container serves from. The live console reads the same parquet for the objective columns; if the parquet
is absent the app falls back to a deterministic synthetic series so it always renders.

## 4. Smoke test after deploy

```bash
curl -s https://quantife.club/health
curl -s -X POST https://quantife.club/api/investment/projects            # manifest (paper-only)
curl -s https://quantife.club/api/investment/projects/default/discoveries
curl -s -X POST https://quantife.club/api/investment/projects/default/missions/objective-compare
# open https://quantife.club/ -> three objective columns + attention queue + governed paper approval
# open https://quantife.club/research -> nightly Bokeh dashboard with the DEMO banner
```
