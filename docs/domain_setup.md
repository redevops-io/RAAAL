# Domain Setup for RAAAL Dashboard

## Purchase Domain from Cloudflare

1. Go to https://www.cloudflare.com/products/registrar/
2. Search for available domains:
   - `raaal-dashboard.com` (~$10/year)
   - `raaal-portfolio.com` (~$10/year)
   - `regime-allocation.com` (~$10/year)
3. Purchase with Cloudflare (at-cost pricing, no markup)

## Configure DNS for Cloudflare Pages

After purchasing, DNS is automatically managed by Cloudflare:

```bash
# Add CNAME record
Type: CNAME
Name: @ (or www)
Target: raaal-dashboard.pages.dev
Proxy: Enabled (orange cloud)
TTL: Auto
```

## Add Custom Domain to Pages Project

### Via API:
```bash
curl -X POST \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/raaal-dashboard/domains" \
  -H "X-Auth-Email: $CLOUDFLARE_EMAIL" \
  -H "X-Auth-Key: $GLOBAL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "your-new-domain.com"
  }'
```

### Via Dashboard:
1. Go to https://dash.cloudflare.com → Pages
2. Select `raaal-dashboard` project
3. Go to "Custom domains" tab
4. Click "Set up a custom domain"
5. Enter your domain name
6. Cloudflare automatically configures SSL certificate

## Verification

Once configured (takes 5-10 minutes):
- `https://your-domain.com` → shows dashboard
- SSL certificate automatically provisioned
- CDN enabled globally

## Alternative: Use Cloudflare Pages Subdomain

If you don't want to purchase a domain, you can use:
- `https://raaal-dashboard.pages.dev` (always works)
- Custom subdomain: `https://portfolio.your-existing-domain.com`

## GitHub Actions Auto-Deploy

With custom domain configured, GitHub Actions will automatically deploy:

```yaml
name: Deploy
on:
  push:
    branches: [master]
  schedule:
    - cron: '0 6 * * *'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        run: ./run_backtest.sh
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
```

Custom domain will automatically serve latest deployment.
