#!/bin/bash
# Deploy RAAAL dashboard to Cloudflare Pages via Direct Upload API

set -e

PROJECT_NAME="${1:-raaal-dashboard}"
DOMAIN="${DOMAIN:-yourdomain.com}"
DASHBOARD_HTML="${2:-reports/regime_dashboard.html}"
CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID}"
CLOUDFLARE_API_TOKEN="${CLOUDFLARE_API_TOKEN:-$GLOBAL_API_TOKEN}"
CLOUDFLARE_EMAIL="${CLOUDFLARE_EMAIL}"

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo "Error: CLOUDFLARE_API_TOKEN or GLOBAL_API_TOKEN environment variable not set"
    exit 1
fi

# Determine auth method
if [ -n "$CLOUDFLARE_EMAIL" ] && [ -n "$GLOBAL_API_TOKEN" ]; then
    AUTH_HEADERS="-H X-Auth-Email:$CLOUDFLARE_EMAIL -H X-Auth-Key:$GLOBAL_API_TOKEN"
else
    AUTH_HEADERS="-H Authorization:Bearer $CLOUDFLARE_API_TOKEN"
fi

if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: CLOUDFLARE_ACCOUNT_ID environment variable not set"
    echo "Get it from: https://dash.cloudflare.com/ -> Account ID in right sidebar"
    exit 1
fi

if [ ! -f "$DASHBOARD_HTML" ]; then
    echo "Error: Dashboard HTML not found at $DASHBOARD_HTML"
    echo "Run: python -m src.visualization.bokeh_app --output $DASHBOARD_HTML"
    exit 1
fi

echo "Creating deployment directory..."
DEPLOY_DIR=$(mktemp -d)
cp "$DASHBOARD_HTML" "$DEPLOY_DIR/index.html"

echo "Creating tarball for upload..."
TARBALL=$(mktemp -u).tar.gz
tar -czf "$TARBALL" -C "$DEPLOY_DIR" .

echo "Uploading to Cloudflare Pages..."
RESPONSE=$(curl -s -X POST \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/$PROJECT_NAME/deployments" \
  $AUTH_HEADERS \
  -F "file=@$TARBALL")

# Clean up
rm -rf "$DEPLOY_DIR" "$TARBALL"

# Parse response
SUCCESS=$(echo "$RESPONSE" | grep -o '"success":true' || echo "")
if [ -n "$SUCCESS" ]; then
    URL=$(echo "$RESPONSE" | grep -o '"url":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo ""
    echo "✅ Deployment successful!"
    echo "URL: $URL"
    echo ""
    echo "To configure ${DOMAIN}:"
    echo "1. Go to Cloudflare DNS settings for ${DOMAIN}"
    echo "2. Add CNAME record: @ -> $PROJECT_NAME.pages.dev"
    echo "3. Or for www: www -> $PROJECT_NAME.pages.dev"
else
    echo "❌ Deployment failed:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi
