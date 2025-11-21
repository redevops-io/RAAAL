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

echo "Creating manifest..."
cat > "$DEPLOY_DIR/_headers" << EOF
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
EOF

cat > "$DEPLOY_DIR/_routes.json" << EOF
{
  "version": 1,
  "include": ["/*"],
  "exclude": []
}
EOF

echo "Creating deployment package..."
cd "$DEPLOY_DIR"

# Use shasum if sha256sum isn't available (macOS/some CI environments)
if command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASH_CMD="shasum -a 256"
else
    echo "Error: Neither sha256sum nor shasum found"
    exit 1
fi

INDEX_HASH=$($HASH_CMD index.html | cut -d' ' -f1)
HEADERS_HASH=$($HASH_CMD _headers | cut -d' ' -f1)
ROUTES_HASH=$($HASH_CMD _routes.json | cut -d' ' -f1)

cd - > /dev/null

echo "Uploading to Cloudflare Pages..."
echo "DEBUG: Account ID = ${CLOUDFLARE_ACCOUNT_ID:0:6}..."
echo "DEBUG: Project = $PROJECT_NAME"
echo "DEBUG: Token = ${CLOUDFLARE_API_TOKEN:0:6}..."
echo "DEBUG: Auth method = $([ -n "$CLOUDFLARE_EMAIL" ] && echo "API Key" || echo "Bearer Token")"

# Disable set -e temporarily to capture curl error
set +e
RESPONSE=$(curl -v -w "\nHTTP_STATUS:%{http_code}" -X POST \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/$PROJECT_NAME/deployments" \
  $AUTH_HEADERS \
  -H "Content-Type: application/json" \
  -d @- << PAYLOAD
{
  "manifest": {
    "/index.html": "$INDEX_HASH",
    "/_headers": "$HEADERS_HASH",
    "/_routes.json": "$ROUTES_HASH"
  }
}
PAYLOAD
)
CURL_EXIT=$?
set -e

if [ $CURL_EXIT -ne 0 ]; then
    echo "❌ Curl failed with exit code: $CURL_EXIT"
    case $CURL_EXIT in
        6) echo "Could not resolve host - check network/DNS" ;;
        7) echo "Failed to connect to host" ;;
        *) echo "See curl exit codes: https://curl.se/docs/manpage.html#EXIT" ;;
    esac
    exit $CURL_EXIT
fi

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS:/d')

echo "DEBUG: HTTP Status = $HTTP_STATUS"
echo "DEBUG: Response = $RESPONSE_BODY"

# Upload files if manifest accepted
if echo "$RESPONSE_BODY" | grep -q '"success":true'; then
    UPLOAD_TOKEN=$(echo "$RESPONSE_BODY" | grep -o '"jwt":"[^"]*"' | cut -d'"' -f4)
    
    echo "Uploading files..."
    cd "$DEPLOY_DIR"
    for file in index.html _headers _routes.json; do
        curl -s -X PUT \
          "https://api.cloudflare.com/client/v4/pages/assets/upload" \
          -H "Authorization: Bearer $UPLOAD_TOKEN" \
          --data-binary "@$file"
    done
    cd - > /dev/null
fi

# Clean up
rm -rf "$DEPLOY_DIR" ../deployment.zip 2>/dev/null || true

# Parse response
SUCCESS=$(echo "$RESPONSE_BODY" | grep -o '"success":true' || echo "")
if [ -n "$SUCCESS" ]; then
    URL=$(echo "$RESPONSE_BODY" | grep -o '"url":"[^"]*"' | head -1 | cut -d'"' -f4)
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
    echo "$RESPONSE_BODY" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE_BODY"
    exit 1
fi
