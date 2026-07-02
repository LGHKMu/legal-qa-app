#!/bin/sh
set -e

AUTH_INC="/etc/nginx/conf.d/basic_auth.inc"
TEMPLATE="/etc/nginx/templates/default.conf.template"
OUTPUT="/etc/nginx/conf.d/default.conf"

if [ "${NGINX_BASIC_AUTH:-off}" = "on" ]; then
    if [ ! -f /etc/nginx/htpasswd ]; then
        echo "[nginx] NGINX_BASIC_AUTH=on 但未挂载 /etc/nginx/htpasswd" >&2
        exit 1
    fi
    cat > "$AUTH_INC" <<'EOF'
auth_basic "Legal QA";
auth_basic_user_file /etc/nginx/htpasswd;
EOF
else
    cp /etc/nginx/conf.d/basic_auth.inc "$AUTH_INC" 2>/dev/null || : > "$AUTH_INC"
fi

export BACKEND_API_KEY="${BACKEND_API_KEY:-}"
envsubst '${BACKEND_API_KEY}' < "$TEMPLATE" > "$OUTPUT"

exec nginx -g 'daemon off;'
