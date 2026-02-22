#!/bin/sh
set -eu

# ── Write config ─────────────────────────────────────────────────────────────
cat > /etc/garage.toml <<EOF
metadata_dir = "/tmp/garage/meta"
data_dir = "/tmp/garage/data"
db_engine = "sqlite"

replication_factor = 1

rpc_bind_addr = "0.0.0.0:3901"
rpc_secret = "${GARAGE_RPC_SECRET}"

[s3_api]
s3_region = "garage"
api_bind_addr = "0.0.0.0:3900"
root_domain = ".s3.garage"

[s3_web]
bind_addr = "0.0.0.0:3902"
root_domain = ".web.garage"

[admin]
api_bind_addr = "0.0.0.0:3903"
admin_token = "${GARAGE_ADMIN_TOKEN}"
EOF

# ── Start server in background ───────────────────────────────────────────────
garage server &
SERVER_PID=$!

# ── Wait for admin API ───────────────────────────────────────────────────────
echo "Waiting for Garage admin API..."
for i in $(seq 1 60); do
    if curl -sf -o /dev/null -H "Authorization: Bearer ${GARAGE_ADMIN_TOKEN}" \
        http://127.0.0.1:3903/v2/GetClusterStatus; then
        echo "Admin API ready."
        break
    fi
    sleep 0.5
done

# ── Bootstrap cluster layout ────────────────────────────────────────────────
NODE_ID=$(garage node id 2>/dev/null | cut -d@ -f1)
echo "Assigning layout to node ${NODE_ID}..."
garage layout assign -z dc1 -c 1G "${NODE_ID}"
garage layout apply --version 1

echo "Garage is ready."
touch /tmp/garage-ready

# ── Keep running ─────────────────────────────────────────────────────────────
wait "${SERVER_PID}"
