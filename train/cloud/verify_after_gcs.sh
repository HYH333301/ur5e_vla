#!/bin/bash
# Wait for the aria2 base-weights fetch to finish, then smoke-test serving
# pi05_base from the local cache (validates weights + JAX + policy stack).
# Marker lines: WATCH-GCS-OK / WATCH-GCS-FAIL / WATCH-SERVE-OK / WATCH-SERVE-FAIL / WATCH-DONE
LOG=/root/shared-nvme/logs/setup_gcs.log
VLOG=/root/shared-nvme/logs/verify_serve.log

while pgrep -f aria2c >/dev/null; do sleep 60; done

if grep -q GCS-FETCH-DONE "$LOG"; then
  echo WATCH-GCS-OK
else
  echo WATCH-GCS-FAIL; exit 1
fi

source /root/shared-nvme/code/env.sh
cd /root/shared-nvme/code/openpi-main
timeout 420 /root/.local/bin/uv run scripts/serve_policy.py --port 8132 \
  policy:checkpoint --policy.config pi05_base \
  --policy.dir gs://openpi-assets/checkpoints/pi05_base >"$VLOG" 2>&1 &
SPID=$!
for i in $(seq 1 70); do
  grep -qiE "serving|websocket|listening" "$VLOG" && break
  kill -0 $SPID 2>/dev/null || break
  sleep 5
done
sleep 5
if grep -qiE "serving|websocket|listening" "$VLOG"; then
  echo WATCH-SERVE-OK
else
  echo WATCH-SERVE-FAIL
fi
kill $SPID 2>/dev/null
sleep 2; pkill -f "serve_polic[y]" 2>/dev/null
tail -20 "$VLOG"
echo WATCH-DONE
