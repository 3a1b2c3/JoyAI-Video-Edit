#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== JoyAI-Video-Edit Server Diagnostic ==="
echo "Time: $(date)"
echo

echo "=== 1. Server Process Status ==="
if ps aux | grep -q "[s]erve_joyomni_streaming"; then
  PID=$(ps aux | grep "[s]erve_joyomni_streaming" | awk '{print $2}')
  echo "✓ Server running (PID: $PID)"
  ps aux | grep "[s]erve_joyomni_streaming"
else
  echo "✗ Server NOT running"
fi
echo

echo "=== 2. Port Listening Status ==="
echo "Checking port 8080..."
if netstat -tlnp 2>/dev/null | grep -q ":8080 " || ss -tlnp 2>/dev/null | grep -q ":8080 "; then
  echo "✓ Port 8080 is LISTENING"
  netstat -tlnp 2>/dev/null | grep ":8080" || ss -tlnp 2>/dev/null | grep ":8080"
else
  echo "✗ Port 8080 NOT listening"
fi
echo

echo "=== 3. WebSocket Connection Test ==="
echo "Testing curl -i http://localhost:8080/ws"
CURL_OUT=$(curl -i http://localhost:8080/ws 2>&1 || true)
echo "$CURL_OUT" | head -10
echo

echo "=== 4. Recent Server Log (last 30 lines) ==="
if [ -f server_test.log ]; then
  tail -30 server_test.log
elif [ -f server.log ]; then
  tail -30 server.log
else
  echo "(no log file found)"
fi
echo

echo "=== 5. GPU Status ==="
nvidia-smi -i 0 --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || echo "(nvidia-smi failed)"
echo

echo "=== 6. Process Memory Usage ==="
if ps aux | grep -q "[s]erve_joyomni_streaming"; then
  PID=$(ps aux | grep "[s]erve_joyomni_streaming" | awk '{print $2}')
  echo "Memory for PID $PID:"
  ps -p $PID -o pid,vsz,rss,comm | tail -1 || echo "(ps failed)"
else
  echo "(server not running)"
fi
echo

echo "=== Diagnostic complete ==="
