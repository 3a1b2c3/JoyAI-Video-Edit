#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting server..."
bash run_server_fp4.sh > server_test.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo

echo "Waiting for server to start (max 120s)..."
for i in {1..120}; do
  if netstat -tlnp 2>/dev/null | grep -q ":8080 "; then
    echo "✓ Server listening on port 8080 (after ${i}s)"
    sleep 2
    break
  fi
  echo -n "."
  sleep 1
  if [ $((i % 10)) -eq 0 ]; then
    echo " (${i}s)"
  fi
done

echo
echo "Testing WebSocket connection..."
RESPONSE=$(curl -i http://localhost:8080/ws 2>&1 || true)
echo "$RESPONSE" | head -20
echo

if echo "$RESPONSE" | grep -q "101\|400\|404"; then
  echo "✓ Server responded (check status above)"
else
  echo "✗ No response from server"
  echo
  echo "Server log (last 50 lines):"
  tail -50 server_test.log
fi

echo
echo "Process still running?"
ps aux | grep $SERVER_PID | grep -v grep || echo "(process may have exited)"
