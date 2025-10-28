#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

sensors=(sensor-a sensor-b sensor-c sensor-d)

echo "Stopping mock sensors..."
for sensor in "${sensors[@]}"; do
  name="mock-${sensor}"
  if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
    echo "Stopping $name..."
    docker stop "$name" >/dev/null
    docker rm "$name" >/dev/null
    echo "✓ Stopped and removed $name"
  else
    echo "⚠ Container $name not found (already stopped?)"
  fi
done

echo -e "\nRemaining containers:"
docker ps --format 'table {{.Names}}\t{{.Status}}'