#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

docker build -t mock-sensor:latest .

sensors=(sensor-a sensor-b sensor-c sensor-d)
ports=(2221 2222 2223 2224)

for i in "${!sensors[@]}"; do
  name="mock-${sensors[$i]}"
  port="${ports[$i]}"
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    docker rm -f "$name" >/dev/null
  fi
  container_id=$(docker run -d --name "$name" -p "${port}:22" mock-sensor:latest)
  docker exec "$name" ssh-keygen -A >/dev/null
  echo "Started $name on port $port (container $container_id)"
  echo "  ssh sensor@localhost -p $port (password: sensor)"
done

docker ps --format 'table {{.Names}}	{{.Ports}}	{{.Status}}'
