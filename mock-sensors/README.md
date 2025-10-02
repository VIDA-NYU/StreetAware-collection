# Docker-Based Mock Sensors

This directory contains everything you need to spin up a small fleet of
containerised "sensors" that expose an actual SSH service. Each sensor is just
an Ubuntu container with nothing running until you SSH in and launch whatever
process you want.

## Prerequisites

- Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- Docker Compose v2 (bundled with modern Docker Desktop)

> Verify your setup with `docker --version` and `docker compose version`.

## Usage

```bash
cd mock-sensors

# Build the image (run once, or whenever Dockerfile changes)
docker build -t mock-sensor:latest .

# Create a sensor container (no process running yet)
docker run -d \
  --name mock-sensor-a \
  -p 2221:22 \
  mock-sensor:latest

# Generate a host key inside the container the first time
docker exec mock-sensor-a ssh-keygen -A

# SSH into a sensor (password: sensor)
ssh sensor@localhost -p 2221

# Once connected, start whatever you like, e.g.:
sensor@mock-sensor-a$ python3 /opt/mock-sensor/log_writer.py

# Tail the container logs (which only capture stdout/stderr of running commands)
docker logs -f mock-sensor-a

# Stop and remove the container when done
docker rm -f mock-sensor-a
```

### Start all four mock sensors quickly

```bash
./build_and_run.sh

# Later, stop them with
./stop.sh
```

Adjust port mappings or create additional containers with `docker run -d -p
2222:22 mock-sensor:latest`, etc. There is no default log persistence; all
output is purely stdin/stdout, leaving the behaviour entirely under your
control.
