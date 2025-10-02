#!/usr/bin/env python3
"""Simple log generator: prints to stdout until interrupted."""

from __future__ import annotations

import os
import random
import sys
import time
from datetime import datetime

MESSAGES = [
    "heartbeat",
    "capture frame",
    "run inference",
    "write chunk",
    "upload batch",
    "calibrate",
    "diagnostic ok",
    "adjust exposure",
]

# Fixed interval
INTERVAL = 1.0

def get_sensor_name() -> str:
    # Get the SSH port from environment, fallback to a default
    port = os.environ.get('SSH_CONNECTION', '').split()
    if port and len(port) >= 2:
        return f"Mock Sensor {port[1]}"  # port[1] will be the destination port
    return "Unknown Sensor"

def main() -> int:
    sensor_name = get_sensor_name()
    counter = 0
    try:
        while True:
            counter += 1
            line = (
                f"[{timestamp()}] {sensor_name}: #{counter:05d} – "
                f"{random.choice(MESSAGES)}"
            )
            print(line, flush=True)
            time.sleep(max(INTERVAL, 0.05))
    except KeyboardInterrupt:
        print(f"[{timestamp()}] {sensor_name}: shutting down", flush=True)
        return 0


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


if __name__ == "__main__":
    sys.exit(main())