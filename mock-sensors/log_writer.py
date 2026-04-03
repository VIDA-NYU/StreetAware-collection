#!/usr/bin/env python3
"""Mock sensor log generator that mimics filter.py behavior."""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from datetime import datetime

# Simulated pipeline components
COMPONENTS = ["CAMERA", "AUDIO", "INFERENCE", "WRITER", "STATS"]

# Simulated log messages per component
MESSAGES = {
    "CAMERA": [
        "Captured frame",
        "Frame buffer full, dropping oldest",
        "Exposure adjusted",
        "White balance calibrated",
        "Resolution: 1920x1080",
    ],
    "AUDIO": [
        "Audio chunk recorded",
        "Sample rate: 44100 Hz",
        "Buffer flushed",
        "Microphone calibrated",
    ],
    "INFERENCE": [
        "Running detection model",
        "Detected 3 objects",
        "Detected 0 objects",
        "Model inference: 45ms",
        "Processing batch",
    ],
    "WRITER": [
        "Writing data chunk",
        "Chunk saved to disk",
        "Compressing data",
        "Syncing to storage",
    ],
    "STATS": [
        "CPU: 45%, Memory: 62%",
        "Disk usage: 23.4 GB",
        "Temperature: 52C",
        "Frames processed: {count}",
        "Data rate: 12.5 MB/s",
    ],
}

# Stats interval
STATS_INTERVAL = 3.0
LOG_INTERVAL = 0.5


def get_sensor_name(name_arg: str | None) -> str:
    if name_arg:
        return name_arg
    port = os.environ.get('SSH_CONNECTION', '').split()
    if port and len(port) >= 2:
        return f"Mock Sensor {port[1]}"
    return "Mock Sensor"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock sensor data collection")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Sensor name for logging")
    parser.add_argument("--duration", "-d", type=int, default=None,
                        help="Duration to run in seconds (default: run indefinitely)")
    parser.add_argument("--stats-interval", type=int, default=3,
                        help="Interval for printing stats in seconds (default: 3)")
    args = parser.parse_args()

    sensor_name = get_sensor_name(args.name)
    duration = args.duration
    stats_interval = args.stats_interval

    # Handle signals for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\n[FILTER] Received signal {signum}, shutting down gracefully...", flush=True)
        shutdown_requested = True

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Startup messages (like filter.py)
    duration_str = f"{duration} seconds" if duration else "indefinitely"
    print(f"[FILTER] Starting sensor data collection...", flush=True)
    print(f"[FILTER] Sensor: {sensor_name}", flush=True)
    print(f"[FILTER] Duration: {duration_str}", flush=True)
    print(f"[FILTER] Process PID: {os.getpid()}", flush=True)
    print(f"[FILTER] Stats interval: {stats_interval}s", flush=True)

    start_time = time.time()
    last_stats_time = start_time
    frame_count = 0

    try:
        while not shutdown_requested:
            elapsed = time.time() - start_time

            # Check duration limit
            if duration and elapsed >= duration:
                break

            frame_count += 1

            # Regular log messages
            component = random.choice(COMPONENTS)
            if component != "STATS":
                msg = random.choice(MESSAGES[component])
                msg = msg.format(count=frame_count)
                print(f"[{component}] {msg}", flush=True)

            # Stats at interval
            if time.time() - last_stats_time >= stats_interval:
                stats_msg = random.choice(MESSAGES["STATS"]).format(count=frame_count)
                print(f"[STATS] {stats_msg}", flush=True)
                print(f"[STATS] Elapsed: {elapsed:.1f}s / {duration_str}", flush=True)
                last_stats_time = time.time()

            time.sleep(LOG_INTERVAL)

    except KeyboardInterrupt:
        pass

    # Completion message
    elapsed = time.time() - start_time
    print(f"[FILTER] Data collection completed after {elapsed:.1f} seconds.", flush=True)
    print(f"[FILTER] Total frames processed: {frame_count}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())