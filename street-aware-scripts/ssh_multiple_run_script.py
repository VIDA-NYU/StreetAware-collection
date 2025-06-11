#!/usr/bin/env python3

import argparse
import paramiko
import threading
import time
import sys
import signal
from threading import Event

# Dictionary to track active SSH connections
active_connections = {}
# Event to signal global shutdown
stop_event = Event()

# Signal handler for graceful shutdown
def shutdown_handler(signum, frame):
    print(f"[MAIN] Received signal {signum}, initiating shutdown.", flush=True)
    stop_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl-C
signal.signal(signal.SIGTERM, shutdown_handler)  # kill
signal.signal(signal.SIGUSR1, shutdown_handler)  # custom


def ssh_into_device(host, username, password, command, timeout):
    """
    SSH into a device, run the command, and stop after `timeout` seconds or on stop_event.
    """
    start_time = time.time()
    client = None
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username=username, password=password, timeout=10)
        except Exception as e:
            print(f"[{host}] Connection failed: {e}", flush=True)
            return

        print(f"[{host}] Connected (timeout={timeout}s)", flush=True)
        active_connections[host] = client

        # Wrap in shell timeout for remote kill
        remote_cmd = f"timeout {timeout}s sh -c '{command}'"
        stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout, get_pty=True)

        # Stream output until remote ends or shutdown requested
        while not stop_event.is_set():
            if stdout.channel.exit_status_ready():
                break
            # Read available data
            if stdout.channel.recv_ready():
                for line in stdout.readline().splitlines():
                    print(f"[{host}] {line}", flush=True)
            time.sleep(0.1)
            # Local timeout check
            if time.time() - start_time > timeout:
                print(f"[{host}] Local timeout reached; closing.", flush=True)
                break

        # Kill remote process if still running
        try:
            stdout.channel.close()
            stderr.channel.close()
        except:
            pass

    except Exception as e:
        print(f"[{host}] Exception: {e}", flush=True)
    finally:
        if client:
            client.close()
        active_connections.pop(host, None)
        print(f"[{host}] Disconnected", flush=True)


def main():
    parser = argparse.ArgumentParser(description="SSH data collector with timeout and graceful shutdown")
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="Session timeout in seconds (default: 600)"
    )
    args = parser.parse_args()
    timeout = args.timeout

    print(f"Starting SSH sessions with timeout: {timeout} seconds", flush=True)

    # List of devices
    devices = [
        {"host": "192.168.0.184", "username": "reip", "password": "reip"},
        {"host": "192.168.0.122", "username": "reip", "password": "reip"},
        {"host": "192.168.0.108", "username": "reip", "password": "reip"},
        {"host": "192.168.0.227", "username": "reip", "password": "reip"},
    ]
    command_to_run = "cd software/reip-pipelines/smart-filter && python3 filter.py"

    # Launch SSH threads as daemons
    threads = []
    for dev in devices:
        t = threading.Thread(
            target=ssh_into_device,
            args=(dev["host"], dev["username"], dev["password"], command_to_run, timeout),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Wait for shutdown signal or all threads to complete
    try:
        while not stop_event.is_set() and any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    print("All SSH sessions completed or shutdown.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
