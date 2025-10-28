#!/usr/bin/env python3

"""Launch commands on multiple sensors over SSH and stream their logs.

The script reads sensor definitions from ``street-aware-scripts/config.py`` so
the same configuration can be consumed by both the backend service and the
frontend. Each sensor entry can provide:

    - host / port / username / password
    - display_name (used for logs + status tracking)
    - process_match (optional string used when stopping a process)
    - command (template string; `{timeout}` and any sensor keys are available)

The script keeps a live SSH session for every sensor, so the moment a line is
printed on the remote stdout/stderr we forward it to our own stdout. At the
same time we update ``job_status_tracker`` so the UI reflects the current
state. States progress as: ``connecting`` → ``starting`` → ``running`` →
``terminated`` / ``failed``.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import sys
import threading
import time
from datetime import datetime
from threading import Event

import paramiko

from job_status_tracker import clear_status, read_status, update_status, get_current_session_id, save_session_log
from config import (
    get_current_mode,
    get_sensor_config,
    get_sensor_command,
    set_sensor_mode,
)


stop_event = Event()


# ---------------------------------------------------------------------------
# Helpers


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _sensor_label(sensor: dict) -> str:
    return sensor.get("display_name") or sensor.get("name") or f"{sensor['host']}:{sensor.get('port', 22)}"


def _build_remote_command(sensor: dict, template: str, timeout: int, timestamp: str) -> str:
    mapping = dict(sensor)
    mapping.setdefault("timeout", timeout)
    mapping.setdefault("display_name", _sensor_label(sensor))
    mapping.setdefault("name", mapping["display_name"])
    mapping.setdefault("timestamp", timestamp)
    command = template.format(**mapping)
    return command


def _connect_client(sensor: dict, *, timeout: int = 10) -> paramiko.SSHClient:
    host = sensor["host"]
    if get_current_mode() == "mock":
        host = "localhost"
    port = sensor.get("port", 22)
    username = sensor["username"]
    password = sensor["password"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, password=password, timeout=timeout)
    return client


def _start_remote_command(client: paramiko.SSHClient, command: str, timeout: int, sensor: dict, timestamp: str) -> tuple[paramiko.channel.Channel, int | None]:
    # Export COLLECTION_TIMESTAMP before running the command
    inner = f"echo $$; export COLLECTION_TIMESTAMP={shlex.quote(timestamp)}; exec timeout {timeout}s bash -lc {shlex.quote(command)}"
    wrapped = f"bash -lc {shlex.quote(inner)}"
    stdin, stdout, stderr = client.exec_command(wrapped)

    pid = None
    try:
        pid_line = stdout.readline().strip()
        if pid_line.isdigit():
            pid = int(pid_line)
    except Exception:
        pid = None

    channel = stdout.channel
    channel.settimeout(1.0)
    # Attach stderr to the same channel
    return channel, pid


def _stream_channel(channel: paramiko.channel.Channel, label: str) -> None:
    buffer = b""
    last_update = 0.0
    session_id = get_current_session_id()

    while not stop_event.is_set():
        chunk = b""
        err_chunk = b""

        if channel.recv_ready():
            chunk = channel.recv(4096)
        if channel.recv_stderr_ready():
            err_chunk = channel.recv_stderr(4096)

        if chunk:
            buffer += chunk
        if err_chunk:
            buffer += err_chunk

        if buffer:
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line and not buffer:
                    continue
                clean = line.decode(errors="replace").rstrip("\r")
                print(f"[{label}] {clean}", flush=True)
                # Save log to persistent storage
                save_session_log(session_id, label, clean)
                update_status(label, "running", session_id=session_id, last_log=_iso_now())
                last_update = time.time()

        if channel.exit_status_ready() and not buffer:
            break

        if not chunk and not err_chunk:
            if buffer and time.time() - last_update >= 1.0:
                clean = buffer.decode(errors="replace").rstrip("\r")
                print(f"[{label}] {clean}", flush=True)
                save_session_log(session_id, label, clean)
                update_status(label, "running", session_id=session_id, last_log=_iso_now())
                buffer = b""
            time.sleep(0.1)

    if buffer:
        clean = buffer.decode(errors="replace").rstrip("\r")
        print(f"[{label}] {clean}", flush=True)
        save_session_log(session_id, label, clean)
        update_status(label, "running", session_id=session_id, last_log=_iso_now())


def _stop_remote_process(sensor: dict, pid: int | None, process_match: str | None) -> None:
    try:
        client = _connect_client(sensor, timeout=5)
    except Exception:
        return

    try:
        if pid:
            client.exec_command(f"kill -TERM {pid} 2>/dev/null || true")
            time.sleep(0.5)
            client.exec_command(f"kill -KILL {pid} 2>/dev/null || true")
        if process_match:
            client.exec_command(f"pkill -f {shlex.quote(process_match)} 2>/dev/null || true")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Worker


def sensor_worker(sensor: dict, command_template: str, timeout: int, timestamp: str) -> None:
    label = _sensor_label(sensor)
    session_id = get_current_session_id()
    
    update_status(
        label,
        "connecting",
        session_id=session_id,
        remote_host=sensor.get("host"),
        port=sensor.get("port"),
    )

    client = None
    channel = None
    pid = None
    try:
        client = _connect_client(sensor)
        update_status(label, "starting", session_id=session_id, remote_host=sensor.get("host"), port=sensor.get("port"))

        command = _build_remote_command(sensor, command_template, timeout, timestamp)
        channel, pid = _start_remote_command(client, command, timeout, sensor, timestamp)
        update_status(
            label,
            "running",
            session_id=session_id,
            pid=pid,
            command=command,
            started_at=_iso_now(),
            remote_host=sensor.get("host"),
            port=sensor.get("port"),
            timestamp=timestamp,
        )

        _stream_channel(channel, label)

        exit_status = channel.recv_exit_status()
        if exit_status == 0:
            update_status(label, "terminated", session_id=session_id, finished_at=_iso_now(), exit_code=exit_status)
            save_session_log(session_id, label, f"Process completed successfully (exit code: {exit_status})")
        else:
            update_status(label, "failed", session_id=session_id, finished_at=_iso_now(), exit_code=exit_status)
            save_session_log(session_id, label, f"Process failed (exit code: {exit_status})")

    except Exception as exc:
        update_status(
            label,
            "failed",
            session_id=session_id,
            error=str(exc),
            finished_at=_iso_now(),
        )
        save_session_log(session_id, label, f"Exception occurred: {str(exc)}")
    finally:
        if stop_event.is_set() and pid:
            _stop_remote_process(sensor, pid, sensor.get("process_match"))

        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        print(f"[{label}] Disconnected", flush=True)


# ---------------------------------------------------------------------------
# CLI / main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH log multiplexer")
    parser.add_argument("--timeout", "-t", type=int, default=600, help="Session timeout in seconds (default: 600)")
    parser.add_argument("--mode", choices=["mock", "real"], default=None, help="Sensor mode to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode:
        set_sensor_mode(args.mode)

    sensors = get_sensor_config()
    if not sensors:
        print("No sensors configured", flush=True)
        return

    command_template = get_sensor_command()
    
    # Generate collection timestamp once for all sensors
    now = datetime.now()
    collection_timestamp = now.strftime("DATE_%m_%d_%Y_TIME_%H_%M_%S")

    print(f"Starting SSH sessions (mode={get_current_mode()}, timeout={args.timeout}s, timestamp={collection_timestamp})", flush=True)
    # Don't clear status - preserve existing session info for resume capability
    # clear_status()

    threads = []
    for sensor in sensors:
        template = sensor.get("command") or command_template
        thread = threading.Thread(
            target=sensor_worker,
            args=(sensor, template, args.timeout, collection_timestamp),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
        for t in threads:
            t.join()

    print("All SSH sessions completed or shutdown.", flush=True)


def _handle_signal(signum, frame):
    print(f"[MAIN] Received signal {signum}, initiating shutdown.", flush=True)
    stop_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    main()
