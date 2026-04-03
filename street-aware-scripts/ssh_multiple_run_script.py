#!/usr/bin/env python3

"""Launch sensor processes on multiple devices over SSH and monitor their status.

The script reads sensor definitions from ``street-aware-scripts/config.py``.
Each sensor entry can provide:

    - host / port / username / password
    - display_name (used for status tracking)
    - process_match (optional string used when stopping a process)
    - command (template string; `{timeout}` and any sensor keys are available)

The script:
1. Starts sensor processes on remote devices (detached with nohup)
2. Monitors their status in real-time
3. Supports session resume after disconnect
4. Logs are saved on the remote device and can be downloaded after completion

States progress as: ``connecting`` → ``starting`` → ``running`` →
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

from job_status_tracker import update_status, get_current_session_id, save_session_log
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


def _build_independent_command(sensor: dict, template: str, timestamp: str, timeout: int) -> tuple[str, str, str]:
    """Build a command that runs independently of the SSH session.
    
    The sensor process will:
    - Run for exactly `timeout` seconds (handled by filter.py --duration)
    - Continue running even if SSH disconnects (using nohup)
    - Log output to a file for later download
    
    Returns:
        (start_command, log_file_path, pid_file_path)
    """
    mapping = dict(sensor)
    mapping.setdefault("display_name", _sensor_label(sensor))
    mapping.setdefault("name", mapping["display_name"])
    mapping.setdefault("timestamp", timestamp)
    mapping.setdefault("timeout", timeout)
    base_command = template.format(**mapping)
    
    # Create a unique log file path based on timestamp and sensor name
    safe_name = mapping["display_name"].replace(" ", "_").replace("/", "_")
    if get_current_mode() == "mock":
        log_dir = f"/tmp/sensor_logs/{timestamp}"
    else:
        log_dir = f"/media/reip/ssd/data/{timestamp}/logs"
    log_file = f"{log_dir}/{safe_name}.log"
    pid_file = f"{log_dir}/{safe_name}.pid"
    
    # Build command that:
    # 1. Creates log directory
    # 2. Runs the process with nohup in background (survives SSH disconnect)
    # 3. Sets COLLECTION_TIMESTAMP environment variable
    # 4. Redirects output to log file
    # 5. Echoes PID first (for immediate capture), then saves to file
    # 
    # We wrap in bash -c to handle cd && commands properly.
    # Echo PID first so we can capture it even if the file write is slow.
    inner_cmd = f"export COLLECTION_TIMESTAMP={shlex.quote(timestamp)}; {base_command}"
    start_cmd = (
        f"mkdir -p {shlex.quote(log_dir)} && "
        f"nohup bash -c {shlex.quote(inner_cmd)} > {shlex.quote(log_file)} 2>&1 & "
        f"PID=$!; echo $PID; echo $PID > {shlex.quote(pid_file)}"
    )
    
    return start_cmd, log_file, pid_file


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


def _start_independent_process(client: paramiko.SSHClient, sensor: dict, template: str, timestamp: str, timeout: int) -> tuple[int | None, str, str]:
    """Start a sensor process that runs independently of the SSH session.
    
    Returns:
        (pid, log_file, pid_file): The remote PID and paths to log/pid files.
    """
    start_cmd, log_file, pid_file = _build_independent_command(sensor, template, timestamp, timeout)
    process_match = sensor.get("process_match", "filter.py")
    
    # Execute the command
    stdin, stdout, stderr = client.exec_command(start_cmd, timeout=15)
    
    # Set channel timeout - give it enough time to echo the PID
    stdout.channel.settimeout(5.0)
    
    # Try to read the output (should contain the PID)
    pid = None
    try:
        output = stdout.read().decode().strip()
        # PID should be the first line (we echo it immediately)
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                print(f"[DEBUG] Got PID {pid} from command output", flush=True)
                break
    except Exception as e:
        print(f"[DEBUG] Timeout reading command output: {e}", flush=True)
    
    # If we didn't get PID from output, try reading the PID file
    if pid is None:
        time.sleep(1)
        for attempt in range(10):
            try:
                stdin2, stdout2, stderr2 = client.exec_command(
                    f"cat {shlex.quote(pid_file)} 2>/dev/null", timeout=5
                )
                stdout2.channel.settimeout(3.0)
                pid_from_file = stdout2.read().decode().strip()
                if pid_from_file.isdigit():
                    pid = int(pid_from_file)
                    print(f"[DEBUG] Got PID {pid} from file", flush=True)
                    break
            except Exception as e:
                pass
            time.sleep(0.5)
    
    # Failsafe: If still no PID, try to extract it from the log file
    # The filter.py script prints "[FILTER] Process PID: XXXXX" at startup
    if pid is None:
        try:
            # Wait for the process to start and write its PID to the log
            time.sleep(3)
            stdin3, stdout3, stderr3 = client.exec_command(
                f"grep -o 'Process PID: [0-9]*' {shlex.quote(log_file)} 2>/dev/null | head -1 | grep -o '[0-9]*'",
                timeout=5
            )
            stdout3.channel.settimeout(3.0)
            pid_from_log = stdout3.read().decode().strip()
            
            if pid_from_log.isdigit():
                pid = int(pid_from_log)
                # Verify the process is actually running
                stdin4, stdout4, stderr4 = client.exec_command(
                    f"kill -0 {pid} 2>/dev/null && echo running || echo stopped", timeout=5
                )
                stdout4.channel.settimeout(3.0)
                if stdout4.read().decode().strip() == "running":
                    print(f"[DEBUG] Got PID {pid} from log file (verified running)", flush=True)
                    # Write it to the PID file for future reference
                    client.exec_command(f"echo {pid} > {shlex.quote(pid_file)}")
                else:
                    print(f"[DEBUG] Log file had PID {pid} but it's not running", flush=True)
                    pid = None
        except Exception as e:
            print(f"[DEBUG] Log file PID extraction failed: {e}", flush=True)
    
    # Last resort: pgrep for the process
    if pid is None:
        try:
            stdin3, stdout3, stderr3 = client.exec_command(
                f"pgrep -n -f {shlex.quote(process_match)} 2>/dev/null", timeout=5
            )
            stdout3.channel.settimeout(3.0)
            pid_from_pgrep = stdout3.read().decode().strip()
            
            if pid_from_pgrep.isdigit():
                pid = int(pid_from_pgrep)
                # Verify the process is actually running
                stdin4, stdout4, stderr4 = client.exec_command(
                    f"kill -0 {pid} 2>/dev/null && echo running || echo stopped", timeout=5
                )
                stdout4.channel.settimeout(3.0)
                if stdout4.read().decode().strip() == "running":
                    print(f"[DEBUG] Got PID {pid} from pgrep fallback (verified running)", flush=True)
                    client.exec_command(f"echo {pid} > {shlex.quote(pid_file)}")
                else:
                    print(f"[DEBUG] pgrep found PID {pid} but it's not running", flush=True)
                    pid = None
        except Exception as e:
            print(f"[DEBUG] pgrep fallback failed: {e}", flush=True)

    if pid is None:
        print("[DEBUG] Warning: Could not capture PID", flush=True)

    return pid, log_file, pid_file


def _check_remote_process(client: paramiko.SSHClient, pid: int, process_match: str = None) -> bool:
    """Check if a remote process is still running.
    
    If PID is provided, checks if that specific PID is running.
    If PID is None but process_match is provided, checks for any matching process.
    """
    try:
        if pid:
            # Check if the exact PID is running
            stdin, stdout, stderr = client.exec_command(f"kill -0 {pid} 2>/dev/null && echo running || echo stopped", timeout=10)
            stdout.channel.settimeout(5.0)
            result = stdout.read().decode().strip()
            return result == "running"
        elif process_match:
            # No PID, check if any matching process exists
            stdin, stdout, stderr = client.exec_command(f"pgrep -f {shlex.quote(process_match)} 2>/dev/null", timeout=10)
            stdout.channel.settimeout(5.0)
            matches = stdout.read().decode().strip()
            return bool(matches)
        else:
            # No PID and no process_match - can't check
            return False
    except Exception as e:
        print(f"[DEBUG] _check_remote_process error: {e}", flush=True)
        return False


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
    """Worker that starts an independent sensor process and monitors its status.
    
    The sensor process runs independently on the remote device using nohup.
    This worker:
    1. Starts the sensor process (or reconnects to an existing one)
    2. Monitors process status in real-time
    3. Automatically reconnects if the SSH connection drops
    4. Does NOT kill the sensor when disconnecting (unless explicitly stopped)
    
    Logs are saved on the remote device and can be downloaded after completion.
    """
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
    pid = None
    log_file = None
    pid_file = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    reconnect_delay = 5  # seconds
    status_check_interval = 3  # seconds
    
    while not stop_event.is_set():
        try:
            # Connect to the sensor
            if client is None:
                client = _connect_client(sensor)
                update_status(label, "starting", session_id=session_id, 
                            remote_host=sensor.get("host"), port=sensor.get("port"))
                reconnect_attempts = 0
            
            template = sensor.get("command") or command_template
            process_match = sensor.get("process_match")
            
            # Check if we need to start a new process or reconnect to existing one
            if pid is None or not _check_remote_process(client, pid, process_match):
                # Start a new independent process
                print(f"[{label}] Starting sensor process (duration={timeout}s)...", flush=True)
                pid, log_file, pid_file = _start_independent_process(client, sensor, template, timestamp, timeout)
                
                if pid:
                    print(f"[{label}] Sensor started with PID {pid}", flush=True)
                    save_session_log(session_id, label, f"Sensor process started with PID {pid}, duration={timeout}s")
                else:
                    print(f"[{label}] Warning: Could not capture PID", flush=True)
                    save_session_log(session_id, label, "Warning: Could not capture PID")
                
                update_status(
                    label,
                    "running",
                    session_id=session_id,
                    pid=pid,
                    log_file=log_file,
                    pid_file=pid_file,
                    started_at=_iso_now(),
                    remote_host=sensor.get("host"),
                    port=sensor.get("port"),
                    timestamp=timestamp,
                    duration=timeout,
                )
            else:
                print(f"[{label}] Reconnected to existing process (PID {pid})", flush=True)
                save_session_log(session_id, label, f"Reconnected to existing process (PID {pid})")
                update_status(label, "running", session_id=session_id, 
                            reconnected_at=_iso_now(), pid=pid, log_file=log_file)
            
            # Monitor process status until it ends or we're stopped
            while not stop_event.is_set():
                time.sleep(status_check_interval)
                
                # Check if process is still running (works with PID or process_match)
                if not _check_remote_process(client, pid, process_match):
                    # Process has ended
                    print(f"[{label}] Sensor process completed", flush=True)
                    save_session_log(session_id, label, "Sensor process completed")
                    update_status(label, "completed", session_id=session_id, 
                                finished_at=_iso_now(), log_file=log_file)
                    break
                
                # Update status to show we're still monitoring
                update_status(label, "running", session_id=session_id, 
                            last_check=_iso_now(), pid=pid)
            
            # Exit the outer loop if process completed
            if not _check_remote_process(client, pid, process_match):
                break
                    
        except Exception as exc:
            reconnect_attempts += 1
            error_msg = f"Connection error (attempt {reconnect_attempts}/{max_reconnect_attempts}): {exc}"
            print(f"[{label}] {error_msg}", flush=True)
            save_session_log(session_id, label, error_msg)
            
            if reconnect_attempts >= max_reconnect_attempts:
                update_status(
                    label,
                    "disconnected",
                    session_id=session_id,
                    error=str(exc),
                    last_error_at=_iso_now(),
                    pid=pid,
                    log_file=log_file,
                )
                print(f"[{label}] Max reconnect attempts reached", flush=True)
                print(f"[{label}] NOTE: Sensor process (PID {pid}) may still be running!", flush=True)
                save_session_log(session_id, label, f"Disconnected. Sensor (PID {pid}) may still be running.")
                break
            
            # Close the client and try to reconnect
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                client = None
            
            update_status(label, "reconnecting", session_id=session_id, 
                        reconnect_attempt=reconnect_attempts, pid=pid)
            print(f"[{label}] Waiting {reconnect_delay}s before reconnecting...", flush=True)
            time.sleep(reconnect_delay)
            continue
    
    # Handle stop event based on whether user explicitly requested to stop sensors
    if stop_event.is_set() and pid:
        if user_requested_stop:
            # User clicked Stop - kill the sensor process
            print(f"[{label}] User requested stop, terminating sensor process (PID {pid})...", flush=True)
            _stop_remote_process(sensor, pid, sensor.get("process_match"))
            update_status(label, "stopped", session_id=session_id, finished_at=_iso_now())
            save_session_log(session_id, label, "Sensor process stopped by user request")
        else:
            # Service shutdown - sensors should keep running
            print(f"[{label}] Service stopping, sensor process (PID {pid}) will continue running", flush=True)
            update_status(label, "running", session_id=session_id, 
                         note="Monitoring disconnected, sensor still running",
                         last_check=_iso_now(), pid=pid, log_file=log_file)
            save_session_log(session_id, label, f"Monitoring stopped, sensor (PID {pid}) continues running")
    
    if client is not None:
        try:
            client.close()
        except Exception:
            pass

    print(f"[{label}] Worker finished", flush=True)


# ---------------------------------------------------------------------------
# CLI / main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH sensor process manager")
    parser.add_argument("--timeout", "-t", type=int, default=600, help="Sensor recording duration in seconds (default: 600)")
    parser.add_argument("--mode", choices=["mock", "real"], default=None, help="Sensor mode to use")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume monitoring existing sensor processes")
    parser.add_argument("--timestamp", type=str, default=None, help="Use a specific collection timestamp (for resuming)")
    return parser.parse_args()


def _find_running_sensors(sensors: list[dict], timestamp: str) -> dict[str, tuple[int, str]]:
    """Find sensors that already have running processes for the given timestamp.
    
    Returns:
        Dict mapping sensor label to (pid, log_file) tuple.
    """
    running = {}
    for sensor in sensors:
        label = _sensor_label(sensor)
        try:
            client = _connect_client(sensor, timeout=5)
            safe_name = label.replace(" ", "_").replace("/", "_")
            log_dir = f"/media/reip/ssd/data/{timestamp}/logs"
            pid_file = f"{log_dir}/{safe_name}.pid"
            log_file = f"{log_dir}/{safe_name}.log"
            
            # Try to read the PID file
            stdin, stdout, stderr = client.exec_command(f"cat {shlex.quote(pid_file)} 2>/dev/null")
            pid_str = stdout.read().decode().strip()
            
            if pid_str.isdigit():
                pid = int(pid_str)
                if _check_remote_process(client, pid):
                    running[label] = (pid, log_file)
                    print(f"[{label}] Found running process with PID {pid}", flush=True)
            
            client.close()
        except Exception as e:
            print(f"[{label}] Could not check for running process: {e}", flush=True)
    
    return running


def resume_worker(sensor: dict, pid: int, log_file: str, timestamp: str) -> None:
    """Worker that reconnects to an existing sensor process and monitors its status."""
    label = _sensor_label(sensor)
    session_id = get_current_session_id()
    status_check_interval = 3  # seconds
    
    update_status(
        label,
        "reconnecting",
        session_id=session_id,
        remote_host=sensor.get("host"),
        port=sensor.get("port"),
        pid=pid,
    )

    client = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    reconnect_delay = 5
    
    while not stop_event.is_set():
        try:
            if client is None:
                client = _connect_client(sensor)
                reconnect_attempts = 0
            
            # Verify the process is still running
            if not _check_remote_process(client, pid):
                print(f"[{label}] Process (PID {pid}) completed", flush=True)
                save_session_log(session_id, label, f"Process (PID {pid}) completed")
                update_status(label, "completed", session_id=session_id, 
                            finished_at=_iso_now(), log_file=log_file)
                break
            
            print(f"[{label}] Resumed monitoring PID {pid}", flush=True)
            save_session_log(session_id, label, f"Resumed monitoring PID {pid}")
            update_status(label, "running", session_id=session_id, 
                        reconnected_at=_iso_now(), pid=pid, log_file=log_file, timestamp=timestamp)
            
            # Monitor process status until it ends or we're stopped
            while not stop_event.is_set():
                time.sleep(status_check_interval)
                
                if not _check_remote_process(client, pid):
                    print(f"[{label}] Sensor process completed", flush=True)
                    save_session_log(session_id, label, "Sensor process completed")
                    update_status(label, "completed", session_id=session_id, 
                                finished_at=_iso_now(), log_file=log_file)
                    break
                
                update_status(label, "running", session_id=session_id, 
                            last_check=_iso_now(), pid=pid)
            
            # Exit outer loop if process completed
            if not _check_remote_process(client, pid):
                break
                    
        except Exception as exc:
            reconnect_attempts += 1
            error_msg = f"Connection error (attempt {reconnect_attempts}/{max_reconnect_attempts}): {exc}"
            print(f"[{label}] {error_msg}", flush=True)
            save_session_log(session_id, label, error_msg)
            
            if reconnect_attempts >= max_reconnect_attempts:
                update_status(label, "disconnected", session_id=session_id, 
                            error=str(exc), last_error_at=_iso_now(), pid=pid, log_file=log_file)
                print(f"[{label}] Max reconnect attempts reached", flush=True)
                break
            
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                client = None
            
            update_status(label, "reconnecting", session_id=session_id, 
                        reconnect_attempt=reconnect_attempts, pid=pid)
            time.sleep(reconnect_delay)
            continue
    
    if client is not None:
        try:
            client.close()
        except Exception:
            pass

    print(f"[{label}] Worker finished", flush=True)


def main() -> None:
    args = parse_args()

    if args.mode:
        set_sensor_mode(args.mode)

    sensors = get_sensor_config()
    if not sensors:
        print("No sensors configured", flush=True)
        return

    command_template = get_sensor_command()
    
    # Use provided timestamp or generate a new one
    if args.timestamp:
        collection_timestamp = args.timestamp
    else:
        now = datetime.now()
        collection_timestamp = now.strftime("DATE_%m_%d_%Y_TIME_%H_%M_%S")

    mode_str = get_current_mode()
    
    if args.resume:
        print(f"Resuming session monitoring (mode={mode_str}, timestamp={collection_timestamp})", flush=True)
        
        # Find running sensors
        running_sensors = _find_running_sensors(sensors, collection_timestamp)
        
        if not running_sensors:
            print("No running sensor processes found for this timestamp", flush=True)
            return
        
        threads = []
        for sensor in sensors:
            label = _sensor_label(sensor)
            if label in running_sensors:
                pid, log_file = running_sensors[label]
                thread = threading.Thread(
                    target=resume_worker,
                    args=(sensor, pid, log_file, collection_timestamp),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
            else:
                print(f"[{label}] No running process found, skipping", flush=True)
    else:
        print(f"Starting SSH sessions (mode={mode_str}, timestamp={collection_timestamp})", flush=True)
        print(f"NOTE: Sensors will run INDEPENDENTLY and continue even if this script stops.", flush=True)
        
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

    print("All sessions completed or shutdown.", flush=True)
    if not stop_event.is_set():
        print(f"To resume monitoring later, run with: --resume --timestamp {collection_timestamp}", flush=True)


# Flag to track if user explicitly requested to kill sensors
user_requested_stop = False

def _handle_signal(signum, frame):
    global user_requested_stop
    if signum == signal.SIGTERM:
        # SIGTERM = user clicked Stop button, should kill sensors
        print(f"[MAIN] Received SIGTERM, stopping sensors...", flush=True)
        user_requested_stop = True
    else:
        # SIGINT = service shutdown, sensors should keep running
        print(f"[MAIN] Received SIGINT, disconnecting (sensors will keep running)...", flush=True)
        user_requested_stop = False
    stop_event.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    main()
