#!/usr/bin/env python3
import os
import paramiko
import socket
import json
import threading
import stat
from datetime import datetime
import re

from config import get_sensor_config

# Get nodes from config
NODES = get_sensor_config()


# Base local directory to store downloads
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def _get_latest_remote_folder(client):
    """
    Returns the latest timestamped folder in /media/reip/ssd/data/
    that matches DATE_MM_DD_YYYY_TIME_HH_MM_SS_LOCAL format.
    """
    cmd = "ls /media/reip/ssd/data"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=5)
    out = stdout.read().decode().strip().split("\n")
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(f"remote list command error: {err}")

    pattern = re.compile(r"^DATE_(\d{2})_(\d{2})_(\d{4})_TIME_(\d{2})_(\d{2})_(\d{2})$")
    dated_folders = []
    for name in out:
        m = pattern.match(name)
        if m:
            # convert to datetime for sorting
            dt = datetime(
                int(m.group(3)), int(m.group(1)), int(m.group(2)),
                int(m.group(4)), int(m.group(5)), int(m.group(6))
            )
            dated_folders.append((dt, name))

    if not dated_folders:
        raise RuntimeError("no matching DATE_..._LOCAL folders found on remote")

    # return the folder name with latest datetime
    dated_folders.sort(reverse=True)
    return dated_folders[0][1]

def _get_remote_date(client):
    """
    Run `date +%b%d%Y` on the remote sensor to discover its data folder name.
    Returns a string like "Jun032025" or raises if the command fails.
    """
    stdin, stdout, stderr = client.exec_command("date +%b%d%Y", timeout=5)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err:
        raise RuntimeError(f"remote date command error: {err}")
    return out

def _remote_tree_size(sftp, remote_path):
    """
    Recursively sum up file sizes under remote_path on the SFTP server.
    Returns 0 if remote_path does not exist.
    """
    total = 0
    try:
        info = sftp.stat(remote_path)
    except IOError:
        return 0

    if stat.S_ISDIR(info.st_mode):
        for entry in sftp.listdir(remote_path):
            child = remote_path.rstrip("/") + "/" + entry
            total += _remote_tree_size(sftp, child)
    else:
        total += info.st_size

    return total

def _recursive_get_with_progress(sftp, remote_path, local_path, progress_cb):
    """
    Copy remote_path → local_path recursively. For each file, call sftp.get with a callback
    that only passes the new chunk size to progress_cb.
    """
    try:
        info = sftp.stat(remote_path)
    except IOError:
        return  # remote path doesn’t exist, skip

    if stat.S_ISDIR(info.st_mode):
        os.makedirs(local_path, exist_ok=True)
        for entry in sftp.listdir(remote_path):
            r_child = remote_path.rstrip("/") + "/" + entry
            l_child = os.path.join(local_path, entry)
            _recursive_get_with_progress(sftp, r_child, l_child, progress_cb)
    else:
        parent = os.path.dirname(local_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        prev = 0
        def file_cb(transferred, _):
            nonlocal prev
            chunk = transferred - prev
            prev = transferred
            if chunk > 0:
                progress_cb(chunk)

        sftp.get(remote_path, local_path, callback=file_cb)

def pull_host(node, local_date_str, report_dict, lock, manual_folder=None, is_manual=False):
    """
    Download one node's data. Steps:
      1) SSH → get that sensor's own date folder name (or use manual_folder)
      2) Build remote_base = /media/reip/ssd/data/<remote_date_str>
      3) Compute total bytes, then recursively SFTP—reporting PROGRESS only on each 1% boundary
      4) Write to local folder data/<local_date_str>/<host>/… (or manual/<host>/<folder> if manual)
    """
    host = node["host"]
    port = node.get("port", 22)
    username = node["username"]
    password = node["password"]
    display_name = node.get("display_name") or f"{host}:{port}"

    # Quick TCP check on port 22
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
    except Exception as e:
        with lock:
            report_dict[display_name] = {"status": "error", "error": f"tcp‐fail: {e}"}
        print(f"COMPLETE {display_name} ERROR", flush=True)
        return

    try:
        # Open SSH session
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
        )

        # 1) Ask the sensor for its own date directory name (or use manual_folder)
        if manual_folder:
            remote_date_str = manual_folder
        else:
            try:
                remote_date_str = _get_latest_remote_folder(client)
            except Exception as e:
                raise RuntimeError(f"failed to fetch remote date: {e}")

        remote_base = f"/media/reip/ssd/data/{remote_date_str}"
        safe_name = display_name.replace('/', '_').replace(':', '_')
        
        if is_manual:
            # Manual downloads go to manual/<sensor>/<folder>
            local_base = os.path.join(BASE_DATA_DIR, "manual", safe_name, remote_date_str)
        else:
            # Auto downloads go to <timestamp>/<sensor>
            local_base = os.path.join(BASE_DATA_DIR, local_date_str, safe_name)

        # Open SFTP
        sftp = client.open_sftp()

        # 2) Compute total bytes under that remote_base
        total_bytes = _remote_tree_size(sftp, remote_base)

        # If no bytes exist, just create an empty folder
        if total_bytes == 0:
            os.makedirs(local_base, exist_ok=True)
            with lock:
                if is_manual:
                    path_str = f"data/manual/{safe_name}/{remote_date_str}"
                else:
                    path_str = f"data/{local_date_str}/{safe_name}"
                report_dict[display_name] = {
                    "status": "downloaded",
                    "path": path_str,
                    "bytes": 0,
                    "total": 0,
                }
                print(f"COMPLETE {display_name} 0", flush=True)
                sftp.close()
                client.close()
                return

        downloaded = 0
        last_percent = -1  # track the last percent we logged
        
        # Send initial progress at 0%
        print(f"PROGRESS {display_name} 0 {total_bytes}", flush=True)

        # 3) As chunks arrive, update downloaded, compute percent, print only on new percent
        def progress_cb(chunk_bytes):
            nonlocal downloaded, last_percent
            downloaded += chunk_bytes
            percent = int((downloaded * 100) / total_bytes)
            if percent > last_percent:
                last_percent = percent
                print(f"PROGRESS {display_name} {downloaded} {total_bytes}", flush=True)

        # Ensure local folder exists
        os.makedirs(local_base, exist_ok=True)
        _recursive_get_with_progress(sftp, remote_base, local_base, progress_cb)

        sftp.close()
        client.close()

        # 4) Mark complete in the shared report dictionary
        with lock:
            if is_manual:
                path_str = f"data/manual/{safe_name}/{remote_date_str}"
            else:
                path_str = f"data/{local_date_str}/{safe_name}"
            report_dict[display_name] = {
                "status": "downloaded",
                "path": path_str,
                "bytes": total_bytes,
                "total": total_bytes,
            }

            print(f"COMPLETE {display_name} {local_base}", flush=True)

    except Exception as e:
        with lock:
            report_dict[display_name] = {"status": "error", "error": str(e)}
        # Include the error message so the UI can show it inline
        msg = str(e).replace("\n", " ")
        print(f"COMPLETE {display_name} ERROR {msg}", flush=True)

def main():
    import sys
    # Check if manual mode with JSON config
    manual_config = None
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Expect JSON on stdin with format: {"Sensor 184": "DATE_10_02_2025_TIME_14_30_00", ...}
        import json
        manual_config = json.loads(sys.stdin.read())
    
    # Use numeric timestamp format: YYYYMMDD_HHMMSS
    now = datetime.now()
    local_date_str = now.strftime("%Y%m%d_%H%M%S")
    
    is_manual = manual_config is not None
    
    if not is_manual:
        date_folder = os.path.join(BASE_DATA_DIR, local_date_str)
        os.makedirs(date_folder, exist_ok=True)

    report = {}
    lock = threading.Lock()
    threads = []

    # Launch one thread per node
    for node in NODES:
        display_name = node.get("display_name") or f"{node['host']}:{node.get('port', 22)}"
        manual_folder = manual_config.get(display_name) if manual_config else None
        
        # Skip sensors not in manual config if in manual mode
        if is_manual and manual_folder is None:
            continue
            
        t = threading.Thread(
            target=pull_host, 
            args=(node, local_date_str, report, lock, manual_folder, is_manual), 
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Print one final "SUMMARY" JSON for the SSE client
    print("SUMMARY " + json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
