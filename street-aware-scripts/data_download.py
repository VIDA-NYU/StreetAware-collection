#!/usr/bin/env python3
import os
import paramiko
import socket
import json
import threading
import stat
from datetime import datetime

# —— CONFIGURE YOUR NODES HERE —— #
NODES = [
    {"host": "192.168.0.184", "username": "reip", "password": "reip"},
    {"host": "192.168.0.122", "username": "reip", "password": "reip"},
    {"host": "192.168.0.108", "username": "reip", "password": "reip"},
    {"host": "192.168.0.227", "username": "reip", "password": "reip"},
]


# Base local directory to store downloads
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

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

def pull_host(node, local_date_str, report_dict, lock):
    """
    Download one node’s data. Steps:
      1) SSH → get that sensor’s own date folder name
      2) Build remote_base = /media/reip/ssd/data/<remote_date_str>
      3) Compute total bytes, then recursively SFTP—reporting PROGRESS only on each 1% boundary
      4) Write to local folder data/<local_date_str>/<host>/…
    """
    host = node["host"]
    username = node["username"]
    password = node["password"]

    # Quick TCP check on port 22
    try:
        sock = socket.create_connection((host, 22), timeout=5)
        sock.close()
    except Exception as e:
        with lock:
            report_dict[host] = {"status": "error", "error": f"tcp‐fail: {e}"}
        print(f"COMPLETE {host} ERROR", flush=True)
        return

    try:
        # Open SSH session
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            username=username,
            password=password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
        )

        # 1) Ask the sensor for its own date directory name
        try:
            remote_date_str = _get_remote_date(client)
        except Exception as e:
            raise RuntimeError(f"failed to fetch remote date: {e}")

        remote_base = f"/media/reip/ssd/data/{remote_date_str}"
        local_base = os.path.join(BASE_DATA_DIR, local_date_str, host)

        # Open SFTP
        sftp = client.open_sftp()

        # 2) Compute total bytes under that remote_base
        total_bytes = _remote_tree_size(sftp, remote_base)

        # If no bytes exist, just create an empty folder
        if total_bytes == 0:
            os.makedirs(local_base, exist_ok=True)
            with lock:
                report_dict[host] = {
                    "status": "downloaded",
                    "path": f"data/{local_date_str}/{host}",
                    "bytes": 0,
                    "total": 0,
                }
            print(f"COMPLETE {host} 0", flush=True)
            sftp.close()
            client.close()
            return

        downloaded = 0
        last_percent = -1  # track the last percent we logged

        # 3) As chunks arrive, update downloaded, compute percent, print only on new percent
        def progress_cb(chunk_bytes):
            nonlocal downloaded, last_percent
            downloaded += chunk_bytes
            percent = int((downloaded * 100) / total_bytes)
            if percent > last_percent:
                last_percent = percent
                print(f"PROGRESS {host} {downloaded} {total_bytes}", flush=True)

        # Ensure local folder exists
        os.makedirs(local_base, exist_ok=True)
        _recursive_get_with_progress(sftp, remote_base, local_base, progress_cb)

        sftp.close()
        client.close()

        # 4) Mark complete in the shared report dictionary
        with lock:
            report_dict[host] = {
                "status": "downloaded",
                "path": f"data/{local_date_str}/{host}",
                "bytes": total_bytes,
                "total": total_bytes,
            }

        print(f"COMPLETE {host} {local_base}", flush=True)

    except Exception as e:
        with lock:
            report_dict[host] = {"status": "error", "error": str(e)}
        print(f"COMPLETE {host} ERROR", flush=True)

def main():
    # Use your local system date to name the top‐level folder
    local_date_str = datetime.now().strftime("%b%d%Y")
    date_folder = os.path.join(BASE_DATA_DIR, local_date_str)
    os.makedirs(date_folder, exist_ok=True)

    report = {}
    lock = threading.Lock()
    threads = []

    # Launch one thread per node, each will:
    #   • SSH → get remote_date_str
    #   • SFTP remote_base → local folder under local_date_str
    for node in NODES:
        t = threading.Thread(
            target=pull_host, args=(node, local_date_str, report, lock), daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Print one final "SUMMARY" JSON for the SSE client
    print("SUMMARY " + json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
