#!/usr/bin/env python3

import os
import re
import json
import socket
import paramiko
from datetime import datetime

from config import get_sensor_config

DATA_ROOT = "/media/reip/ssd/data"

# Matches DATE_MM_DD_YYYY_TIME_HH_MM_SS or with _LOCAL suffix
PATTERN = re.compile(r"^DATE_\d{2}_\d{2}_\d{4}_TIME_\d{2}_\d{2}_\d{2}(?:_LOCAL)?$")

def list_folders_for_host(sensor):
    host = sensor["host"]
    port = sensor.get("port", 22)
    username = sensor["username"]
    password = sensor["password"]
    label = sensor.get("display_name") or f"{host}:{port}"

    # Quick TCP check on port 22
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
    except Exception as e:
        return label, {"status": "error", "error": f"tcp-fail: {e}"}

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, port=port, username=username, password=password, timeout=6)

        cmd = f"ls -1 {DATA_ROOT} 2>/dev/null || true"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=6)
        out = stdout.read().decode().strip().split("\n") if stdout else []
        client.close()

        folders = [name for name in out if name and PATTERN.match(name)]
        # Sort descending by parsed datetime inside the name if possible
        def key_fn(name):
            try:
                # DATE_MM_DD_YYYY_TIME_HH_MM_SS[_LOCAL]
                parts = name.split("_")
                mm, dd, yyyy = int(parts[1]), int(parts[2]), int(parts[3])
                hh, mi, ss = int(parts[5]), int(parts[6]), int(parts[7].split("_")[0])
                return datetime(yyyy, mm, dd, hh, mi, ss)
            except Exception:
                return datetime.min
        folders.sort(key=key_fn, reverse=True)
        return label, {"status": "ok", "folders": folders}
    except Exception as e:
        return label, {"status": "error", "error": str(e)}


def main():
    sensors = get_sensor_config()
    result = {}
    for sensor in sensors:
        label, info = list_folders_for_host(sensor)
        result[label] = info
    print(json.dumps(result))

if __name__ == "__main__":
    main()
