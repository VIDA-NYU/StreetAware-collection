# job_status_tracker.py

import json
import os
from threading import Lock

STATUS_FILE = "/tmp/ssh_job_status.json"
status_lock = Lock()

def _safe_read():
    if not os.path.exists(STATUS_FILE):
        return {}
    with open(STATUS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _safe_write(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_status(host, state, pid=None):
    with status_lock:
        data = _safe_read()
        if host not in data:
            data[host] = {}
        data[host]["state"] = state
        if pid is not None:
            data[host]["pid"] = pid
        _safe_write(data)

def clear_status():
    with status_lock:
        _safe_write({})

def read_status():
    with status_lock:
        return _safe_read()
