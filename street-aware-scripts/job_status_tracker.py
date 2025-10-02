# job_status_tracker.py

import json
import os
import uuid
from datetime import datetime
from threading import Lock

# Get the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Create .data directory if it doesn't exist
DATA_DIR = os.path.join(BASE_DIR, '.data')
os.makedirs(DATA_DIR, exist_ok=True)

STATUS_FILE = os.path.join(DATA_DIR, 'ssh_job_status.json')
SESSIONS_DIR = os.path.join(DATA_DIR, 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)
status_lock = Lock()

# Current session ID - persists across restarts
CURRENT_SESSION_FILE = os.path.join(DATA_DIR, 'current_session.txt')

def _safe_read():
    if not os.path.exists(STATUS_FILE):
        _safe_write({})  # Initialize with empty state if file doesn't exist
        return {}
    with open(STATUS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            _safe_write({})  # Reset if file is corrupted
            return {}

def _safe_write(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def get_current_session_id():
    """Get or create current session ID."""
    if os.path.exists(CURRENT_SESSION_FILE):
        with open(CURRENT_SESSION_FILE, 'r') as f:
            session_id = f.read().strip()
            if session_id:
                return session_id
    
    # Create new session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
    with open(CURRENT_SESSION_FILE, 'w') as f:
        f.write(session_id)
    return session_id

def update_status(host, state, pid=None, clear_pid=False, session_id=None, **extra):
    """Persist host state along with optional metadata such as PID or log path."""
    with status_lock:
        data = _safe_read()
        if host not in data:
            data[host] = {}

        record = data[host]
        record["state"] = state
        record["last_updated"] = datetime.now().isoformat()
        
        # Always preserve session_id if provided or exists
        if session_id:
            record["session_id"] = session_id
        elif "session_id" not in record:
            record["session_id"] = get_current_session_id()

        if clear_pid:
            record.pop("pid", None)
        elif pid is not None:
            record["pid"] = pid

        for key, value in extra.items():
            if value is None:
                record.pop(key, None)
            else:
                record[key] = value

        data[host] = record
        _safe_write(data)

def clear_status():
    """Clear status but preserve session info for recovery."""
    with status_lock:
        # Instead of clearing everything, mark all as 'cleared' but keep PIDs and session info
        data = _safe_read()
        for host, record in data.items():
            if record.get("state") in ["running", "connecting", "starting"]:
                record["state"] = "interrupted"
                record["interrupted_at"] = datetime.now().isoformat()
        _safe_write(data)

def force_clear_status():
    """Completely clear all status - use with caution."""
    with status_lock:
        _safe_write({})

def read_status():
    with status_lock:
        return _safe_read()

def get_session_hosts(session_id):
    """Get all hosts for a specific session."""
    with status_lock:
        data = _safe_read()
        return {host: record for host, record in data.items() 
                if record.get("session_id") == session_id}

def get_active_sessions():
    """Get all active session IDs."""
    with status_lock:
        data = _safe_read()
        sessions = set()
        for record in data.values():
            if "session_id" in record:
                sessions.add(record["session_id"])
        return list(sessions)

def save_session_log(session_id, host, log_entry):
    """Save log entry for a specific session and host."""
    session_log_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_log_dir, exist_ok=True)
    
    log_file = os.path.join(session_log_dir, f"{host.replace('/', '_').replace(':', '_')}.log")
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().isoformat()
        f.write(f"[{timestamp}] {log_entry}\n")

def get_session_logs(session_id, host=None):
    """Get logs for a session, optionally filtered by host."""
    session_log_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_log_dir):
        return {}
    
    logs = {}
    if host:
        log_file = os.path.join(session_log_dir, f"{host.replace('/', '_').replace(':', '_')}.log")
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs[host] = f.read().splitlines()
    else:
        for filename in os.listdir(session_log_dir):
            if filename.endswith('.log'):
                host_name = filename[:-4].replace('_', ' ')  # rough reverse of sanitization
                log_file = os.path.join(session_log_dir, filename)
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs[host_name] = f.read().splitlines()
    
    return logs
