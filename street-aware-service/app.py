#!/usr/bin/env python3

import os
import asyncio
import json
import sys
import shutil
import tempfile
import zipfile
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "street-aware-scripts"))
sys.path.insert(0, SCRIPT_DIR)

from job_status_tracker import read_status, get_active_sessions, get_session_hosts, get_session_logs, get_current_session_id
from config import (
    get_current_mode,
    get_public_sensor_config,
)
sys.path.remove(SCRIPT_DIR)


from fastapi import FastAPI, Query, HTTPException, Body
import base64
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Event broker for real-time updates
class EventBroker:
    def __init__(self):
        self._subscribers = set()

    async def subscribe(self):
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue):
        self._subscribers.remove(queue)

    async def publish(self, message):
        for queue in self._subscribers:
            await queue.put(message)

# Global event broker
event_broker = EventBroker()
log_broker = EventBroker()

# Relative paths to scripts
SCRIPT_PATH = os.path.normpath("../street-aware-scripts/ssh_multiple_run_script.py")
HEALTH_SCRIPT = os.path.normpath("../street-aware-scripts/health_check.py")

# Holds the currently running subprocess
current_proc = None
log_task: asyncio.Task | None = None
process_lock = asyncio.Lock()

LOG_BASE_DIR = Path(__file__).resolve().parent / ".log_cache"
CURRENT_LOG_DIR: Path | None = None
COMBINED_LOG_NAME = "combined.ndjson"
LOG_BACKLOG_LIMIT = 5000


def _prepare_log_directory() -> Path:
    """Create a fresh directory for the current log session."""
    global CURRENT_LOG_DIR

    if LOG_BASE_DIR.exists():
        shutil.rmtree(LOG_BASE_DIR)
    LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)

    session_dir = LOG_BASE_DIR / datetime.utcnow().strftime("session-%Y%m%dT%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_LOG_DIR = session_dir
    return session_dir


def _sanitize_host_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe or "host"


def _iter_log_backlog(limit: int = LOG_BACKLOG_LIMIT):
    """Yield recent log entries stored on disk as dictionaries."""
    if CURRENT_LOG_DIR is None:
        return

    combined_path = CURRENT_LOG_DIR / COMBINED_LOG_NAME
    if not combined_path.exists():
        return

    try:
        with combined_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return

    if limit and len(lines) > limit:
        lines = lines[-limit:]

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


async def _collect_process_output(proc: asyncio.subprocess.Process, session_dir: Path):
    """Pipe SSH script output to disk and notify subscribers."""
    global current_proc, log_task

    log_handles: dict[str, any] = {}
    combined_path = session_dir / COMBINED_LOG_NAME
    combined_file = combined_path.open("a", encoding="utf-8")

    async def publish(entry: dict):
        await log_broker.publish(entry)

    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break

            decoded = line.decode(errors="replace").rstrip()
            match = re.match(r"\s*\[(.*?)\]\s*(.*)", decoded)
            if match:
                host, message = match.group(1), match.group(2)
            else:
                host, message = None, decoded

            entry = {
                "host": host,
                "line": message,
                "timestamp": datetime.utcnow().isoformat(timespec="microseconds") + "Z",
            }

            key = _sanitize_host_name(host or "General")
            handle = log_handles.get(key)
            if handle is None:
                file_path = session_dir / f"{key}.txt"
                handle = open(file_path, "a", encoding="utf-8")
                log_handles[key] = handle
            handle.write(message + "\n")
            handle.flush()

            combined_file.write(json.dumps(entry) + "\n")
            combined_file.flush()

            await publish(entry)
    finally:
        combined_file.close()
        for handle in log_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        await proc.wait()
        await log_broker.publish({"type": "end"})
        current_proc = None
        log_task = None

# Background task to monitor status changes
async def monitor_status():
    last_status = {}
    while True:
        try:
            current_status = read_status()
            if current_status != last_status:
                await event_broker.publish({
                    "type": "status",
                    "data": current_status
                })
                last_status = current_status.copy()
        except Exception as e:
            print(f"Error monitoring status: {e}")
        await asyncio.sleep(0.1)  # Check every 100ms

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(monitor_status())
async def _start_ssh_process(timeout: int) -> bool:
    """Start the SSH collection process if not already running."""
    global current_proc, log_task

    async with process_lock:
        if current_proc and current_proc.returncode is None:
            return False

        session_dir = _prepare_log_directory()

        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-u",
            SCRIPT_PATH,
            "--timeout",
            str(timeout),
            "--mode",
            get_current_mode(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        current_proc = proc
        log_task = asyncio.create_task(_collect_process_output(proc, session_dir))
        return True

@app.get("/mode")
def get_mode():
    """Get current sensor mode (mock/real)"""
    return {"mode": get_current_mode()}

@app.get("/ssh-job-status")
def ssh_job_status():
    """
    Return the current SSH job status (state + PID) per device.
    """
    return JSONResponse(read_status())

@app.get("/sessions")
def get_sessions():
    """
    Return all active session IDs and their details.
    """
    sessions = get_active_sessions()
    session_details = {}
    for session_id in sessions:
        hosts = get_session_hosts(session_id)
        session_details[session_id] = {
            "id": session_id,
            "hosts": hosts,
            "host_count": len(hosts),
            "active_count": len([h for h in hosts.values() if h.get("state") in ["running", "connecting", "starting"]])
        }
    return JSONResponse(session_details)

@app.get("/sessions/{session_id}")
def get_session_details(session_id: str):
    """
    Get detailed information about a specific session.
    """
    hosts = get_session_hosts(session_id)
    if not hosts:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return JSONResponse({
        "session_id": session_id,
        "hosts": hosts,
        "logs_available": True
    })

@app.get("/sessions/{session_id}/logs")
def get_session_logs_endpoint(session_id: str, host: str = Query(None)):
    """
    Get logs for a session, optionally filtered by host.
    """
    logs = get_session_logs(session_id, host)
    if not logs:
        raise HTTPException(status_code=404, detail="No logs found for this session")
    
    return JSONResponse(logs)

@app.get("/current-session")
def get_current_session():
    """
    Get the current session ID and its status.
    """
    session_id = get_current_session_id()
    hosts = get_session_hosts(session_id)
    return JSONResponse({
        "session_id": session_id,
        "hosts": hosts,
        "is_current": True
    })


@app.get("/sensors")
def sensor_list():
    """Expose the current sensor configuration without credentials."""
    return JSONResponse(get_public_sensor_config())


@app.on_event("shutdown")
async def cleanup_child():
    """Called when the FastAPI app is shutting down (e.g. on Ctrl-C)."""
    await _stop_current_process()

@app.post("/start-ssh/start")
async def start_ssh_job(timeout: int = Query(600, ge=1)):
    """Start the SSH streaming process if it is not already running."""
    started = await _start_ssh_process(timeout)
    status = "started" if started else "already-running"
    return JSONResponse({"status": status, "timeout": timeout, "mode": get_current_mode()})

@app.get("/start-ssh/logs")
async def stream_logs():
    """
    SSE endpoint:
      • sends a huge retry so the client won’t auto-reconnect automatically
      • streams cached log backlog followed by live updates
      • emits a custom `end` event when the process exits
    """
    queue = await log_broker.subscribe()

    async def event_generator():
        # keep-alive and prevent auto-reconnect
        yield {"retry": 2147483647}

        # send backlog first (oldest to newest)
        for entry in _iter_log_backlog():
            yield json.dumps(entry)

        try:
            while True:
                message = await queue.get()
                if message.get("type") == "end":
                    yield {"event": "end", "data": ""}
                    break
                yield json.dumps(message)
        finally:
            await log_broker.unsubscribe(queue)

    return EventSourceResponse(event_generator(), headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.get("/start-ssh/logs/archive")
def download_log_archive():
    """Return a ZIP archive containing the most recent session logs."""
    if CURRENT_LOG_DIR is None or not CURRENT_LOG_DIR.exists():
        raise HTTPException(status_code=404, detail="No SSH logs available")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for log_path in CURRENT_LOG_DIR.glob("*"):
                if not log_path.is_file():
                    continue
                zf.write(log_path, arcname=log_path.name)

        filename = f"{CURRENT_LOG_DIR.name}.zip"
        cleanup = BackgroundTask(lambda path=tmp_path: os.unlink(path))
        return FileResponse(
            tmp_path,
            media_type="application/zip",
            filename=filename,
            background=cleanup,
        )
    except Exception as exc:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to build log archive: {exc}")

async def _stop_current_process() -> bool:
    """Terminate the running SSH process if present."""
    global log_task, current_proc

    async with process_lock:
        proc = current_proc
        if not proc or proc.returncode is not None:
            return False

        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        current_proc = None

    if log_task:
        try:
            await log_task
        except Exception:
            pass

    return True


@app.post("/start-ssh/stop")
async def stop_script():
    """Manually terminate the SSH script if it’s running."""
    stopped = await _stop_current_process()
    if not stopped:
        raise HTTPException(status_code=404, detail="No active process to stop")
    return {"status": "stopped"}

@app.get("/health")
async def get_health_status():
    """
    Runs the health_check.py script as a subprocess, captures its stdout,
    parses JSON, and returns it. If it fails, returns 500.
    """
    if not os.path.isfile(HEALTH_SCRIPT):
        raise HTTPException(status_code=500, detail="health_check.py not found")

    # Run the script in its own directory to ensure imports and paths work
    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", HEALTH_SCRIPT,
        cwd=os.path.dirname(HEALTH_SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        # Include stderr in the error response for debugging
        raise HTTPException(
            status_code=500,
            detail=f"health_check failed: {stderr.decode(errors='ignore')}"
        )

    try:
        statuses = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON from health_check: {e}"
        )

    return JSONResponse(statuses)


DATA_SCRIPT = os.path.normpath("../street-aware-scripts/data_download.py")
LIST_FOLDERS_SCRIPT = os.path.normpath("../street-aware-scripts/list_remote_folders.py")
current_download_proc = None

def _terminate_download():
    global current_download_proc
    if current_download_proc and current_download_proc.returncode is None:
        current_download_proc.kill()
        current_download_proc = None

@app.on_event("shutdown")
def cleanup_download():
    _terminate_download()

async def run_download_script(manual_config: dict | None = None):
    global current_download_proc
    # Build args; if manual, add --manual and provide JSON via stdin
    args = ["python3", "-u", DATA_SCRIPT]
    stdin_pipe = None
    if manual_config is not None:
        args.append("--manual")
        stdin_pipe = asyncio.subprocess.PIPE

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=os.path.dirname(DATA_SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin_pipe,
    )
    current_download_proc = proc

    # If manual, send JSON payload then close stdin
    if manual_config is not None and proc.stdin is not None:
        payload = json.dumps(manual_config).encode()
        proc.stdin.write(payload)
        try:
            await proc.stdin.drain()
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass

    try:
        # Read each line as it comes, emit as “data: <line>\n\n”
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            yield raw.decode(errors="replace").rstrip()
    finally:
        await proc.wait()
        current_download_proc = None

@app.get("/download-data")
async def download_data_sse():
    """
    SSE endpoint that streams each stdout line from data_download.py as soon as it appears.
    React should use `onmessage` to receive these lines and `addEventListener("end")` to know when to close.
    """


    if not os.path.isfile(DATA_SCRIPT):
        raise HTTPException(status_code=500, detail="data_download.py not found")
    
    async def event_generator():
        # stream log data
        async for log_line in run_download_script():
            yield f"{log_line}\n\n"

        # signal the client we’re done
        yield "event: end\n"
        yield "data:\n\n"

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/download-data/manual")
async def download_data_manual_sse_get(sel: str = Query("")):
    """
    GET variant for SSE. 'sel' should be a base64-encoded JSON mapping of
    display_name -> remote folder name.
    """
    if not os.path.isfile(DATA_SCRIPT):
        raise HTTPException(status_code=500, detail="data_download.py not found")

    selection: dict
    if sel:
        try:
            decoded = base64.b64decode(sel).decode()
            selection = json.loads(decoded)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid sel payload: {e}")
    else:
        selection = {}

    async def event_generator():
        async for log_line in run_download_script(selection or {}):
            yield f"{log_line}\n\n"
        yield "event: end\n"
        yield "data:\n\n"

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.post("/download-data/manual")
async def download_data_manual_sse(selection: dict = Body(...)):
    """
    SSE endpoint for manual download. Body should be a JSON mapping of
    display_name -> remote folder name (e.g., "DATE_10_02_2025_TIME_143000").
    Only the listed sensors will be downloaded.
    """
    if not os.path.isfile(DATA_SCRIPT):
        raise HTTPException(status_code=500, detail="data_download.py not found")

    async def event_generator():
        async for log_line in run_download_script(selection or {}):
            yield f"{log_line}\n\n"
        yield "event: end\n"
        yield "data:\n\n"

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@app.get("/remote-folders")
async def get_remote_folders():
    """
    Return available remote data folders per sensor by invoking list_remote_folders.py.
    """
    if not os.path.isfile(LIST_FOLDERS_SCRIPT):
        raise HTTPException(status_code=500, detail="list_remote_folders.py not found")

    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", LIST_FOLDERS_SCRIPT,
        cwd=os.path.dirname(LIST_FOLDERS_SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"list_remote_folders failed: {stderr.decode(errors='ignore')}")
    try:
        data = json.loads(stdout.decode())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON from list_remote_folders")
    return JSONResponse(data)

@app.post("/download-data/stop")
def stop_download():
    """
    Optionally allow the UI to kill the download early. React can POST here.
    """
    if current_download_proc and current_download_proc.returncode is None:
        _terminate_download()
        return {"status": "stopping"}
    raise HTTPException(status_code=404, detail="No active download to stop")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )
