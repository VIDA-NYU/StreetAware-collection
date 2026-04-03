#!/usr/bin/env python3

"""Backend service for sensor data collection.

Simplified version that:
- Starts/stops sensor processes
- Monitors status in real-time via polling
- Supports session resume after disconnect
- Downloads logs from sensors after recording completes
"""

import os
import asyncio
import json
import signal
import sys
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "street-aware-scripts"))
sys.path.insert(0, SCRIPT_DIR)

from job_status_tracker import read_status, read_status_with_staleness_check, get_active_sessions, get_session_hosts, get_session_logs, get_current_session_id, create_new_session
from config import get_current_mode, get_public_sensor_config, get_sensor_config

sys.path.remove(SCRIPT_DIR)

from fastapi import FastAPI, Query, HTTPException, Body
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from sse_starlette.sse import EventSourceResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:3000", "http://127.0.0.1:4000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Relative paths to scripts
SCRIPT_PATH = os.path.normpath("../street-aware-scripts/ssh_multiple_run_script.py")
HEALTH_SCRIPT = os.path.normpath("../street-aware-scripts/health_check.py")

# Holds the currently running subprocess
current_proc = None
process_lock = asyncio.Lock()


async def _start_ssh_process(timeout: int) -> tuple[bool, str]:
    """Start the SSH collection process if not already running.
    
    Returns:
        (started, session_id): Whether process was started and the session ID
    """
    global current_proc

    async with process_lock:
        if current_proc is not None and current_proc.returncode is None:
            return False, get_current_session_id()

        # Create a new session and clear old status data
        session_id = create_new_session()

        current_proc = await asyncio.create_subprocess_exec(
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
        
        # Start a background task to wait for process completion
        asyncio.create_task(_wait_for_process(current_proc))
        return True, session_id


async def _wait_for_process(proc):
    """Wait for process to complete and clean up."""
    global current_proc
    await proc.wait()
    if current_proc == proc:
        current_proc = None


async def _start_ssh_resume_process(timestamp: str) -> bool:
    """Start the SSH script in resume mode to reconnect to running sensors."""
    global current_proc

    async with process_lock:
        if current_proc and current_proc.returncode is None:
            return False

        current_proc = await asyncio.create_subprocess_exec(
            "python3",
            "-u",
            SCRIPT_PATH,
            "--resume",
            "--timestamp",
            timestamp,
            "--mode",
            get_current_mode(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        asyncio.create_task(_wait_for_process(current_proc))
        return True


async def _stop_current_process(kill_sensors: bool = False) -> bool:
    """Terminate the running SSH process if present.
    
    Args:
        kill_sensors: If True, also stop the remote sensor processes.
                     If False, just stop monitoring (sensors keep running).
    """
    global current_proc

    async with process_lock:
        proc = current_proc
        if not proc or proc.returncode is not None:
            return False

        if kill_sensors:
            # Send SIGTERM which will trigger sensor termination
            proc.terminate()
        else:
            # Send SIGINT which will just stop monitoring without killing sensors
            proc.send_signal(signal.SIGINT)
        
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        current_proc = None

    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/mode")
def get_mode():
    """Get current sensor mode (mock/real)"""
    return {"mode": get_current_mode()}


def _check_remote_pid(sensor: dict, pid: int) -> bool:
    """Check if a PID is still running on a remote sensor."""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            sensor["host"],
            port=sensor.get("port", 22),
            username=sensor["username"],
            password=sensor["password"],
            timeout=5
        )
        
        stdin, stdout, stderr = client.exec_command(
            f"kill -0 {pid} 2>/dev/null && echo running || echo stopped",
            timeout=5
        )
        result = stdout.read().decode().strip()
        client.close()
        return result == "running"
    except Exception as e:
        print(f"[DEBUG] Error checking PID {pid} on {sensor.get('host')}: {e}")
        return False


def _verify_and_update_stale_status():
    """Check stale entries and verify if remote processes are still running.
    
    Updates status to 'completed' if process is no longer running,
    or refreshes the last_check if still running.
    
    Only verifies entries that have a PID - entries without PID are left as-is
    since we can't verify them.
    """
    from datetime import datetime, timezone
    from job_status_tracker import read_status, update_status
    
    data = read_status_with_staleness_check(stale_threshold_seconds=15)
    sensors = get_sensor_config()
    sensor_map = {s["display_name"]: s for s in sensors}
    
    verified_any = False
    
    for host, record in data.items():
        if not record.get("stale"):
            continue
            
        pid = record.get("pid")
        sensor = sensor_map.get(host)
        
        if not sensor:
            continue
        
        # Only verify if we have a PID to check
        if not pid:
            # No PID - can't verify, leave as-is (don't mark completed)
            # The process might still be starting
            continue
        
        # Check if PID is still running
        is_running = _check_remote_pid(sensor, pid)
        verified_any = True
        
        if is_running:
            # Process still running - update last_check to resume tracking
            update_status(
                host, "running",
                pid=pid,
                last_check=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                note="Resumed tracking after reconnect"
            )
            print(f"[INFO] {host}: PID {pid} still running, resumed tracking")
        else:
            # Process no longer running - mark as completed
            update_status(
                host, "completed",
                pid=pid,
                finished_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                note="Process ended while disconnected"
            )
            print(f"[INFO] {host}: PID {pid} no longer running, marked completed")
    
    return read_status()


@app.get("/ssh-job-status")
def ssh_job_status():
    """Return the current SSH job status per device.
    
    Automatically verifies stale entries by checking if remote PIDs are still running.
    Updates status to 'completed' if process ended, or resumes tracking if still running.
    """
    # First check for stale entries and verify them
    data = read_status_with_staleness_check(stale_threshold_seconds=15)
    
    # If any entries are stale, verify and update them
    has_stale = any(record.get("stale") for record in data.values())
    if has_stale:
        data = _verify_and_update_stale_status()
    
    return JSONResponse(data)


@app.get("/sensors")
def sensor_list():
    """Expose the current sensor configuration without credentials."""
    return JSONResponse(get_public_sensor_config())


@app.get("/sessions")
def get_sessions():
    """Return all active session IDs and their details."""
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
    """Get detailed information about a specific session."""
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
    """Get session status logs (not sensor output logs)."""
    logs = get_session_logs(session_id, host)
    if not logs:
        raise HTTPException(status_code=404, detail="No logs found for this session")
    
    return JSONResponse(logs)


@app.get("/download-sensor-logs")
async def download_sensor_logs():
    """Download actual sensor output logs from remote machines.
    
    This fetches the log files that were written by the sensor processes
    on the remote machines during the recording session.
    """
    import paramiko
    
    # Get current status to find log file paths and sensor info
    status = read_status()
    if not status:
        raise HTTPException(status_code=404, detail="No session data found")
    
    # Get the collection timestamp from any sensor
    timestamp = None
    for sensor_data in status.values():
        if sensor_data.get("timestamp"):
            timestamp = sensor_data["timestamp"]
            break
    
    if not timestamp:
        raise HTTPException(status_code=404, detail="No collection timestamp found")
    
    # Get sensor config for connection details (includes credentials)
    sensor_config = get_sensor_config()
    mode = get_current_mode()
    
    all_logs = {}
    
    for sensor_name, sensor_data in status.items():
        log_file = sensor_data.get("log_file")
        if not log_file:
            continue
        
        # Find matching sensor config
        sensor_info = None
        for s in sensor_config:
            display_name = s.get("display_name") or s.get("name", "")
            if display_name == sensor_name:
                sensor_info = s
                break
        
        if not sensor_info:
            all_logs[sensor_name] = f"[Error: Could not find sensor config for {sensor_name}]"
            continue
        
        try:
            # Connect to sensor and fetch log file
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            host = "localhost" if mode == "mock" else sensor_info.get("host")
            port = sensor_info.get("port", 22)
            username = sensor_info.get("username")
            password = sensor_info.get("password")
            
            client.connect(hostname=host, port=port, username=username, password=password, timeout=10)
            
            # Read the log file
            stdin, stdout, stderr = client.exec_command(f"cat {log_file} 2>/dev/null")
            log_content = stdout.read().decode('utf-8', errors='replace')
            error_content = stderr.read().decode('utf-8', errors='replace')
            
            client.close()
            
            if log_content:
                all_logs[sensor_name] = log_content
            elif error_content:
                all_logs[sensor_name] = f"[Error reading log: {error_content}]"
            else:
                all_logs[sensor_name] = "[Log file is empty or does not exist]"
                
        except Exception as e:
            all_logs[sensor_name] = f"[Error connecting to sensor: {str(e)}]"
    
    if not all_logs:
        raise HTTPException(status_code=404, detail="No logs could be retrieved from sensors")
    
    return JSONResponse({
        "timestamp": timestamp,
        "logs": all_logs
    })


@app.get("/current-session")
def get_current_session():
    """Get the current session ID and its status."""
    session_id = get_current_session_id()
    hosts = get_session_hosts(session_id)
    return JSONResponse({
        "session_id": session_id,
        "hosts": hosts,
        "is_current": True
    })


@app.post("/start-ssh/start")
async def start_ssh_job(timeout: int = Query(600, ge=1)):
    """Start the SSH process if it is not already running."""
    started, session_id = await _start_ssh_process(timeout)
    status = "started" if started else "already-running"
    return JSONResponse({
        "status": status, 
        "timeout": timeout, 
        "mode": get_current_mode(),
        "session_id": session_id
    })


@app.post("/start-ssh/resume")
async def resume_ssh_job(timestamp: str = Query(..., description="Collection timestamp to resume")):
    """Resume monitoring existing sensor processes."""
    started = await _start_ssh_resume_process(timestamp)
    if started:
        return JSONResponse({"status": "resumed", "timestamp": timestamp, "mode": get_current_mode()})
    else:
        return JSONResponse({"status": "already-running", "timestamp": timestamp, "mode": get_current_mode()})


@app.post("/start-ssh/stop")
async def stop_script():
    """Manually terminate the SSH script and stop remote sensors."""
    stopped = await _stop_current_process(kill_sensors=True)
    if not stopped:
        raise HTTPException(status_code=404, detail="No active process to stop")
    return {"status": "stopped"}


@app.get("/start-ssh/running")
async def is_running():
    """Check if the SSH process is currently running."""
    running = current_proc is not None and current_proc.returncode is None
    return {"running": running}


@app.on_event("shutdown")
async def cleanup_child():
    """Called when the FastAPI app is shutting down.
    
    We do NOT kill sensors here - they should keep running independently.
    """
    await _stop_current_process(kill_sensors=False)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def get_health_status():
    """Run health_check.py and return results."""
    if not os.path.isfile(HEALTH_SCRIPT):
        raise HTTPException(status_code=500, detail="health_check.py not found")

    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", HEALTH_SCRIPT,
        cwd=os.path.dirname(HEALTH_SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
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


# ---------------------------------------------------------------------------
# Camera preview
# ---------------------------------------------------------------------------

import paramiko
import io

def _capture_camera_frame(sensor: dict, camera_dev: int) -> tuple[bytes | None, str | None]:
    """Capture a single frame from a camera on a remote sensor.
    
    Uses GStreamer to capture one JPEG frame via SSH.
    Returns (jpeg_bytes, error_message).
    """
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            sensor["host"],
            port=sensor.get("port", 22),
            username=sensor["username"],
            password=sensor["password"],
            timeout=10
        )
        
        # GStreamer pipeline to capture one JPEG frame
        # v4l2src captures from camera, jpegenc encodes to JPEG, fdsink outputs to stdout
        gst_cmd = (
            f"gst-launch-1.0 -q v4l2src device=/dev/video{camera_dev} num-buffers=1 "
            f"! videoconvert ! videoscale ! video/x-raw,width=640,height=480 "
            f"! jpegenc quality=85 ! fdsink"
        )
        
        stdin, stdout, stderr = client.exec_command(gst_cmd, timeout=15)
        stdout.channel.settimeout(10.0)
        
        # Read the JPEG data
        jpeg_data = stdout.read()
        error_output = stderr.read().decode(errors='ignore').strip()
        
        client.close()
        
        if jpeg_data and len(jpeg_data) > 100:  # Valid JPEG should be > 100 bytes
            return jpeg_data, None
        else:
            return None, error_output or "No image data received"
            
    except Exception as e:
        return None, str(e)


@app.get("/camera/preview/{sensor_name}/{camera_id}")
async def get_camera_preview(sensor_name: str, camera_id: int):
    """Get a preview image from a specific camera on a sensor.
    
    Args:
        sensor_name: The sensor name (e.g., "sensor-184")
        camera_id: The camera number (0 or 1 for each sensor)
    
    Returns:
        Base64 encoded JPEG image or error message.
    """
    # Check if recording is in progress
    if current_proc is not None and current_proc.returncode is None:
        raise HTTPException(status_code=409, detail="Cannot preview while recording is in progress")
    
    # Find the sensor config
    sensors = get_sensor_config()
    sensor = next((s for s in sensors if s["name"] == sensor_name), None)
    
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_name}' not found")
    
    # Map camera_id to device number (each sensor has 2 cameras)
    # Camera 0 = /dev/video0, Camera 1 = /dev/video2
    device_map = {0: 0, 1: 2}
    if camera_id not in device_map:
        raise HTTPException(status_code=400, detail="camera_id must be 0 or 1")
    
    device_num = device_map[camera_id]
    
    # Capture the frame
    jpeg_data, error = await asyncio.get_event_loop().run_in_executor(
        None, _capture_camera_frame, sensor, device_num
    )
    
    if error:
        return JSONResponse({
            "success": False,
            "error": error,
            "sensor": sensor["display_name"],
            "camera_id": camera_id
        }, status_code=500)
    
    # Return base64 encoded image
    return JSONResponse({
        "success": True,
        "image": base64.b64encode(jpeg_data).decode('ascii'),
        "sensor": sensor["display_name"],
        "camera_id": camera_id
    })


@app.get("/camera/preview-all")
async def get_all_camera_previews():
    """Get preview images from all cameras on all sensors.
    
    Returns a list of preview results for each camera.
    """
    # Check if recording is in progress
    if current_proc is not None and current_proc.returncode is None:
        raise HTTPException(status_code=409, detail="Cannot preview while recording is in progress")
    
    sensors = get_sensor_config()
    results = []
    
    # Capture from all cameras in parallel
    async def capture_sensor_cameras(sensor):
        sensor_results = []
        for camera_id in [0, 1]:
            device_num = 0 if camera_id == 0 else 2
            jpeg_data, error = await asyncio.get_event_loop().run_in_executor(
                None, _capture_camera_frame, sensor, device_num
            )
            
            if error:
                sensor_results.append({
                    "success": False,
                    "error": error,
                    "sensor": sensor["display_name"],
                    "sensor_name": sensor["name"],
                    "camera_id": camera_id
                })
            else:
                sensor_results.append({
                    "success": True,
                    "image": base64.b64encode(jpeg_data).decode('ascii'),
                    "sensor": sensor["display_name"],
                    "sensor_name": sensor["name"],
                    "camera_id": camera_id
                })
        return sensor_results
    
    # Run all sensor captures in parallel
    all_results = await asyncio.gather(*[capture_sensor_cameras(s) for s in sensors])
    
    # Flatten results
    for sensor_results in all_results:
        results.extend(sensor_results)
    
    return JSONResponse({"cameras": results})


# ---------------------------------------------------------------------------
# Data download endpoints (kept from original)
# ---------------------------------------------------------------------------

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
    """SSE endpoint that streams download progress."""
    if not os.path.isfile(DATA_SCRIPT):
        raise HTTPException(status_code=500, detail="data_download.py not found")
    
    async def event_generator():
        async for log_line in run_download_script():
            yield f"{log_line}\n\n"
        yield "event: end\n"
        yield "data:\n\n"

    return EventSourceResponse(
        event_generator(),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/download-data/manual")
async def download_data_manual_sse_get(sel: str = Query("")):
    """GET variant for manual download SSE."""
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/download-data/manual")
async def download_data_manual_sse(selection: dict = Body(...)):
    """SSE endpoint for manual download."""
    if not os.path.isfile(DATA_SCRIPT):
        raise HTTPException(status_code=500, detail="data_download.py not found")

    async def event_generator():
        async for log_line in run_download_script(selection or {}):
            yield f"{log_line}\n\n"
        yield "event: end\n"
        yield "data:\n\n"

    return EventSourceResponse(
        event_generator(),
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/remote-folders")
async def get_remote_folders():
    """Return available remote data folders per sensor."""
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
    """Kill the download early."""
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
