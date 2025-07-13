#!/usr/bin/env python3

import os
import asyncio
import json
import sys

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "street-aware-scripts"))
sys.path.insert(0, SCRIPT_DIR)

from job_status_tracker import read_status
sys.path.remove(SCRIPT_DIR)


from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Relative path to your SSH script
SCRIPT_PATH = os.path.normpath("../street-aware-scripts/ssh_multiple_run_script.py")

# Holds the currently running subprocess
current_proc = None

def _terminate_current():
    """Terminate the child if still running."""
    global current_proc
    if current_proc and current_proc.returncode is None:
        current_proc.terminate()
        current_proc = None


@app.get("/ssh-job-status")
def ssh_job_status():
    """
    Return the current SSH job status (state + PID) per device.
    """
    return JSONResponse(read_status())


@app.on_event("shutdown")
def cleanup_child():
    """Called when the FastAPI app is shutting down (e.g. on Ctrl-C)."""
    _terminate_current()

async def run_ssh_script(timeout: int):
    """
    Launch the SSH script in unbuffered mode and yield each stdout line.
    """
    global current_proc
    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", SCRIPT_PATH,
        "--timeout", str(timeout),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    current_proc = proc
    try:
        # Read and yield lines as they appear
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip()
    finally:
        await proc.wait()
        current_proc = None

@app.get("/start-ssh/logs")
async def stream_logs(timeout: int = Query(600, ge=1)):
    """
    SSE endpoint:
      • sends a huge retry so the client won’t auto-reconnect
      • streams each log line as `data: …`
      • emits a custom `end` event when done
    """
    async def event_generator():
        # keep-alive and prevent auto-reconnect
        yield "retry: 2147483647\n\n"

        # stream log data
        async for log_line in run_ssh_script(timeout):
            yield f"data: {log_line}\n\n"

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

@app.post("/start-ssh/stop")
def stop_script():
    """
    Manually terminate the SSH script if it’s running.
    """
    if current_proc and current_proc.returncode is None:
        current_proc.terminate()
        return {"status": "stopping"}
    raise HTTPException(status_code=404, detail="No active process to stop")

HEALTH_SCRIPT = os.path.normpath("../street-aware-scripts/health_check.py")

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
current_download_proc = None

def _terminate_download():
    global current_download_proc
    if current_download_proc and current_download_proc.returncode is None:
        current_download_proc.kill()
        current_download_proc = None

@app.on_event("shutdown")
def cleanup_download():
    _terminate_download()

async def run_download_script():
    global current_download_proc
    # Spawn data_download.py with -u (unbuffered), so every print(..., flush=True) shows up immediately
    proc = await asyncio.create_subprocess_exec(
        "python3", "-u", DATA_SCRIPT,
        cwd=os.path.dirname(DATA_SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    current_download_proc = proc

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
