import React, { useState, useEffect, useCallback } from "react";
import { fetchSensors } from "../utils/sensorConfig";

const STATUS_POLL_INTERVAL = 2000; // Poll status every 2 seconds

export default function SSHControl() {
  const [timeoutSec, setTimeoutSec] = useState(60);
  const [running, setRunning] = useState(false);
  const [sensors, setSensors] = useState([]);
  const [jobStatus, setJobStatus] = useState({});
  const [lastTimestamp, setLastTimestamp] = useState(localStorage.getItem('lastCollectionTimestamp') || '');
  const [resumeTimestamp, setResumeTimestamp] = useState('');
  const [downloading, setDownloading] = useState(false);

  // Load sensor configuration
  useEffect(() => {
    async function loadSensorHosts() {
      try {
        const config = await fetchSensors();
        setSensors(config);
      } catch (error) {
        console.error("Failed to load sensor hosts:", error);
      }
    }
    loadSensorHosts();
  }, []);

  // Poll job status
  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch("/ssh-job-status");
      if (res.ok) {
        const data = await res.json();
        setJobStatus(data);
        
        // Check if any sensors are still running (and not stale)
        const hasRunning = Object.values(data).some(info => 
          info && 
          ["running", "connecting", "starting", "reconnecting"].includes(info.state) &&
          !info.stale  // Don't count stale entries as running
        );
        
        // Check if all sensors completed (or are stale)
        const allCompleted = Object.values(data).length > 0 && 
          Object.values(data).every(info => 
            info && (
              ["completed", "terminated", "stopped", "failed", "disconnected"].includes(info.state) ||
              info.stale  // Stale entries are effectively not running
            )
          );
        
        if (allCompleted && running) {
          setRunning(false);
        } else if (hasRunning && !running) {
          setRunning(true);
        }
        
        // Extract timestamp for resume capability
        const firstSensor = Object.values(data)[0];
        if (firstSensor?.timestamp && firstSensor.timestamp !== lastTimestamp) {
          setLastTimestamp(firstSensor.timestamp);
          localStorage.setItem('lastCollectionTimestamp', firstSensor.timestamp);
        }
      }
    } catch (err) {
      console.error("Failed to fetch job status:", err);
    }
  }, [running, lastTimestamp]);

  // Poll status periodically
  useEffect(() => {
    pollStatus(); // Initial poll
    const interval = setInterval(pollStatus, STATUS_POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [pollStatus]);

  // Start the SSH job
  const startJob = async () => {
    if (running) return;

    try {
      const response = await fetch(`/start-ssh/start?timeout=${timeoutSec}`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      console.info("Start response:", data);
      setRunning(true);
      setJobStatus({});
    } catch (error) {
      console.error("Failed to start SSH job:", error);
      window.alert("Unable to start SSH job. Check the backend service.");
    }
  };

  // Resume monitoring existing sensors
  const resumeSession = async () => {
    const tsToUse = resumeTimestamp || lastTimestamp;
    if (!tsToUse) {
      window.alert("No timestamp available for resume. Please enter a collection timestamp.");
      return;
    }

    try {
      const response = await fetch(`/start-ssh/resume?timestamp=${encodeURIComponent(tsToUse)}`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      console.info("Resume response:", data);
      setRunning(true);
    } catch (error) {
      console.error("Failed to resume SSH job:", error);
      window.alert("Unable to resume SSH job. Make sure the timestamp is correct and sensors are still running.");
    }
  };

  // Stop the SSH job
  const stopJob = async () => {
    if (!running) return;

    try {
      const res = await fetch("/start-ssh/stop", {
        method: "POST",
      });
      if (!res.ok && res.status !== 404) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      console.error("Failed to stop job:", err);
    }
    setRunning(false);
  };

  // Download logs from sensors
  const downloadLogs = async () => {
    // Check if we have sensors with log files
    const sensorsWithLogs = Object.entries(jobStatus).filter(([_, info]) => 
      info && info.log_file
    );

    if (sensorsWithLogs.length === 0) {
      window.alert("No recordings with logs available for download.");
      return;
    }

    setDownloading(true);
    try {
      // Use the new endpoint to fetch actual sensor logs from remote machines
      const response = await fetch("/download-sensor-logs");
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      // Create a text file with all logs
      const logContent = Object.entries(data.logs)
        .map(([host, content]) => `${"=".repeat(60)}\n=== ${host} ===\n${"=".repeat(60)}\n\n${content}`)
        .join('\n\n');
      
      const blob = new Blob([logContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sensor-logs-${data.timestamp || lastTimestamp || 'session'}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download logs:", error);
      window.alert(`Failed to download logs: ${error.message}`);
    } finally {
      setDownloading(false);
    }
  };

  // Get status color
  const getStatusColor = (state, isStale) => {
    if (isStale) return "bg-orange-500";
    switch (state) {
      case "running": return "bg-green-500";
      case "completed": return "bg-blue-500";
      case "connecting":
      case "starting":
      case "reconnecting": return "bg-yellow-500";
      case "terminated":
      case "stopped": return "bg-gray-500";
      case "failed":
      case "disconnected": return "bg-red-500";
      default: return "bg-gray-400";
    }
  };

  const getStatusBadgeClass = (state, isStale) => {
    if (isStale) return "bg-orange-100 text-orange-800";
    switch (state) {
      case "running": return "bg-green-100 text-green-800";
      case "completed": return "bg-blue-100 text-blue-800";
      case "connecting":
      case "starting":
      case "reconnecting": return "bg-yellow-100 text-yellow-800";
      case "terminated":
      case "stopped": return "bg-gray-100 text-gray-800";
      case "failed":
      case "disconnected": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  const getStatusText = (state, isStale) => {
    if (isStale) return "stale (not monitored)";
    return state || "unknown";
  };

  // Check if any sensors have completed
  const hasCompletedSensors = Object.values(jobStatus).some(info => 
    info && ["completed", "terminated", "stopped"].includes(info.state)
  );

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">SSH Data Collection</h2>
          <p className="text-sm text-gray-600">Start sensor recordings and monitor their status</p>
        </div>
        {running && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-600 font-medium">Active</span>
          </div>
        )}
      </div>

      <div className="form-group">
        <label className="form-label">Recording Duration (seconds)</label>
        <input
          type="number"
          min="1"
          value={timeoutSec}
          onChange={(e) => setTimeoutSec(Number(e.target.value))}
          className="form-input"
          style={{ width: "120px" }}
        />
        <p className="text-xs text-gray-500 mt-1">
          Sensors will record for exactly this duration, even if disconnected from this control panel.
        </p>
      </div>

      <div className="form-group">
        <label className="form-label">Resume Timestamp (for reconnecting)</label>
        <input
          type="text"
          value={resumeTimestamp || lastTimestamp}
          onChange={(e) => setResumeTimestamp(e.target.value)}
          placeholder="e.g., DATE_12_04_2025_TIME_14_51_34"
          className="form-input"
          style={{ width: "320px" }}
        />
        {lastTimestamp && (
          <p className="text-xs text-green-600 mt-1">Last session: {lastTimestamp}</p>
        )}
      </div>

      <div className="flex flex-wrap gap-3 mb-6">
        <button
          onClick={startJob}
          disabled={running}
          className="btn btn-primary"
        >
          {running ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              Running…
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Recording
            </>
          )}
        </button>

        <button
          onClick={stopJob}
          disabled={!running}
          className="btn btn-danger"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
          </svg>
          Stop
        </button>

        <button
          onClick={resumeSession}
          disabled={running}
          className="btn btn-secondary"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Resume Session
        </button>

        <button
          onClick={downloadLogs}
          disabled={!hasCompletedSensors || downloading}
          className="btn btn-success"
        >
          {downloading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              Downloading…
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Logs
            </>
          )}
        </button>
      </div>

      {/* Status Table */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Sensor Status</h4>
        <p className="text-sm text-gray-600 mb-3">
          <span className="inline-flex items-center gap-1">
            <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Sensors run independently — they continue recording even if you close this page.
          </span>
        </p>
        
        <div className="bg-gray-50 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sensor
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  PID
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Last Update
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {Object.keys(jobStatus).length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-4 py-8 text-center text-gray-500">
                    No active sessions. Click "Start Recording" to begin.
                  </td>
                </tr>
              ) : (
                Object.entries(jobStatus).map(([host, info]) => {
                  const { state, pid, duration, last_check, started_at, finished_at, stale, stale_seconds } = info || {};
                  const lastUpdate = last_check || finished_at || started_at;
                  const isStale = stale === true;
                  return (
                    <tr key={host} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm font-medium text-gray-900">
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${getStatusColor(state, isStale)}`}></div>
                          {host}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusBadgeClass(state, isStale)}`}>
                          {getStatusText(state, isStale)}
                        </span>
                        {isStale && stale_seconds && (
                          <span className="ml-2 text-xs text-orange-600">
                            ({Math.floor(stale_seconds / 60)}m ago)
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500 font-mono">
                        {pid ?? "—"}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500">
                        {duration ? `${duration}s` : "—"}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500">
                        {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : "—"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
