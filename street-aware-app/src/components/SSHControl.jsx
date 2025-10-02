import React, { useState, useRef, useEffect } from "react";
import JSZip from "jszip";
import { fetchSensors } from "../utils/sensorConfig";

const MAX_DISPLAY_LINES = 10;
const RECONNECT_DELAY_MS = 2000;


export default function SSHControl() {
  const [timeoutSec, setTimeoutSec] = useState(60);
  const [running, setRunning] = useState(false);
  const [logsByHost, setLogsByHost] = useState({});
  const [panelOpen, setPanelOpen] = useState(false);
  const [sensors, setSensors] = useState([]);

  // Initialize sensor hosts from config
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
  useEffect(() => {
    if (!sensors.length) return;
    setLogsByHost(prev => {
      const next = { ...prev };
      sensors.forEach(({ display_name }) => {
        if (!next[display_name]) {
          next[display_name] = [];
        }
      });
      if (!next['General']) {
        next['General'] = [];
      }
      return next;
    });
  }, [sensors]);
  const esRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(false);

  const [jobStatus, setJobStatus] = useState({});

  const downloadLogsAsZip = async (useFullLogs = true) => {
    if (useFullLogs) {
      try {
        const response = await fetch("http://localhost:8080/start-ssh/logs/archive");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "logs-full.zip";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } catch (error) {
        console.error("Failed to download full logs:", error);
        window.alert("Full log download is not available yet. Ensure a session is running or has completed.");
      }
      return;
    }

    const zip = new JSZip();

    for (const sensor of sensors) {
      const host = sensor.display_name;
      const lines = logsByHost[host] || [];
      if (lines.length) {
        zip.file(`${host}.txt`, lines.join("\n"));
      }
    }

    if (logsByHost["General"]?.length) {
      zip.file("General.txt", logsByHost["General"].join("\n"));
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "logs.zip";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };


  const attachLogStream = (reset = true) => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (reset) {
      setLogsByHost({});
    }

    const es = new EventSource("http://localhost:8080/start-ssh/logs");

    es.onmessage = (e) => {
      if (!e.data) {
        return;
      }
      try {
        const parsed = JSON.parse(e.data);
        if (parsed.type === "end") {
          return;
        }

        const host = parsed.host || 'General';
        const message = parsed.line ?? '';

        console.log('[SSHControl] log', { host, message });

        setLogsByHost(prev => {
          const next = { ...prev };
          const lines = next[host] ? [...next[host], message] : [message];
          next[host] = lines.slice(-MAX_DISPLAY_LINES);
          return next;
        });
      } catch (error) {
        console.error("Failed to process log entry:", error);
        console.log('[SSHControl] raw event data', e.data);
      }
    };

    es.onerror = () => {
      es.close();
      esRef.current = null;
      if (!shouldReconnectRef.current) {
        setRunning(false);
        return;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      reconnectTimerRef.current = setTimeout(() => {
        attachLogStream(false);
      }, RECONNECT_DELAY_MS);
    };

    es.addEventListener("end", () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      es.close();
      esRef.current = null;
      setRunning(false);
      if (sensors.length) {
        downloadLogsAsZip(true);
      }
    });

    esRef.current = es;
  };


  // Poll job status while running
  useEffect(() => {
    if (!running) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8080/ssh-job-status");
        const data = await res.json();
        setJobStatus(data);
      } catch (err) {
        console.error("Failed to fetch job status:", err);
      }
    }, 3000); // poll every 3s while running

    return () => clearInterval(interval);
  }, [running]);


  // Start the SSH job and open SSE stream
  const startJob = async () => {
    if (running) return;

    setPanelOpen(true);
    setLogsByHost({});

    try {
      const response = await fetch(`http://localhost:8080/start-ssh/start?timeout=${timeoutSec}`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (data.status !== "started") {
        console.info("SSH job already active; attaching to existing session.");
      }
      shouldReconnectRef.current = true;
      setRunning(true);
      attachLogStream(true);
    } catch (error) {
      console.error("Failed to start or attach to SSH job:", error);
      shouldReconnectRef.current = false;
      setRunning(false);
      window.alert("Unable to start or attach to SSH job. Check the backend service.");
    }
  };

  const resumeLogs = async () => {
    setPanelOpen(true);

    let hasActiveJob = true;
    try {
      const res = await fetch("http://localhost:8080/ssh-job-status");
      if (res.ok) {
        const data = await res.json();
        setJobStatus(data);
        hasActiveJob = Object.values(data).some((info) =>
          info && ["running", "starting", "connecting"].includes(info.state)
        );
      }
    } catch (error) {
      console.error("Failed to fetch job status for resume:", error);
      hasActiveJob = false;
    }

    shouldReconnectRef.current = hasActiveJob;
    setRunning(hasActiveJob);
    attachLogStream(true);
  };

  // Send the stop command to backend; server will emit 'end'
  const stopJob = async () => {
    if (!running) return;

    shouldReconnectRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    try {
      const res = await fetch("http://localhost:8080/start-ssh/stop", {
        method: "POST",
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      console.error("Failed to stop job:", err);
    }

    // Allow EventSource 'end' handler to close the stream and update state
  };

  // Auto-scroll as new logs arrive (overall container)
  useEffect(() => {
    if (!panelOpen) return;
    const container = document.getElementById('overall-log-container');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [logsByHost, panelOpen]);

  // Cleanup if component unmounts
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      shouldReconnectRef.current = false;
    };
  }, []);

  useEffect(() => {
  if (!panelOpen) return;

  requestAnimationFrame(() => {
    sensors.forEach((sensor) => {
      const host = sensor.display_name;
      const el = document.getElementById(`log-scroll-${host}`);
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
    const generalEl = document.getElementById('log-scroll-General');
    if (generalEl) {
      generalEl.scrollTop = generalEl.scrollHeight;
    }
  });
}, [logsByHost, panelOpen, sensors]);
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">SSH Data Collection</h2>
          <p className="text-sm text-gray-600">Start SSH sessions and collect data from connected devices</p>
        </div>
        {running && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-600 font-medium">Active</span>
          </div>
        )}
      </div>

      <div className="form-group">
        <label className="form-label">
          Session Timeout (seconds)
        </label>
        <input
          type="number"
          min="1"
          value={timeoutSec}
          onChange={(e) => setTimeoutSec(Number(e.target.value))}
          className="form-input"
          style={{ width: "120px" }}
        />
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start SSH & Collect
            </>
          )}
        </button>

        <button
          onClick={stopJob}
          disabled={!running}
          className="btn btn-danger"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          Stop Job
        </button>

        <button
          onClick={resumeLogs}
          disabled={running}
          className="btn btn-secondary"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 21l-6-6 6-6m8 12l-6-6 6-6" />
          </svg>
          Resume Session
        </button>

        <button
          onClick={() => setPanelOpen(!panelOpen)}
          className="btn btn-secondary"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {panelOpen ? "Hide Logs" : "Show Logs"}
        </button>

        <button
          onClick={() => downloadLogsAsZip(true)}
          disabled={Object.keys(logsByHost).length === 0}
          className="btn btn-success"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Download Logs
        </button>
      </div>

      {running && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Live Job Status</h4>
          <div className="bg-gray-50 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Host
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Process ID
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {Object.entries(jobStatus).map(([host, { state, pid }]) => (
                  <tr key={host} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">
                      {host}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`status-indicator status-${state === "running" ? "up" : state === "terminated" ? "down" : "pending"} capitalize`}>
                        {state}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500 font-mono">
                      {pid ?? "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}


      {panelOpen && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900">Live Logs</h4>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Real-time streaming</span>
            </div>
          </div>
          
          <div
            id="overall-log-container"
            className="log-grid bg-gray-900 rounded-lg p-4"
          >
            {sensors.map((sensor) => {
              const host = sensor.display_name;
              const lines = logsByHost[host] || [];
              return (
                <div
                  key={host}
                  className="log-panel bg-gray-800 rounded-lg p-4"
                >
                  <div className="flex items-center justify-between mb-3 sticky top-0 bg-gray-800 z-10 py-2">
                    <h5 className="font-semibold text-yellow-400">{host}</h5>
                    <span className="text-xs text-gray-400">
                      {lines.length} lines
                    </span>
                  </div>
                  <div className="log-panel-body text-sm text-gray-300" id={`log-scroll-${host}`}>
                    {lines.length === 0 ? (
                      <div className="text-gray-500 italic">Waiting for logs...</div>
                    ) : (
                      lines.map((line, i) => (
                        <div key={`${host}-${i}`} className="mb-1 leading-relaxed whitespace-pre-wrap">
                          {line}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              );
            })}

            {logsByHost['General'] && (
              <div className="log-panel bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3 sticky top-0 bg-gray-800 z-10 py-2">
                  <h5 className="font-semibold text-yellow-400">General</h5>
                  <span className="text-xs text-gray-400">
                    {logsByHost['General'].length} lines
                  </span>
                </div>
                <div className="log-panel-body text-sm text-gray-300" id="log-scroll-General">
                  {logsByHost['General'].map((line, i) => (
                    <div key={`general-${i}`} className="mb-1 leading-relaxed whitespace-pre-wrap">
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
