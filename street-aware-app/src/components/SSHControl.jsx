import React, { useState, useRef, useEffect } from "react";
import JSZip from "jszip";

const MAX_LOG_LINES = 500;

// Add this at the top, before your component
const FIXED_HOSTS = [
  "192.168.0.184",
  "192.168.0.122",
  "192.168.0.108",
  "192.168.0.227",
];


export default function SSHControl() {
  const [timeoutSec, setTimeoutSec] = useState(60);
  const [running, setRunning] = useState(false);
  // logsByHost: { [host: string]: string[] }
  const [logsByHost, setLogsByHost] = useState({});
  const [panelOpen, setPanelOpen] = useState(false);
  const esRef = useRef(null);
  const logContainerRef = useRef(null);

  const [jobStatus, setJobStatus] = useState({});

  const downloadLogsAsZip = async () => {
    const zip = new JSZip();

    // Add each host's logs as a .txt file
    for (const host of FIXED_HOSTS) {
      const lines = logsByHost[host] || [];
      const content = lines.join("\n");
      zip.file(`${host}.txt`, content);
    }

    // Add "General" logs if present
    if (logsByHost["General"]) {
      zip.file("General.txt", logsByHost["General"].join("\n"));
    }

    // Generate zip blob and trigger download
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
  const startJob = () => {
    if (running) return;

    setLogsByHost({});
    setRunning(true);
    setPanelOpen(true);

    const es = new EventSource(
      `http://localhost:8080/start-ssh/logs?timeout=${timeoutSec}`
    );

    es.onmessage = (e) => {
      try {
        // Remove "data: " prefix if present
        let raw = e.data;
        if (raw.startsWith("data: ")) {
          raw = raw.slice(6);
        }
        const parsed = JSON.parse(raw);
        let host = parsed.host ? parsed.host.trim() : 'General';
        if (!FIXED_HOSTS.includes(host) && host !== 'General') {
          host = 'General';
        }
        const line = parsed.line;
        setLogsByHost(prev => {
          const next = { ...prev };
          if (!next[host]) next[host] = [];
          next[host] = [...next[host], line].slice(-MAX_LOG_LINES);
          return next;
        });
      } catch {
        setLogsByHost(prev => {
          const next = { ...prev };
          if (!next['General']) next['General'] = [];
          next['General'] = [...next['General'], e.data].slice(-MAX_LOG_LINES);
          return next;
        });
      }
    };

    // On any error or intentional close from server, stop and never reconnect
    es.onerror = () => {
      es.close();
      esRef.current = null;
      setRunning(false);
    };

    // Listen for custom 'end' event emitted by server when done
    es.addEventListener("end", () => {
      es.close();
      esRef.current = null;
      setRunning(false);
    });

    esRef.current = es;
  };

  // Send the stop command to backend; server will emit 'end'
  const stopJob = async () => {
    if (!running) return;
    try {
      await fetch("http://localhost:8080/start-ssh/stop", {
        method: "POST",
      });
    } catch (err) {
      console.error("Failed to stop job:", err);
    }
    // Do NOT close EventSource here—let onerror or 'end' handler do it
  };

  // Auto-scroll as new logs arrive
  useEffect(() => {
    if (panelOpen && logContainerRef.current) {
      logContainerRef.current.scrollTop =
        logContainerRef.current.scrollHeight;
    }
  }, [logsByHost, panelOpen]);

  // Cleanup if component unmounts
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
  if (!panelOpen) return;

  requestAnimationFrame(() => {
    FIXED_HOSTS.forEach((host) => {
      const el = document.getElementById(`log-scroll-${host}`);
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
  });
}, [logsByHost, panelOpen]);


  console.log("logsByHost", logsByHost);

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        maxWidth: "100%",
        margin: "1rem auto",
        padding: "0 1rem",
      }}
    >
      <div style={{ maxWidth: 800, margin: "0 auto", padding: "0 1rem" }}>
        <label style={{ display: "block", marginBottom: 8 }}>
          Session timeout (seconds):
          <input
            type="number"
            min="1"
            value={timeoutSec}
            onChange={(e) => setTimeoutSec(Number(e.target.value))}
            style={{ marginLeft: 8, width: 80 }}
          />
        </label>

        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={startJob}
            disabled={running}
            style={{
              padding: "6px 12px",
              cursor: running ? "not-allowed" : "pointer",
            }}
          >
            {running ? "Running…" : "Start SSH & Collect"}
          </button>

          <button
            onClick={stopJob}
            disabled={!running}
            style={{
              padding: "6px 12px",
              background: "#e74c3c",
              color: "white",
            }}
          >
            Stop Job
          </button>

          <button
            onClick={() => setPanelOpen(!panelOpen)}
            style={{ marginLeft: "auto", padding: "6px 12px" }}
          >
            {panelOpen ? "Hide Logs" : "Show Logs"}
          </button>
          <button
            onClick={downloadLogsAsZip}
            disabled={Object.keys(logsByHost).length === 0}
            style={{
              padding: "6px 12px",
              background: "#3498db",
              color: "white",
            }}
          >
            Download Logs
          </button>

        </div>
      </div>

      {running && (
        <div style={{ marginTop: 16 }}>
          <h4>Live Job Status</h4>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "14px",
            }}
          >
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>
                  Host
                </th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>
                  Status
                </th>
                <th style={{ textAlign: "left", borderBottom: "1px solid #ccc" }}>
                  PID
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(jobStatus).map(([host, { state, pid }]) => (
                <tr key={host}>
                  <td>{host}</td>
                  <td style={{ color: state === "running" ? "green" : state === "terminated" ? "red" : "gray" }}>
                    {state}
                  </td>
                  <td>{pid ?? "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}


      {panelOpen && (
        <div
          style={{
            marginTop: 12,
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(600px, 1fr))",
            gap: 16,
            background: "#1e1e1e",
            color: "#f1f1f1",
            fontFamily: "monospace",
            fontSize: 14,
            borderRadius: 4,
            padding: 12,
            minHeight: 200,
            maxHeight: "80vh",
            overflow: "auto",
          }}
        >
          {FIXED_HOSTS.map((host) => (
            <div
              key={host}
              style={{
                flex: 1,
                minWidth: 0,
                border: "1px solid #444",
                borderRadius: 4,
                padding: 8,
                background: "#222",
                display: "flex",
                flexDirection: "column",
                maxHeight: 350,
              }}
            >
              <div style={{ fontWeight: "bold", color: "#ffd700", marginBottom: 4, textAlign: "center" }}>{host}</div>
              <div
                id={`log-scroll-${host}`}
                style={{
                  flex: 1,
                  overflowY: "auto",
                  minHeight: 100,
                  maxHeight: 300,
                  whiteSpace: "pre-wrap",
                }}
              >
                {(logsByHost[host] || []).length === 0 ? (
                  <div style={{ color: "#888" }}>(No logs yet)</div>
                ) : (
                  logsByHost[host].map((line, i) => (
                    <div key={i}>{line}</div>
                  ))
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
