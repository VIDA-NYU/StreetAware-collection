import React, { useState, useRef, useEffect } from "react";

const MAX_LOG_LINES = 500;

export default function SSHControl() {
  const [timeoutSec, setTimeoutSec] = useState(60);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const esRef = useRef(null);
  const logContainerRef = useRef(null);

  // Start the SSH job and open SSE stream
  const startJob = () => {
    if (running) return;

    setLogs([]);
    setRunning(true);
    setPanelOpen(true);

    const es = new EventSource(
      `http://localhost:8080/start-ssh/logs?timeout=${timeoutSec}`
    );

    es.onmessage = (e) => {
      setLogs((old) => {
        const next = [...old, e.data];
        return next.length > MAX_LOG_LINES
          ? next.slice(next.length - MAX_LOG_LINES)
          : next;
      });
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
  }, [logs, panelOpen]);

  // Cleanup if component unmounts
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, []);

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        maxWidth: 600,
        margin: "1rem auto",
      }}
    >
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
          disabled={!logs.length}
          style={{ marginLeft: "auto", padding: "6px 12px" }}
        >
          {panelOpen ? "Hide Logs" : "Show Logs"}
        </button>
      </div>

      {panelOpen && (
        <div
          ref={logContainerRef}
          style={{
            marginTop: 12,
            padding: 12,
            background: "#1e1e1e",
            color: "#f1f1f1",
            fontFamily: "monospace",
            fontSize: 14,
            borderRadius: 4,
            maxHeight: 300,
            overflowY: "auto",
            whiteSpace: "pre-wrap",
          }}
        >
          {logs.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}
