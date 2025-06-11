import React, { useState, useRef, useEffect } from "react";

export default function DownloadWithProgressBar() {
  const [hosts, setHosts] = useState({
    "192.168.0.184": { done: 0, total: 1, status: "pending", path: null },
    "192.168.0.122": { done: 0, total: 1, status: "pending", path: null },
    "192.168.0.108": { done: 0, total: 1, status: "pending", path: null },
    "192.168.0.227": { done: 0, total: 1, status: "pending", path: null },
  });

  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState(null);
  const esRef = useRef(null);

  const startDownload = () => {
    if (streaming) return;

    // Reset all hosts to “pending”
    setHosts({
      "192.168.0.184": { done: 0, total: 1, status: "pending", path: null },
      "192.168.0.122": { done: 0, total: 1, status: "pending", path: null },
      "192.168.0.108": { done: 0, total: 1, status: "pending", path: null },
      "192.168.0.227": { done: 0, total: 1, status: "pending", path: null },
    });
    setStreaming(true);
    setStreamError(null);

    const es = new EventSource("http://localhost:8080/download-data");
    es.onmessage = (e) => {
      const line = e.data.trim();
      // console.log("Received SSE:", line);
      if (line.startsWith("PROGRESS ")) {
        // Format: PROGRESS <host> <done> <total>
        const parts = line.split(" ");
        if (parts.length === 4) {
          const host = parts[1];
          const done = parseInt(parts[2], 10);
          const total = parseInt(parts[3], 10);
          setHosts((prev) => ({
            ...prev,
            [host]: {
              ...prev[host],
              done,
              total,
              status: "downloading",
            },
          }));
        }
      } else if (line.startsWith("COMPLETE ")) {
        // Format: COMPLETE <host> <local_path>  OR  COMPLETE <host> ERROR
        const parts = line.split(" ");
        const host = parts[1];
        const rest = parts.slice(2).join(" ");
        if (rest === "ERROR") {
          setHosts((prev) => ({
            ...prev,
            [host]: {
              ...prev[host],
              status: "error",
            },
          }));
        } else {
          setHosts((prev) => ({
            ...prev,
            [host]: {
              ...prev[host],
              status: "downloaded",
              path: rest,
            },
          }));
        }
      } else if (line.startsWith("SUMMARY ")) {
        // Format: SUMMARY <json>
        const rawJson = line.slice("SUMMARY ".length);
        try {
          const summary = JSON.parse(rawJson);
          setHosts(summary);
        } catch {
          // ignore parse errors
        }
        // Close the stream and mark not streaming
        es.close();
        esRef.current = null;
        setStreaming(false);
      }
      // Any other lines you can ignore or log if you like
    };

    es.onerror = (err) => {
      console.error("SSE error:", err);
      es.close();
      esRef.current = null;
      setStreaming(false);
      setStreamError("Connection lost or server error");
    };

    esRef.current = es;
  };

  const cancelDownload = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setStreaming(false);
  };

  // Make sure to clean up if component unmounts
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, []);

  // Helper to render each host’s progress bar + status
  const renderHostRow = (host, data) => {
    const { done, total, status, path } = data;
    const percent =
      total && status === "downloading"
        ? Math.floor((done / total) * 100)
        : 0;

    let statusDisplay;
    switch (status) {
      case "pending":
        statusDisplay = "Pending";
        break;
      case "downloading":
        statusDisplay = `Downloading… ${percent}%`;
        break;
      case "downloaded":
        statusDisplay = <span style={{ color: "green" }}>Done</span>;
        break;
      case "error":
        statusDisplay = <span style={{ color: "red" }}>Error</span>;
        break;
      default:
        statusDisplay = status;
    }

    return (
      <div
        key={host}
        style={{
          marginBottom: "1rem",
          border: "1px solid #ccc",
          padding: "8px",
          borderRadius: 4,
        }}
      >
        <div style={{ fontWeight: "bold" }}>{host}</div>
        <div
          style={{
            height: "12px",
            background: "#eee",
            borderRadius: 6,
            overflow: "hidden",
          }}
        >
          {status === "downloading" && (
            <div
              style={{
                width: `${percent}%`,
                height: "100%",
                background: "#4caf50",
                transition: "width 0.2s",
              }}
            />
          )}
        </div>
        <div style={{ marginTop: 4 }}>{statusDisplay}</div>
        {status === "downloaded" && path && (
          <div style={{ fontSize: "0.85em", color: "#555", marginTop: 2 }}>
            Saved at <code>{path}</code>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        maxWidth: 800,
        margin: "1rem auto",
      }}
    >
      <button
        onClick={startDownload}
        disabled={streaming}
        style={{
          padding: "8px 16px",
          background: "#007bff",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: streaming ? "not-allowed" : "pointer",
        }}
      >
        {streaming ? "Downloading…" : "Download Data (Per-IP)"}
      </button>

      {streaming && (
        <button
          onClick={cancelDownload}
          style={{
            marginLeft: 8,
            padding: "8px 16px",
            background: "#dc3545",
            color: "white",
            border: "none",
            borderRadius: 4,
          }}
        >
          Cancel All
        </button>
      )}

      {streamError && (
        <p style={{ color: "red", marginTop: "0.5rem" }}>{streamError}</p>
      )}

      <div style={{ marginTop: "1rem" }}>
        {Object.entries(hosts).map(([host, data]) =>
          renderHostRow(host, data)
        )}
      </div>
    </div>
  );
}
