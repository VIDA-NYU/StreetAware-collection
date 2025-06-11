import React, { useState, useEffect } from "react";

// Poll interval in milliseconds
const POLL_INTERVAL_MS = 10000;

export default function HealthMonitor() {
  const [statuses, setStatuses] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch /health once and then on interval
  useEffect(() => {
    let intervalId;

    const fetchHealth = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("http://localhost:8080/health");
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setStatuses(data); // e.g. { "192.168.0.184": "up", … }
      } catch (e) {
        console.error("Health fetch error:", e);
        setError(e.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchHealth();

    // Poll every POLL_INTERVAL_MS
    intervalId = setInterval(fetchHealth, POLL_INTERVAL_MS);

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, []);

  // Render a colored circle: red=down, green=up, gray=unknown
  const renderLed = (status) => {
    const color = status === "up" ? "#4CAF50"  // green
                : status === "down" ? "#F44336" // red
                : "#9E9E9E";                   // gray if missing
    return (
      <span
        style={{
          display: "inline-block",
          marginRight: 8,
          width: 12,
          height: 12,
          borderRadius: "50%",
          backgroundColor: color,
        }}
      />
    );
  };

  return (
    <div style={{ fontFamily: "Arial, sans-serif", maxWidth: 600, margin: "1rem auto" }}>
      <h2>SSH Node Health</h2>

      {/* {loading && <p>Loading…</p>} */}
      {/* {error && <p style={{ color: "red" }}>Error: {error}</p>} */}

      {/* Display one row per host */}
      <ul style={{ listStyle: "none", paddingLeft: 0 }}>
        {Object.entries(statuses).map(([host, status]) => (
          <li key={host} style={{ marginBottom: "0.5rem" }}>
            {renderLed(status)} {host} — <span style={{ textTransform: "capitalize" }}>{status}</span>
          </li>
        ))}
      </ul>

      {/* If no data yet and not loading, show a placeholder */}
      {!loading && !error && Object.keys(statuses).length === 0 && (
        <p>No hosts available.</p>
      )}
    </div>
  );
}
