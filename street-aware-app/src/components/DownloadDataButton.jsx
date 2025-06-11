import React, { useState } from "react";

export default function DownloadDataButton() {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const handleDownload = async () => {
    if (loading) return;
    setLoading(true);
    setReport(null);
    setError(null);

    try {
      const resp = await fetch("http://localhost:8080/download-data", {
        method: "POST",
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = await resp.json();
      setReport(data);
    } catch (e) {
      console.error("Download Data Error:", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "Arial, sans-serif", margin: "1rem 0" }}>
      <button
        onClick={handleDownload}
        disabled={loading}
        style={{
          padding: "8px 16px",
          background: "#007bff",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Downloading…" : "Download Data"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: "0.5rem" }}>
          Error: {error}
        </p>
      )}

      {report && (
        <div style={{ marginTop: "1rem" }}>
          <h4>Download Summary:</h4>
          <ul style={{ listStyle: "none", paddingLeft: 0 }}>
            {Object.entries(report).map(([host, info]) => (
              <li key={host} style={{ marginBottom: "0.5rem" }}>
                <strong>{host}</strong> –{" "}
                {info.status === "downloaded" ? (
                  <span style={{ color: "green" }}>
                    Downloaded to <code>{info.path}</code>
                  </span>
                ) : (
                  <span style={{ color: "red" }}>
                    Error: {info.error}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
