import React, { useState, useRef, useEffect } from "react";
import { fetchSensors } from "../utils/sensorConfig";

export default function DownloadWithProgressBar() {
  const [hosts, setHosts] = useState({});
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState(null);
  const esRef = useRef(null);
  const knownHostsRef = useRef([]);

  // Initialize hosts state from config
  useEffect(() => {
    async function initializeHosts() {
      try {
        const sensorHosts = await fetchSensors();
        const initialHosts = sensorHosts.reduce((acc, sensor) => {
          const name = sensor.display_name;
          acc[name] = { done: 0, total: 1, status: "pending", path: null };
          return acc;
        }, {});
        setHosts(initialHosts);
        knownHostsRef.current = Object.keys(initialHosts);
      } catch (error) {
        console.error("Failed to initialize hosts:", error);
      }
    }
    initializeHosts();
  }, []);

  const startDownload = () => {
    if (streaming) return;
    
    // Reset all hosts to "pending"
    setHosts(prevHosts => {
      const resetHosts = {};
      Object.keys(prevHosts).forEach(host => {
        resetHosts[host] = { done: 0, total: 1, status: "pending", path: null };
      });
      knownHostsRef.current = Object.keys(resetHosts);
      return resetHosts;
    });
    setStreaming(true);
    setStreamError(null);

    const resolveHost = (incoming) => {
      const list = knownHostsRef.current || [];
      if (list.includes(incoming)) return incoming;
      const ci = list.find(h => h.toLowerCase() === incoming.toLowerCase());
      if (ci) return ci;
      // Some backends might emit abbreviated names; prefer not to create a new card
      return null;
    };

    const es = new EventSource("http://localhost:8080/download-data");
    es.onmessage = (e) => {
      const line = e.data.trim();
      if (!line) return;
      // Ignore server-sent comment pings like ": ping - ..."
      if (line.startsWith(":")) return;

      // PROGRESS <host with spaces allowed> <done_bytes> <total_bytes>
      const progressMatch = line.match(/^PROGRESS\s+(.+)\s+(\d+)\s+(\d+)$/);
      if (progressMatch) {
        const parsedHost = progressMatch[1];
        const host = resolveHost(parsedHost);
        const done = parseInt(progressMatch[2], 10);
        const total = parseInt(progressMatch[3], 10);
        if (!host) {
          // Unknown host in stream; skip to avoid creating stray cards
          console.warn("Progress for unknown host:", parsedHost);
          return;
        }
        setHosts((prev) => {
          return {
            ...prev,
            [host]: { ...prev[host], done, total, status: "downloading" },
          };
        });
        return;
      }

      // COMPLETE <host> <rest...>
      if (line.startsWith("COMPLETE ")) {
        const after = line.slice("COMPLETE ".length);
        const candidates = (knownHostsRef.current || []).sort((a,b) => b.length - a.length);
        const matchHost = candidates.find(h => after.startsWith(h));
        if (!matchHost) {
          console.warn("COMPLETE for unknown host:", after);
          return;
        }
        const remainder = after.slice(matchHost.length).trim();
        if (remainder.startsWith("ERROR")) {
          const errMsg = remainder.slice("ERROR".length).trim() || null;
          setHosts(prev => ({
            ...prev,
            [matchHost]: { ...(prev[matchHost] || { done: 0, total: 0, path: null }), status: "error", errorMessage: errMsg },
          }));
          return;
        }
        // Success: remainder can be "0" or a path possibly with spaces
        const path = remainder === "0" ? null : remainder;
        setHosts(prev => ({
          ...prev,
          [matchHost]: { ...(prev[matchHost] || {}), status: "downloaded", path },
        }));
        return;
      }

      if (line.startsWith("SUMMARY ")) {
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

  // Helper to format bytes to human-readable format
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Helper to render each host's progress bar + status
  const renderHostRow = (host, data) => {
    const { done, total, status, path } = data;
    const percent =
      total && total > 0 && status === "downloading"
        ? Math.floor((done / total) * 100)
        : 0;

    let statusDisplay;
    let statusColor;
    switch (status) {
      case "pending":
        statusDisplay = "Pending";
        statusColor = "text-gray-500";
        break;
      case "downloading":
        statusDisplay = `Downloading… ${percent}%`;
        statusColor = "text-blue-600";
        break;
      case "downloaded":
        statusDisplay = "Completed";
        statusColor = "text-green-600";
        break;
      case "error":
        statusDisplay = "Error";
        statusColor = "text-red-600";
        break;
      default:
        statusDisplay = status;
        statusColor = "text-gray-500";
    }

    return (
      <div
        key={host}
        className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className={`status-dot status-dot-${status === "downloaded" ? "up" : status === "error" ? "down" : "pending"}`}></div>
            <div>
              <div className="font-semibold text-gray-900">{host}</div>
              <div className="text-sm text-gray-600">Data Collection Node</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium ${statusColor}`}>
              {statusDisplay}
            </span>
            {status === "downloading" && (
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            )}
          </div>
        </div>

        {status === "downloading" && (
          <div className="mb-3">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Progress</span>
              <span>{formatBytes(done)} / {formatBytes(total)}</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${percent}%` }}
              />
            </div>
          </div>
        )}

        {status === "downloaded" && path && (
          <div className="bg-green-50 rounded-lg p-3 mt-3">
            <div className="flex items-center gap-2 text-sm text-green-700">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium">Download Complete</span>
            </div>
            <div className="text-xs text-green-600 mt-1 font-mono">
              {path}
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="bg-red-50 rounded-lg p-3 mt-3">
            <div className="flex items-center gap-2 text-sm text-red-700">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium">Download Failed</span>
            </div>
            {data?.errorMessage && (
              <div className="text-xs text-red-700 mt-2 font-mono break-all">
                {data.errorMessage}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">Data Download</h2>
          <p className="text-sm text-gray-600">Download collected data from all connected devices</p>
        </div>
        {streaming && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-blue-600 font-medium">Downloading</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-3 mb-6">
        <button
          onClick={startDownload}
          disabled={streaming}
          className="btn btn-primary"
        >
          {streaming ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              Downloading…
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Data (Per-IP)
            </>
          )}
        </button>

        {streaming && (
          <button
            onClick={cancelDownload}
            className="btn btn-danger"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Cancel All
          </button>
        )}
      </div>

      {streamError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <div className="status-dot status-dot-down"></div>
            <span className="text-red-800 font-medium">Download Error</span>
          </div>
          <p className="text-red-700 text-sm mt-1">{streamError}</p>
        </div>
      )}

      <div className="space-y-4">
        {Object.entries(hosts).map(([host, data]) =>
          renderHostRow(host, data)
        )}
      </div>
    </div>
  );
}
