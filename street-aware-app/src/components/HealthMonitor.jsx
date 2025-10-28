import React, { useState, useEffect } from "react";

// Poll interval in milliseconds
const POLL_INTERVAL_MS = 10000;

export default function HealthMonitor() {
  const [statuses, setStatuses] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState(null);

  // Fetch current mode and sensor statuses
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
        if (data.__mode) {
          setMode(data.__mode);
          delete data.__mode;
        }
        setStatuses(data);
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

  // Get status count for summary
  const displayStatuses = Object.fromEntries(
    Object.entries(statuses).filter(([key, value]) => key !== "__mode" && value && typeof value === "object")
  );

  const statusCounts = Object.values(displayStatuses).reduce((acc, info) => {
    acc[info.status] = (acc[info.status] || 0) + 1;
    return acc;
  }, {});

  const totalHosts = Object.keys(displayStatuses).length;
  const upHosts = statusCounts.up || 0;

  const downHosts = statusCounts.down || 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">Device Health Monitor</h2>
          <p className="text-sm text-gray-600">Real-time status of all connected devices</p>
        </div>
        <div className="flex items-center gap-2">
          {loading && (
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
          )}
          <span className="text-xs text-gray-500">Auto-refresh: 10s</span>
        </div>
      </div>

      {mode && (
        <div className="flex items-center space-x-2 mb-6">
          <span className="text-gray-700">Mode:</span>
          <span className="font-semibold">{mode.toUpperCase()}</span>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-gray-900">{totalHosts}</div>
          <div className="text-sm text-gray-600">Total Devices</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-600">{upHosts}</div>
          <div className="text-sm text-green-700">Online</div>
        </div>
        <div className="bg-red-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-red-600">{downHosts}</div>
          <div className="text-sm text-red-700">Offline</div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <div className="status-dot status-dot-down"></div>
            <span className="text-red-800 font-medium">Connection Error</span>
          </div>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Device List */}
      {totalHosts > 0 ? (
        <div className="space-y-3">
          {Object.entries(displayStatuses).map(([name, info]) => (
            <div
              key={name}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className={`status-dot status-dot-${info.status}`}></div>
                <div>
                  <div className="font-medium text-gray-900">{name}</div>
                  <div className="text-sm text-gray-600">{info.mode === 'mock' ? `localhost:${info.port}` : info.host}</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`status-indicator status-${info.status} capitalize`}>
                  {info.status === "up" ? "Online" : info.status === "down" ? "Offline" : "Unknown"}
                </span>
                {info.status === "up" && (
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Devices Connected</h3>
          <p className="text-gray-600">Devices will appear here once they're detected and connected.</p>
        </div>
      )}
    </div>
  );
}
