import React, { useState, useEffect } from "react";

export default function SessionManager() {
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(null);
  const [selectedSession, setSelectedSession] = useState("");
  const [sessionLogs, setSessionLogs] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSessions();
    fetchCurrentSession();
  }, []);

  const fetchSessions = async () => {
    try {
      const resp = await fetch("http://localhost:8080/sessions");
      const data = await resp.json();
      setSessions(data);
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  const fetchCurrentSession = async () => {
    try {
      const resp = await fetch("http://localhost:8080/current-session");
      const data = await resp.json();
      setCurrentSession(data);
    } catch (e) {
      console.error("Failed to fetch current session", e);
    }
  };

  const fetchSessionLogs = async (sessionId, host = null) => {
    setLoading(true);
    try {
      const url = host 
        ? `http://localhost:8080/sessions/${sessionId}/logs?host=${encodeURIComponent(host)}`
        : `http://localhost:8080/sessions/${sessionId}/logs`;
      const resp = await fetch(url);
      const logs = await resp.json();
      setSessionLogs(prev => ({ ...prev, [sessionId]: logs }));
    } catch (e) {
      console.error("Failed to fetch session logs", e);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (state) => {
    switch (state) {
      case "running": return "text-green-600";
      case "terminated": return "text-blue-600";
      case "failed": return "text-red-600";
      case "interrupted": return "text-yellow-600";
      case "connecting": case "starting": return "text-orange-600";
      default: return "text-gray-600";
    }
  };

  const getStatusDot = (state) => {
    switch (state) {
      case "running": return "status-dot-up";
      case "terminated": return "status-dot-down";
      case "failed": return "status-dot-down";
      case "interrupted": return "status-dot-pending";
      case "connecting": case "starting": return "status-dot-pending";
      default: return "status-dot-pending";
    }
  };

  const formatTimestamp = (isoString) => {
    if (!isoString) return "N/A";
    return new Date(isoString).toLocaleString();
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">Session Manager</h2>
          <p className="text-sm text-gray-600">Track and recover SSH collection sessions with PIDs and logs</p>
        </div>
        <button 
          className="btn btn-secondary" 
          onClick={() => { fetchSessions(); fetchCurrentSession(); }}
        >
          Refresh
        </button>
      </div>

      {/* Current Session */}
      {currentSession && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span className="font-medium text-blue-900">Current Session</span>
            </div>
            <span className="text-sm text-blue-700 font-mono">{currentSession.session_id}</span>
          </div>
          
          <div className="space-y-2">
            {Object.entries(currentSession.hosts || {}).map(([host, data]) => (
              <div key={host} className="flex items-center justify-between bg-white rounded p-2">
                <div className="flex items-center gap-2">
                  <div className={`status-dot ${getStatusDot(data.state)}`}></div>
                  <span className="font-medium">{host}</span>
                  {data.pid && <span className="text-xs bg-gray-100 px-2 py-1 rounded">PID: {data.pid}</span>}
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-sm ${getStatusColor(data.state)}`}>{data.state}</span>
                  <button 
                    className="text-xs text-blue-600 hover:text-blue-800"
                    onClick={() => fetchSessionLogs(currentSession.session_id, host)}
                  >
                    View Logs
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All Sessions */}
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-3">All Sessions</h3>
        {Object.keys(sessions).length === 0 ? (
          <p className="text-gray-500 text-sm">No sessions found</p>
        ) : (
          <div className="space-y-3">
            {Object.entries(sessions).map(([sessionId, sessionData]) => (
              <div key={sessionId} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-sm text-gray-700">{sessionId}</span>
                    <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                      {sessionData.host_count} hosts, {sessionData.active_count} active
                    </span>
                  </div>
                  <button 
                    className="btn btn-sm btn-secondary"
                    onClick={() => fetchSessionLogs(sessionId)}
                  >
                    Load Logs
                  </button>
                </div>
                
                <div className="grid md:grid-cols-2 gap-2">
                  {Object.entries(sessionData.hosts || {}).map(([host, data]) => (
                    <div key={host} className="flex items-center justify-between bg-gray-50 rounded p-2">
                      <div className="flex items-center gap-2">
                        <div className={`status-dot ${getStatusDot(data.state)}`}></div>
                        <span className="text-sm font-medium">{host}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {data.pid && <span className="text-xs bg-white px-2 py-1 rounded">PID: {data.pid}</span>}
                        <span className={`text-xs ${getStatusColor(data.state)}`}>{data.state}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Session Logs */}
      {Object.keys(sessionLogs).length > 0 && (
        <div>
          <h3 className="text-lg font-medium mb-3">Session Logs</h3>
          {Object.entries(sessionLogs).map(([sessionId, logs]) => (
            <div key={sessionId} className="mb-4">
              <div className="text-sm font-medium text-gray-700 mb-2">Session: {sessionId}</div>
              {Object.entries(logs).map(([host, hostLogs]) => (
                <div key={host} className="mb-3">
                  <div className="text-xs font-medium text-gray-600 mb-1">{host}</div>
                  <div className="bg-gray-900 text-green-400 p-3 rounded text-xs font-mono max-h-40 overflow-y-auto">
                    {Array.isArray(hostLogs) ? (
                      hostLogs.slice(-50).map((line, idx) => (
                        <div key={idx}>{line}</div>
                      ))
                    ) : (
                      <div>No logs available</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="inline-flex items-center gap-2 text-gray-600">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-300 border-t-blue-600"></div>
            Loading logs...
          </div>
        </div>
      )}
    </div>
  );
}
