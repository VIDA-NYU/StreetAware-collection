import React, { useEffect, useRef, useState } from "react";
import { fetchSensors } from "../utils/sensorConfig";

export default function ManualDownload() {
  const [foldersByHost, setFoldersByHost] = useState({});
  const [selection, setSelection] = useState({});
  const [hosts, setHosts] = useState({});
  const [loadingFolders, setLoadingFolders] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const esRef = useRef(null);
  const knownHostsRef = useRef([]);

  useEffect(() => {
    async function init() {
      try {
        const sensorHosts = await fetchSensors();
        const initialHosts = sensorHosts.reduce((acc, sensor) => {
          const name = sensor.display_name;
          acc[name] = { done: 0, total: 1, status: "pending", path: null };
          return acc;
        }, {});
        setHosts(initialHosts);
        knownHostsRef.current = Object.keys(initialHosts);
      } catch (e) {
        console.error("Failed to initialize manual hosts", e);
      }
    }
    init();
  }, []);

  const fetchRemoteFolders = async () => {
    setLoadingFolders(true);
    try {
      const resp = await fetch("http://localhost:8080/remote-folders");
      const data = await resp.json();
      setFoldersByHost(data);
    } catch (e) {
      console.error("Failed to fetch remote folders", e);
    } finally {
      setLoadingFolders(false);
    }
  };

  const setChoice = (host, folder) => {
    setSelection((prev) => ({ ...prev, [host]: folder || undefined }));
  };

  const UTF8toB64 = (obj) => {
    const json = JSON.stringify(obj || {});
    // handle unicode safely
    return btoa(unescape(encodeURIComponent(json)));
  };

  const resolveHost = (incoming) => {
    const list = knownHostsRef.current || [];
    if (list.includes(incoming)) return incoming;
    const ci = list.find((h) => h.toLowerCase() === incoming.toLowerCase());
    return ci || null;
  };

  const formatBytes = (bytes) => {
    if (!bytes || bytes <= 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const startManualDownload = () => {
    if (streaming) return;

    // Filter only chosen entries
    const chosen = Object.fromEntries(
      Object.entries(selection).filter(([, folder]) => folder)
    );
    if (Object.keys(chosen).length === 0) {
      alert("Select at least one folder.");
      return;
    }

    // Reset UI state
    setHosts((prev) => {
      const reset = {};
      Object.keys(prev).forEach((h) => {
        reset[h] = { done: 0, total: 1, status: "pending", path: null };
      });
      return reset;
    });

    const sel = UTF8toB64(chosen);
    const es = new EventSource(
      `http://localhost:8080/download-data/manual?sel=${encodeURIComponent(sel)}`
    );

    es.onmessage = (e) => {
      const line = (e.data || "").trim();
      if (!line) return;
      if (line.startsWith(":")) return; // ignore comments

      // PROGRESS <host> <done> <total>
      const progressMatch = line.match(/^PROGRESS\s+(.+)\s+(\d+)\s+(\d+)$/);
      if (progressMatch) {
        const parsedHost = progressMatch[1];
        const host = resolveHost(parsedHost);
        const done = parseInt(progressMatch[2], 10);
        const total = parseInt(progressMatch[3], 10);
        if (!host) return;
        setHosts((prev) => ({
          ...prev,
          [host]: { ...(prev[host] || {}), done, total, status: "downloading" },
        }));
        return;
      }

      // COMPLETE generic parsing (host may have spaces, remainder may be path with spaces)
      if (line.startsWith("COMPLETE ")) {
        const after = line.slice("COMPLETE ".length);
        const candidates = (knownHostsRef.current || []).sort((a, b) => b.length - a.length);
        const matchHost = candidates.find((h) => after.startsWith(h));
        if (!matchHost) return;
        const remainder = after.slice(matchHost.length).trim();
        if (remainder.startsWith("ERROR")) {
          const errMsg = remainder.slice("ERROR".length).trim() || null;
          setHosts((prev) => ({
            ...prev,
            [matchHost]: {
              ...(prev[matchHost] || { done: 0, total: 0, path: null }),
              status: "error",
              errorMessage: errMsg,
            },
          }));
          return;
        }
        const path = remainder === "0" ? null : remainder;
        setHosts((prev) => ({
          ...prev,
          [matchHost]: { ...(prev[matchHost] || {}), status: "downloaded", path },
        }));
        return;
      }

      if (line.startsWith("SUMMARY ")) {
        // Final state sync
        try {
          const summary = JSON.parse(line.slice("SUMMARY ".length));
          setHosts((prev) => ({ ...prev, ...summary }));
        } catch {}
      }
    };

    es.onerror = () => {
      try { es.close(); } catch {}
      esRef.current = null;
      setStreaming(false);
    };

    esRef.current = es;
    setStreaming(true);
  };

  const cancel = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setStreaming(false);
  };

  const renderRow = (host, data) => {
    const { done, total, status, path, errorMessage } = data || {};
    const percent = total && total > 0 && status === "downloading" ? Math.floor((done / total) * 100) : 0;
    return (
      <div key={host} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <div className="font-semibold">{host}</div>
          <div className="text-sm text-gray-600">
            {status === "downloading" ? `Downloading… ${percent}%` : status === "downloaded" ? "Completed" : status === "error" ? "Error" : "Pending"}
          </div>
        </div>
        {status === "downloading" && (
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Progress</span>
              <span>{formatBytes(done)} / {formatBytes(total)}</span>
            </div>
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${percent}%` }} /></div>
          </div>
        )}
        {status === "downloaded" && path && (
          <div className="text-xs text-green-700 bg-green-50 rounded p-2 mt-2 font-mono break-all">{path}</div>
        )}
        {status === "error" && (
          <div className="text-xs text-red-700 bg-red-50 rounded p-2 mt-2 font-mono break-all">{errorMessage || "Download failed"}</div>
        )}
      </div>
    );
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">Manual Download</h2>
          <p className="text-sm text-gray-600">Choose folders per sensor and stream the download</p>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-secondary" onClick={fetchRemoteFolders} disabled={loadingFolders}>
            {loadingFolders ? "Loading…" : "Refresh Folders"}
          </button>
          {streaming ? (
            <button className="btn btn-danger" onClick={cancel}>Cancel</button>
          ) : (
            <button className="btn btn-primary" onClick={startManualDownload} disabled={Object.keys(foldersByHost).length === 0}>Start Manual Download</button>
          )}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-3 mb-6">
        {Object.keys(hosts).map((host) => {
          const info = foldersByHost[host];
          const folders = info && info.status === "ok" ? info.folders : [];
          const err = info && info.status === "error" ? info.error : null;
          return (
            <div key={host} className="bg-white border border-gray-200 rounded p-3">
              <div className="text-sm font-medium mb-2">{host}</div>
              {err ? (
                <div className="text-xs text-red-600">{err}</div>
              ) : (
                <select
                  className="w-full border rounded p-2 text-sm"
                  value={selection[host] || ""}
                  onChange={(e) => setChoice(host, e.target.value)}
                >
                  <option value="">-- Select folder --</option>
                  {folders.map((f) => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                </select>
              )}
            </div>
          );
        })}
      </div>

      <div className="space-y-3">
        {Object.entries(hosts).map(([h, d]) => renderRow(h, d))}
      </div>
    </div>
  );
}
