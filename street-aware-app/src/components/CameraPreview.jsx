import React, { useState, useEffect, useCallback } from "react";

export default function CameraPreview() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // Check if recording is in progress
  const checkRecordingStatus = useCallback(async () => {
    try {
      const res = await fetch("/start-ssh/running");
      if (res.ok) {
        const data = await res.json();
        setIsRecording(data.running);
      }
    } catch (err) {
      console.error("Failed to check recording status:", err);
    }
  }, []);

  // Fetch camera previews
  const fetchPreviews = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch("/camera/preview-all");
      if (!res.ok) {
        if (res.status === 409) {
          setError("Cannot preview while recording is in progress");
          setIsRecording(true);
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      
      const data = await res.json();
      setCameras(data.cameras || []);
      setLastRefresh(new Date());
    } catch (err) {
      console.error("Failed to fetch camera previews:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Check recording status on mount
  useEffect(() => {
    checkRecordingStatus();
    const interval = setInterval(checkRecordingStatus, 5000);
    return () => clearInterval(interval);
  }, [checkRecordingStatus]);

  // Group cameras by sensor
  const camerasBySensor = cameras.reduce((acc, cam) => {
    const sensorName = cam.sensor || "Unknown";
    if (!acc[sensorName]) {
      acc[sensorName] = [];
    }
    acc[sensorName].push(cam);
    return acc;
  }, {});

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-1">Camera Preview</h2>
          <p className="text-sm text-gray-600">
            View live snapshots from all sensor cameras
          </p>
        </div>
        {lastRefresh && (
          <span className="text-xs text-gray-500">
            Last refresh: {lastRefresh.toLocaleTimeString()}
          </span>
        )}
      </div>

      {isRecording && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2 text-yellow-800">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-sm font-medium">
              Camera preview is disabled while recording is in progress
            </span>
          </div>
        </div>
      )}

      <div className="flex gap-3 mb-6">
        <button
          onClick={fetchPreviews}
          disabled={loading || isRecording}
          className="btn btn-primary"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
              Loading...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh All Cameras
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2 text-red-800">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}

      {cameras.length === 0 && !loading && !error && (
        <div className="text-center py-12 text-gray-500">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p className="text-lg font-medium">No camera previews loaded</p>
          <p className="text-sm mt-1">Click "Refresh All Cameras" to capture snapshots</p>
        </div>
      )}

      {/* Camera Grid */}
      <div className="space-y-6">
        {Object.entries(camerasBySensor).map(([sensorName, sensorCameras]) => (
          <div key={sensorName} className="border border-gray-200 rounded-lg overflow-hidden">
            <div className="bg-gray-100 px-4 py-2 border-b border-gray-200">
              <h3 className="font-semibold text-gray-800">{sensorName}</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
              {sensorCameras
                .sort((a, b) => a.camera_id - b.camera_id)
                .map((cam) => (
                  <div key={`${cam.sensor_name}-${cam.camera_id}`} className="relative">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                      Camera {cam.camera_id}
                    </div>
                    {cam.success ? (
                      <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
                        <img
                          src={`data:image/jpeg;base64,${cam.image}`}
                          alt={`${cam.sensor} Camera ${cam.camera_id}`}
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute bottom-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded">
                          OK
                        </div>
                      </div>
                    ) : (
                      <div className="aspect-video bg-gray-800 rounded-lg flex items-center justify-center">
                        <div className="text-center p-4">
                          <svg className="w-12 h-12 mx-auto mb-2 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                          </svg>
                          <p className="text-red-400 text-sm font-medium">Camera Error</p>
                          <p className="text-gray-400 text-xs mt-1 max-w-xs">
                            {cam.error || "Unknown error"}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
