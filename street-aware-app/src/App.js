// src/App.js
import React from 'react';
import SSHControl from './components/SSHControl';
import HealthMonitor from "./components/HealthMonitor";
import DownloadWithProgressBar from "./components/DownloadWithProgressBar";

function App() {
  return (
    <div className="App">
      <h1>Device SSH Collector</h1>
      <HealthMonitor />
      <SSHControl />
      <DownloadWithProgressBar />
    </div>
  );
}

export default App;
