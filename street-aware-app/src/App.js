// src/App.js
import React from 'react';
import SSHControl from './components/SSHControl';
import HealthMonitor from "./components/HealthMonitor";
import DownloadWithProgressBar from "./components/DownloadWithProgressBar";
import ManualDownload from "./components/ManualDownload";
import SessionManager from "./components/SessionManager";
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="container">
          <div className="py-6">
            <h1 className="text-3xl font-bold text-gray-900 text-center">
              StreetAware Data Collection System
            </h1>
            <p className="text-center text-gray-600 mt-2">
              Monitor device health, collect data via SSH, and manage downloads
            </p>
          </div>
        </div>
      </header>

      <main className="container py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Health Monitor Section */}
          <div className="lg:col-span-2">
            <HealthMonitor />
          </div>

          {/* SSH Control Section */}
          <div className="lg:col-span-2">
            <SSHControl />
          </div>
          
          {/* Download Section */}
          <div className="lg:col-span-2">
            <DownloadWithProgressBar />
          </div>

          {/* Manual Download Section */}
          <div className="lg:col-span-2">
            <ManualDownload />
          </div>

          {/* Session Manager Section */}
          <div className="lg:col-span-2">
            <SessionManager />
          </div>
        </div>
      </main>

      <footer className="bg-gray-50 border-t border-gray-200 mt-16">
        <div className="container py-6">
          <div className="text-center text-gray-600">
            <p>&copy; 2024 StreetAware Collection System. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
