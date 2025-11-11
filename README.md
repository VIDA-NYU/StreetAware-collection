# StreetAware Project - Data Collection & Synchronization Toolkit

This guide covers the complete workflow for the StreetAware project, from data collection to advanced video synchronization and repair. All steps are streamlined for clarity and professional use.

## 1. App Setup - Using Bash Script

### Clone the repository:

```
git clone https://github.com/VIDA-NYU/StreetAware-collection.git
```

### Make scripts executable:

```
chmod +x setup.sh run.sh stop.sh
```

### Run setup:

```
./setup.sh
```

---

## 2. App Run - Using Bash Script

### Start the app:

```
./run.sh
```

### Stop the app:

```
./stop.sh
```
---

## 3. How to Use the App

### - Health Check

The app includes a UI section for **health checks**. Below is a sample image indicating what the health check screen looks like. This helps verify if sensors are connected and responsive.

![Health Check UI](assets/health_check.png)

### - Collect Data

To begin collecting sensor data:

1. In the **Collect Data** section of the app UI, set a value (in seconds) for:
   - **Total Collection Duration**
   - **Session Timeout**

2. Click the **Start SSH & Collect** button.

3. To stop data collection manually before timeout, use the **Stop Job** button.

![Collect Data UI](assets/collect_data.png)

### - Download Data to Local Device

After the collection is complete:

![Download Data UI](assets/download_data.png)

- Click on **Download Data (per-ip)**.
- This will fetch sensor data to your local machine.

Downloaded data is stored at:

```
street-aware-scripts/data/<current-date>
```


You can then upload the collected data to the **Research Space** for future analysis. [! Note: Currently the video files generated are not research ready and needs some header fixes required to be implemented. See **fix_avi_header.py** in the next section]

---

## 4. Video Synchronization & Processing Toolkit

### Prerequisites
- Python 3.7+
- OpenCV (with Python bindings)
- NumPy
- natsort
- matplotlib (for timestamp analysis)
- ffmpeg (for AVI repair)
- GStreamer with Python bindings (optional, for GPU-accelerated encoding)
- CUDA-enabled GPU (optional, for GPU-accelerated processing)

Install dependencies:
```bash
cd street-aware-scripts
pip install -r requirements.txt
```

### Script Summary

#### Basic Analysis
- **check_frame_count.py**: Analyze available frames and timeline data.
  ```bash
  python check_frame_count.py <data_path>
  ```

#### Core Synchronization Scripts
- **sync_video.py**: **RECOMMENDED** - Enhanced video synchronization with timestamp analysis and correction for 4K multi-camera setup with GPU support.
  
  **Quick Start Example:**
  ```bash
  # Basic usage with default settings
  python sync_video.py /path/to/data
  
  # Custom output directory and processing first 100 frames
  python sync_video.py /path/to/data --output-dir my_output --max-frames 100
  
  # Full example with all options
  python sync_video.py /path/to/data --output-dir synced_videos --threshold 50 --max-frames 300 --fps 20 --rotation 0
  ```
  
  **Command Options:**
  - `data_path`: Path to data directory containing camera folders (required)
  - `--output-dir`: Output directory for synchronized videos (default: `synced_output_enhanced`)
  - `--threshold`: Synchronization threshold in milliseconds (default: 50)
  - `--max-frames`: Maximum number of frames to process (default: 300)
  - `--fps`: Output video frame rate (default: 20)
  - `--rotation`: Rotation angle: 0, 90, 180, or 270 degrees (default: 0)
  
  **Enhanced Features:**
  - Integrates `fix_timestamp.py` timestamp analysis logic
  - Detects and corrects frame drops and timing irregularities
  - Uses cross-timestamp correlation for accurate threshold conversion
  - Provides robust period calculation and frame ID correction
  - Handles network delays and clock drift automatically
  - Two-step timestamp conversion: GStreamer → Python → Global
  - 2-frame buffer window logic for optimal frame selection
  - Automatic GPU detection and fallback to CPU processing
  
  **Camera Configuration:**
  The script processes multiple cameras by default. Each camera is identified by an IP address and camera number (format: `IP_ADDRESS_CAMERA_NUMBER`, e.g., `192.168.0.108_0`).
  
  To configure which cameras to process:
  1. Open `sync_video.py` in a text editor
  2. Locate the `cameras` list in the `build_synchronized_videos_enhanced()` function
  3. Modify the list to include your camera IDs (format: `IP_ADDRESS_CAMERA_NUMBER`)
  
  **Expected Data Structure:**
  Your data directory should be organized as follows:
  ```
  <data_path>/
  ├── <IP_ADDRESS_1>/
  │   ├── video/
  │   │   ├── <CAMERA_NUM>_*.avi
  │   │   └── ...
  │   └── time/
  │       ├── <CAMERA_NUM>_*_*.json
  │       └── ...
  ├── <IP_ADDRESS_2>/
  │   └── ...
  └── ...
  ```


#### Video Repair
- **fix_avi_header.py**: Repair corrupted AVI headers (requires ffmpeg).
  ```bash
  python fix_avi_header.py <data_path> [--backup] [--cameras CAM1 CAM2 ...] [--test-only] [--verify-only] [--analyze-only]
  ```

**Tip:** All scripts support `--help` for option details.

### Output Structure

The `sync_video.py` script creates organized output folders:

```
<output_dir>/                      # Default: synced_output_enhanced
├── per_camera/                    # Individual synced videos per camera
│   ├── <IP_ADDRESS>_<CAMERA_NUM>_enhanced_sync.mp4
│   ├── <IP_ADDRESS>_<CAMERA_NUM>_enhanced_sync.mp4
│   └── ...                        # One video per configured camera
└── mosaic/                        # Mosaic video and metadata
    ├── mosaic_enhanced_sync.mp4   # Grid mosaic (info panel + camera feeds)
    ├── enhanced_sync_tracking.json # Detailed frame synchronization metadata
    └── master_timeline_enhanced.json # Master timeline with global timestamps
```

**Output Files:**
- **Mosaic Video**: Grid layout with info panel showing global timestamp, time, and frame number, plus all camera feeds arranged in a grid
- **Per-Camera Videos**: Individual synchronized videos for each configured camera (named as `<IP_ADDRESS>_<CAMERA_NUM>_enhanced_sync.mp4`)
- **Tracking JSON**: Detailed metadata including:
  - Synchronization parameters (threshold, conversion factors, correlation data)
  - Frame-by-frame synchronization info (which frames were used, distances, black frame reasons)
  - Final statistics (synchronization rate, camera utilization)
- **Master Timeline**: Global timestamps used for synchronization

---

## 5. Reference & Updates
For the most up-to-date documentation and updates, refer to the following Google Doc:

👉 [StreetAware Master Documentation](https://docs.google.com/document/d/1m13t26RZbAX_EhKLEvc13xLq-o2AdOja44-rMRILN5U/edit?usp=sharing)

