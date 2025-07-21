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
- ffmpeg (for AVI repair)

### Script Summary
- **check_frame_count.py**: Analyze available frames and timeline data.
  ```bash
  python check_frame_count.py <data_path>
  ```
- **sync_videos.py**: Create a synchronized mosaic video (sequential, CPU/GPU).
  ```bash
  python sync_videos.py <data_path> [--output OUTPUT] [--threshold THRESHOLD] [--max-frames MAX_FRAMES] [--fps FPS]
  ```
- **sync_videos_GPU.py**: Fast, parallel CUDA mosaic video creation (GPU-accelerated, CPU fallback).
  ```bash
  python sync_videos_GPU.py <data_path> [--output OUTPUT] [--threshold THRESHOLD] [--max-frames MAX_FRAMES] [--fps FPS]
  ```
- **fix_avi_header.py**: Repair corrupted AVI headers (requires ffmpeg).
  ```bash
  python fix_avi_header.py <data_path> [--backup] [--cameras CAM1 CAM2 ...] [--test-only] [--verify-only] [--analyze-only]
  ```

**Tip:** All scripts support `--help` for option details.

---

## 5. Reference & Updates
For the most up-to-date documentation and updates, refer to the following Google Doc:

👉 [StreetAware Master Documentation](https://docs.google.com/document/d/1m13t26RZbAX_EhKLEvc13xLq-o2AdOja44-rMRILN5U/edit?usp=sharing)

