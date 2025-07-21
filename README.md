# StreetAware Project & Video Synchronization Toolkit

This guide covers the complete workflow for the StreetAware project, from data collection to advanced video synchronization and repair. All steps are streamlined for clarity and professional use.

---

## 1. Data Collection & Transfer
- **Power on** cameras and sensors; ensure AC power and network connection.
- **Enable** the Global Time Sync Module.
- **Clone** the SSH script repository:
  ```bash
  git clone https://github.com/VIDA-NYU/StreetAware-collection.git
  ```
- **Run** the SSH multi-run script:
  ```bash
  python ssh_multiple_run_script.py
  ```
- **Stop** after desired duration (Ctrl+C).
- **Find sensor IPs** via router dashboard.
- **Copy data** from `/Data` on each sensor using `scp` or drag-and-drop.
- **Upload** collected data to the Research Space.

---

## 2. Environment Setup
- **Clone** the repository (if not already):
  ```bash
  git clone https://github.com/VIDA-NYU/StreetAware-collection.git
  ```
- **Frontend:**
  ```bash
  cd street-aware-app
  npm install
  ```
- **Backend:**
  ```bash
  cd street-aware-service
  python -m venv myenv
  source myenv/bin/activate
  pip install -r requirements.txt
  ```
- **Scripts:**
  ```bash
  cd street-aware-scripts
  source myenv/bin/activate
  pip install -r requirements.txt
  ```

---

## 3. Running the Application
- **Frontend:**
  ```bash
  cd street-aware-app
  npm run start
  ```
- **Backend:**
  ```bash
  cd street-aware-service
  source myenv/bin/activate
  python app.py
  ```
- **Scripts:**
  ```bash
  cd street-aware-scripts
  source myenv/bin/activate
  # Run scripts as needed
  ```
- **Alternative:** Use `setup.sh`, `run.sh`, and `stop.sh` for automated setup and control.

---

## 4. Using the App
- **Health Check:** UI section verifies sensor connectivity.
- **Collect Data:** Set duration and session timeout, then start collection from the UI.
- **Download Data:** Use the UI to fetch data to your local device. Data is stored in `street-aware-scripts/data/<current-date>`.

---

## 5. Video Synchronization & Processing Toolkit

### Prerequisites
- Python 3.7+
- OpenCV (with Python bindings)
- NumPy
- natsort
- ffmpeg (for AVI repair)

Install dependencies:
```bash
pip install -r requirements.txt
```

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

## 6. Reference & Updates
- For the latest documentation, see the [StreetAware Master Documentation](https://docs.google.com/document/d/1m13t26RZbAX_EhKLEvc13xLq-o2AdOja44-rMRILN5U/edit?usp=sharing).

---

**Efficient, professional, and ready for production workflows.**
