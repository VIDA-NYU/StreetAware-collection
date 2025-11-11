#!/usr/bin/env python3
# Enhanced Mosaic Video Synchronization with Timestamp Analysis
# Integrates fix_timestamp.py logic for robust timeline creation

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from natsort import natsorted
import shutil
from matplotlib import pyplot as plt

# Global timestamp conversion constants (from analysis)
# 1 global unit ≈ 5/6 milliseconds
MS_PER_GLOBAL_UNIT = 5/6   # Convert global units to ms
GLOBAL_UNITS_PER_MS = 6/5  # Convert ms to global units

def check_gpu_availability():
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"CUDA GPU detected: {gpu_count} device(s) available")
            return True
        else:
            print("No CUDA GPU devices found, using CPU")
            return False
    except:
        print("CUDA not available, using CPU")
        return False

def get_processing_device():
    if check_gpu_availability():
        return "gpu"
    else:
        return "cpu"

def resize_frame_gpu(frame, target_size):
    try:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
        return gpu_resized.download()
    except:
        return cv2.resize(frame, target_size)

def resize_frame_cpu(frame, target_size):
    return cv2.resize(frame, target_size)

def rotate_frame(frame, angle):
    if angle == 0:
        return frame
    elif angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Rotation angle must be 0, 90, 180, or 270")

def analyze_timestamps(ts, title, fps, n=10, debug=True, verbose=False):
    """
    Analyze timestamps to detect frame drops, duplicates, and calculate accurate periods.
    Based on fix_timestamp.py logic.
    """
    title, min_t = title, np.min(ts)
    ts = ts - min_t  # Work with relative values

    p_ref, dt = 1 / fps, np.diff(ts)
    
    # Use larger steps for more accurate period calculation
    p_ref_n, dt_n = n * p_ref, np.diff(ts[n::n])
    ps_n = dt_n[(0.95 * p_ref_n < dt_n) & (dt_n < 1.05 * p_ref_n)]
    per_n = np.mean(ps_n) if len(ps_n) > 0 else p_ref_n
    per = per_n / n

    # Compute median offset
    ofs = ts - np.round(ts / per) * per
    off = np.median(ofs)

    # Compute frame ids
    ids = np.round((ts - off) / per).astype(np.int32)

    # Check for duplicates after rounding due to frame acquisition delays
    for it in range(100):
        idx = np.nonzero(np.diff(ids) <= 0)[0]
        if idx.shape[0] == 0:
            break
        for i in reversed(idx):
            ids[i] = ids[i + 1] - 1

    # Remaining gaps (lost frames)
    gaps = np.nonzero(np.diff(ids) > 1)[0]
    lost = ids[gaps + 1] - ids[gaps] - 1

    if debug:
        print(f"\nAnalyzing {title} timestamps...")
        print(f"FPS: {fps}")
        print(f"Period: {per * 1000:.2f} ms")
        print(f"Offset: {off * 1000:.2f} ms")
        print(f"Gaps: {len(gaps)}")
        if len(lost):
            print(f"Lost frames: {lost}")

    return ids * per, per, (gaps, lost), (per, off)

def correlate(x, y, exclude_outliers=True):
    """
    Find linear correlation between two timestamp arrays.
    Based on fix_timestamp.py logic.
    """
    if exclude_outliers:
        m, std = np.mean(y), np.std(y)
        idx = np.nonzero(np.abs(y - m) / std < 3)[0]
        n_out = y.shape[0] - idx.shape[0]
        if n_out > 0:
            print(f"{n_out} outlier(s) removed")
        x, y = x[idx], y[idx]

    mb = np.polyfit(x, y, 1)  # fit a line
    return mb[0], mb[1]

def extract_timestamp_pairs(timeline_data):
    """
    Extract all (global_timestamp, gstreamer_timestamp) pairs from timeline data.
    """
    pairs = []
    for entry in timeline_data:
        if 'global_timestamp' in entry and 'gstreamer_timestamp' in entry:
            pairs.append((entry['global_timestamp'], entry['gstreamer_timestamp']))
    return pairs

def extract_all_timestamp_pairs(timeline_data):
    """
    Extract all timestamp pairs: (gstreamer, python, global) from timeline data.
    """
    pairs = []
    for entry in timeline_data:
        if all(key in entry for key in ['gstreamer_timestamp', 'python_timestamp', 'global_timestamp']):
            pairs.append((
                entry['gstreamer_timestamp'],
                entry['python_timestamp'], 
                entry['global_timestamp']
            ))
    return pairs

def convert_threshold_to_global_units_enhanced(threshold_ms, timeline_data):
    """
    Convert user threshold in milliseconds to global timestamp units using two-step conversion:
    1. GStreamer → Python timestamp
    2. Python timestamp → Global timestamp
    Based on fix_timestamp.py logic.
    """
    # Extract all timestamp triples
    triples = extract_all_timestamp_pairs(timeline_data)
    
    if len(triples) >= 2:
        # Convert to numpy arrays
        gstreamer_ts = np.array([t[0] for t in triples])
        python_ts = np.array([t[1] for t in triples])
        global_ts = np.array([t[2] for t in triples])
        
        # Step 1: GStreamer → Python timestamp correlation
        gs_to_py_slope, gs_to_py_intercept = correlate(gstreamer_ts, python_ts)
        print(f"GStreamer → Python: python_ts = {gs_to_py_slope:.6f} * gstreamer_ts + {gs_to_py_intercept:.6f}")
        
        # Step 2: Python → Global timestamp correlation  
        py_to_global_slope, py_to_global_intercept = correlate(python_ts, global_ts)
        print(f"Python → Global: global_ts = {py_to_global_slope:.6f} * python_ts + {py_to_global_intercept:.6f}")
        
        # Combined conversion factor
        seconds_per_global_unit = gs_to_py_slope * py_to_global_slope
        print(f"Combined conversion: seconds_per_global_unit = {seconds_per_global_unit:.6f}")
        
        # Convert threshold
        threshold_sec = threshold_ms / 1000.0
        threshold_global = threshold_sec / seconds_per_global_unit if seconds_per_global_unit != 0 else threshold_ms * GLOBAL_UNITS_PER_MS
        
        print(f"Threshold conversion: {threshold_ms} ms = {threshold_global:.2f} global units")
        return threshold_global, seconds_per_global_unit, (gs_to_py_slope, gs_to_py_intercept, py_to_global_slope, py_to_global_intercept)
    else:
        # Fallback to direct correlation
        pairs = extract_timestamp_pairs(timeline_data)
        if len(pairs) >= 2:
            global_ts = np.array([p[0] for p in pairs])
            gstreamer_ts = np.array([p[1] for p in pairs])
            
            slope, intercept = correlate(global_ts, gstreamer_ts)
            seconds_per_global_unit = slope
            
            print(f"Direct correlation (fallback): gstreamer_ts = {slope:.6f} * global_ts + {intercept:.6f}")
            print(f"Seconds per global unit: {seconds_per_global_unit:.6f}")
            
            threshold_sec = threshold_ms / 1000.0
            threshold_global = threshold_sec / seconds_per_global_unit if seconds_per_global_unit != 0 else threshold_ms * GLOBAL_UNITS_PER_MS
            
            print(f"Threshold conversion: {threshold_ms} ms = {threshold_global:.2f} global units")
            return threshold_global, seconds_per_global_unit, (slope, intercept, 1.0, 0.0)
        else:
            # Final fallback
            threshold_global = threshold_ms * GLOBAL_UNITS_PER_MS
            print(f"Using default threshold conversion: {threshold_ms} ms = {threshold_global:.2f} global units")
            return threshold_global, 1.0 / GLOBAL_UNITS_PER_MS, (1.0, 0.0, 1.0, 0.0)

def apply_two_step_conversion(gstreamer_ts, conversion_params):
    """
    Apply two-step conversion from gstreamer timestamp to global timestamp.
    Based on fix_timestamp.py logic: gs → python → global
    """
    gs_to_py_slope, gs_to_py_intercept, py_to_global_slope, py_to_global_intercept = conversion_params
    
    # Step 1: GStreamer → Python timestamp
    python_ts = gstreamer_ts * gs_to_py_slope + gs_to_py_intercept
    
    # Step 2: Python → Global timestamp
    global_ts = python_ts * py_to_global_slope + py_to_global_intercept
    
    return global_ts

class CameraReader:
    def __init__(self, camera_id, data_path, output_dir=None):
        self.camera_id = camera_id
        self.data_path = data_path
        self.output_dir = output_dir
        self.ip, self.cam_num = camera_id.split('_')
        
        # Load timeline with enhanced timestamp analysis
        self.timeline = self.load_enhanced_timeline()
        self.timeline_index = 0
        
        # Initialize video reader
        self.video_files = self.get_video_files()
        self.current_video_index = 0
        self.current_video = None
        
        # Two-frame buffer: left and right frames
        self.left_frame = None
        self.right_frame = None
        
        # Load first two frames
        self.load_next_frame()
        self.load_next_frame()
        
        print(f"Initialized {camera_id}: {len(self.timeline)} timestamps, {len(self.video_files)} video files")
    
    def preprocess_timeline_files(self):
        """Preprocess and merge JSON files like fix_timestamp.py."""
        time_path = os.path.join(self.data_path, self.ip, "time")
        
        if not os.path.exists(time_path):
            print(f"Time path not found: {time_path}")
            return False
        
        # Create merged file path in output directory
        if self.output_dir:
            merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        else:
            merged_file = os.path.join(self.data_path, f"{self.ip}_{self.cam_num}.json")
        
        # Check if merged file already exists
        if os.path.exists(merged_file):
            print(f"Using existing merged file: {merged_file}")
            return True
        
        print(f"Creating merged file for {self.camera_id}...")
        
        # Collect all JSON files for this camera
        pattern = f"{self.cam_num}_*_*.json"
        timestamp_files = list(Path(time_path).glob(pattern))
        timestamp_files = natsorted(timestamp_files)
        
        if not timestamp_files:
            print(f"No timestamp files found for {self.camera_id}")
            return False
        
        # Merge all files
        merged_data = []
        for file_path in timestamp_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    merged_data.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Save merged file
        try:
            with open(merged_file, 'w') as f:
                json.dump(merged_data, f, indent=4)
            print(f"Created merged file: {merged_file} with {len(merged_data)} entries")
            return True
        except Exception as e:
            print(f"Error saving merged file: {e}")
            return False
    
    def sort_timeline_data(self):
        """Sort merged timeline data like fix_timestamp.py."""
        if self.output_dir:
            merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        else:
            merged_file = os.path.join(self.data_path, f"{self.ip}_{self.cam_num}.json")
        
        if not os.path.exists(merged_file):
            print(f"Merged file not found: {merged_file}")
            return False
        
        print(f"Sorting timeline data for {self.camera_id}...")
        
        try:
            # Load merged data
            with open(merged_file, 'r') as f:
                merged_data = json.load(f)
            
            # Extract and sort buffer entries
            sorted_data = []
            for dictionary in merged_data:
                for key, value in natsorted(dictionary.items()):
                    if key.startswith('buffer_') and isinstance(value, dict):
                        sorted_data.append(value)
            
            # Save sorted data back to file
            with open(merged_file, 'w') as f:
                json.dump(sorted_data, f, indent=4)
            
            print(f"Sorted timeline data: {len(sorted_data)} entries")
            return True
        except Exception as e:
            print(f"Error sorting timeline data: {e}")
            return False
    
    def load_enhanced_timeline(self):
        """Load timeline with enhanced timestamp analysis and correction."""
        # First, preprocess and merge files
        if not self.preprocess_timeline_files():
            return []
        
        # Then sort the merged data
        if not self.sort_timeline_data():
            return []
        
        # Load the processed timeline data
        if self.output_dir:
            merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        else:
            merged_file = os.path.join(self.data_path, f"{self.ip}_{self.cam_num}.json")
        
        try:
            with open(merged_file, 'r') as f:
                timeline_data = json.load(f)
            
            # Convert to enhanced format with source tracking
            enhanced_timeline = []
            for i, entry in enumerate(timeline_data):
                if isinstance(entry, dict) and 'global_timestamp' in entry:
                    enhanced_entry = {
                        'global_timestamp': entry['global_timestamp'],
                        'gstreamer_timestamp': entry.get('gstreamer_timestamp'),
                        'python_timestamp': entry.get('python_timestamp'),
                        'frame_id': entry.get('frame_id'),
                        'entry_index': i
                    }
                    enhanced_timeline.append(enhanced_entry)
            
            # Sort by global timestamp
            enhanced_timeline.sort(key=lambda x: x['global_timestamp'])
            
            # Apply timestamp analysis and correction
            if len(enhanced_timeline) > 0:
                enhanced_timeline = self.analyze_and_correct_timestamps(enhanced_timeline)
            
            return enhanced_timeline
            
        except Exception as e:
            print(f"Error loading enhanced timeline: {e}")
            return []
    
    def analyze_and_correct_timestamps(self, timeline_data):
        """Apply timestamp analysis and correction to timeline data."""
        # Extract gstreamer timestamps for analysis
        gstreamer_timestamps = []
        valid_indices = []
        
        for i, entry in enumerate(timeline_data):
            if entry.get('gstreamer_timestamp') is not None:
                gstreamer_timestamps.append(entry['gstreamer_timestamp'])
                valid_indices.append(i)
        
        if len(gstreamer_timestamps) < 2:
            print(f"Warning: Insufficient gstreamer timestamps for {self.camera_id}")
            return timeline_data
        
        # Analyze timestamps
        gstreamer_ts = np.array(gstreamer_timestamps)
        corrected_ts, period, (gaps, lost), (per, off) = analyze_timestamps(
            gstreamer_ts, f"{self.camera_id}_gstreamer", 25.0, debug=True, verbose=False
        )
        
        # Apply corrections to timeline
        corrected_timeline = []
        for i, entry in enumerate(timeline_data):
            if i in valid_indices:
                ts_idx = valid_indices.index(i)
                if ts_idx < len(corrected_ts):
                    # Update with corrected timestamp
                    entry['corrected_gstreamer_timestamp'] = corrected_ts[ts_idx]
                    entry['original_gstreamer_timestamp'] = entry['gstreamer_timestamp']
                    entry['gstreamer_timestamp'] = corrected_ts[ts_idx]
            
            corrected_timeline.append(entry)
        
        print(f"Applied timestamp corrections to {self.camera_id}: {len(gaps)} gaps, {sum(lost) if lost.size > 0 else 0} lost frames")
        return corrected_timeline
    
    def get_video_files(self):
        """Get list of video files for this camera."""
        video_path = os.path.join(self.data_path, self.ip, "video")
        
        if not os.path.exists(video_path):
            return []
        
        pattern = f"{self.cam_num}_*.avi"
        video_files = list(Path(video_path).glob(pattern))
        return natsorted(video_files)
    
    def load_next_frame(self):
        """Load the next frame from timeline using 2-frame window logic."""
        while self.timeline_index < len(self.timeline):
            frame_info = self.timeline[self.timeline_index]
            frame_id = frame_info.get('frame_id')
            frame = self.get_frame_from_video_segments(frame_id)
            self.timeline_index += 1
            if frame is not None:
                # Update frame buffer (left -> right -> new)
                self.left_frame = self.right_frame
                self.right_frame = frame
                return True
        return False
    
    def get_frame_from_video_segments(self, frame_id):
        """Get a specific frame by frame ID from video segments."""
        for video_file in self.video_files:
            if self.current_video is None or self.current_video_index != self.video_files.index(video_file):
                if self.current_video is not None:
                    self.current_video.release()
                self.current_video = cv2.VideoCapture(str(video_file))
                self.current_video_index = self.video_files.index(video_file)
            
            if not self.current_video.isOpened():
                continue
            
            total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_id is not None and frame_id < total_frames:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = self.current_video.read()
                
                if ret:
                    return frame
            else:
                continue
        
        return None
    
    def get_current_frame_timestamps(self):
        """Get global timestamps for the currently loaded left and right frames."""
        if self.timeline_index >= len(self.timeline):
            return None, None
        
        # Get timestamps for the currently loaded frames
        left_timestamp = self.timeline[self.timeline_index - 2]['global_timestamp'] if self.timeline_index > 1 else None
        right_timestamp = self.timeline[self.timeline_index - 1]['global_timestamp'] if self.timeline_index > 0 else None
        
        return left_timestamp, right_timestamp
    
    def get_current_frames(self):
        """Get current left and right frames."""
        return self.left_frame, self.right_frame
    
    def get_best_frame_for_timestamp(self, target_global_timestamp, threshold_global_units):
        """
        Find the best frame from left/right buffer for the target global timestamp.
        """
        left_timestamp, right_timestamp = self.get_current_frame_timestamps()
        left_frame, right_frame = self.get_current_frames()
        
        best_frame = None
        best_distance = float('inf')
        best_info = None
        
        # Check both left and right frames
        for idx, (frame, ts) in enumerate(zip([left_frame, right_frame], [left_timestamp, right_timestamp])):
            if ts is not None and frame is not None:
                distance = abs(target_global_timestamp - ts)
                if distance < best_distance:
                    best_distance = distance
                    best_frame = frame
                    
                    # Get frame info from timeline
                    timeline_idx = self.timeline_index - 2 + idx
                    if 0 <= timeline_idx < len(self.timeline):
                        timeline_entry = self.timeline[timeline_idx]
                        picked_frame_id = timeline_entry.get('frame_id')
                        picked_gstreamer_timestamp = timeline_entry.get('gstreamer_timestamp')
                    else:
                        picked_frame_id = None
                        picked_gstreamer_timestamp = None
                    
                    best_info = {
                        'picked_global_timestamp': ts,
                        'picked_gstreamer_timestamp': picked_gstreamer_timestamp,
                        'picked_frame_id': picked_frame_id,
                        'distance_global_units': distance,
                        'distance_ms': distance * MS_PER_GLOBAL_UNIT,
                        'buffer_position': 'left' if idx == 0 else 'right'
                    }
        
        # Check if best frame is within threshold
        if best_frame is not None and best_distance <= threshold_global_units:
            return best_frame, best_info
        else:
            return None, best_info
    
    def advance_frame(self):
        """Advance to next frame using 2-frame window logic."""
        return self.load_next_frame()
    
    def should_advance_for_timestamp(self, target_global_timestamp, threshold_global_units):
        """Determine if camera should advance frames based on target timestamp."""
        left_timestamp, right_timestamp = self.get_current_frame_timestamps()
        
        should_advance = False
        if left_timestamp is not None and right_timestamp is not None:
            if target_global_timestamp > max(left_timestamp, right_timestamp):
                should_advance = True
            elif target_global_timestamp > right_timestamp + threshold_global_units:
                should_advance = True
        elif right_timestamp is not None:
            if target_global_timestamp > right_timestamp:
                should_advance = True
        
        return should_advance
    
    def advance_to_timestamp(self, target_global_timestamp, threshold_global_units):
        """Advance frames until we have frames appropriate for the target timestamp."""
        if self.should_advance_for_timestamp(target_global_timestamp, threshold_global_units):
            advanced = True
            attempts = 0
            while advanced and attempts < 10:
                advanced = self.advance_frame()
                attempts += 1
                if advanced:
                    new_left, new_right = self.get_current_frame_timestamps()
                    if new_right is not None and abs(target_global_timestamp - new_right) < abs(target_global_timestamp - (new_right or float('inf'))):
                        break
    
    def cleanup(self):
        """Clean up video reader."""
        if self.current_video is not None:
            self.current_video.release()

def create_enhanced_master_timeline(all_camera_readers, fps=30):
    """
    Create enhanced master timeline using timestamp analysis and correlation.
    """
    print("Creating enhanced master timeline with timestamp analysis...")
    
    # Collect all timeline data for correlation analysis
    all_timeline_data = []
    for reader in all_camera_readers:
        all_timeline_data.extend(reader.timeline)
    
    # Extract timestamp pairs for correlation
    pairs = extract_timestamp_pairs(all_timeline_data)
    
    if len(pairs) >= 2:
        # Convert to numpy arrays
        global_ts = np.array([p[0] for p in pairs])
        gstreamer_ts = np.array([p[1] for p in pairs])
        
        # Find linear relationship
        slope, intercept = correlate(global_ts, gstreamer_ts)
        seconds_per_global_unit = slope
        
        print(f"Global correlation: gstreamer_ts = {slope:.6f} * global_ts + {intercept:.6f}")
        print(f"Seconds per global unit: {seconds_per_global_unit:.6f}")
    else:
        seconds_per_global_unit = 1.0 / GLOBAL_UNITS_PER_MS
        print("Using default seconds per global unit")
    
    # Collect all global timestamps
    all_global_timestamps = []
    for reader in all_camera_readers:
        for entry in reader.timeline:
            all_global_timestamps.append(entry['global_timestamp'])
    
    if not all_global_timestamps:
        print("No global timestamps found!")
        return []
    
    # Find start and end times
    start_time = min(all_global_timestamps)
    end_time = max(all_global_timestamps)
    
    print(f"Global timeline range: {start_time} to {end_time}")
    print(f"Total duration: {end_time - start_time} global units")
    print(f"Duration in seconds: {(end_time - start_time) * seconds_per_global_unit:.2f}")
    
    # Calculate frame interval in global units
    frame_interval_sec = 1.0 / fps
    frame_interval_global = frame_interval_sec / seconds_per_global_unit
    
    print(f"Frame interval: {frame_interval_global:.2f} global units ({frame_interval_sec*1000:.2f} ms)")
    
    # Create master timeline with regular intervals
    master_timeline = []
    current_time = start_time
    while current_time <= end_time:
        master_timeline.append(int(current_time))
        current_time += frame_interval_global
    
    print(f"Created enhanced master timeline with {len(master_timeline)} points")
    
    return master_timeline, seconds_per_global_unit

def build_synchronized_videos_enhanced(data_path, output_dir, threshold_ms=50, max_frames=300, fps=20, rotation=0):
    """
    Build synchronized videos using enhanced timestamp analysis.
    """
    print("Building Synchronized Videos with Enhanced Timestamp Analysis")
    
    processing_device = get_processing_device()
    resize_function = resize_frame_gpu if processing_device == "gpu" else resize_frame_cpu
    
    cameras = [
        '192.168.0.108_0', '192.168.0.108_2',
        '192.168.0.122_0', '192.168.0.122_2',
        '192.168.0.184_0', '192.168.0.184_2',
        '192.168.0.227_0', '192.168.0.227_2'
    ]
    
    # Prepare output directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    per_camera_dir = os.path.join(output_dir, 'per_camera')
    mosaic_dir = os.path.join(output_dir, 'mosaic')
    os.makedirs(per_camera_dir, exist_ok=True)
    os.makedirs(mosaic_dir, exist_ok=True)
    
    print("\nStep 1: Initializing camera readers with enhanced timeline analysis")
    camera_readers = []
    for camera_id in cameras:
        print(f"Initializing {camera_id}")
        reader = CameraReader(camera_id, data_path, output_dir)
        camera_readers.append(reader)
    
    print("\nStep 2: Creating enhanced master timeline")
    master_timeline, seconds_per_global_unit = create_enhanced_master_timeline(camera_readers, fps)
    if max_frames:
        master_timeline = master_timeline[:max_frames]
        print(f"Processing first {max_frames} frames")
    
    # Convert threshold using correct 1200 global units per second relationship
    all_timeline_data = []
    for reader in camera_readers:
        all_timeline_data.extend(reader.timeline)
    
    # Use the known relationship: 1200 global units = 1 second
    RADIO_FREQ = 1200  # global units per second (from fix_timestamp.py)
    seconds_per_global_unit = 1.0 / RADIO_FREQ
    
    # Convert threshold: threshold_ms -> global units
    threshold_sec = threshold_ms / 1000.0
    threshold_global_units = threshold_sec * RADIO_FREQ
    
    print(f"Using correct conversion: 1200 global units = 1 second")
    print(f"Seconds per global unit: {seconds_per_global_unit:.6f}")
    print(f"Threshold conversion: {threshold_ms} ms = {threshold_global_units:.2f} global units")
    
    # For metadata, use direct correlation values if available
    pairs = extract_timestamp_pairs(all_timeline_data)
    if len(pairs) >= 2:
        global_ts = np.array([p[0] for p in pairs])
        gstreamer_ts = np.array([p[1] for p in pairs])
        slope, intercept = correlate(global_ts, gstreamer_ts)
        conversion_params = (slope, intercept, 1.0, 0.0)
        print(f"Direct correlation: gstreamer_ts = {slope:.6f} * global_ts + {intercept:.6f}")
    else:
        conversion_params = (seconds_per_global_unit, 0.0, 1.0, 0.0)
    
    # Save master timeline
    with open(os.path.join(mosaic_dir, 'master_timeline_enhanced.json'), 'w') as f:
        json.dump(master_timeline, f, indent=2)
    print(f"Saved enhanced master timeline to {os.path.join(mosaic_dir, 'master_timeline_enhanced.json')}")
    
    # Frame tracking with enhanced metadata
    frame_tracking = {
        "synchronization_info": {
            "method": "enhanced_timestamp_analysis",
            "threshold_ms": threshold_ms,
            "threshold_global_units": threshold_global_units,
            "seconds_per_global_unit": seconds_per_global_unit,
            "conversion_method": "direct_correlation",
            "correlation_slope": conversion_params[0],
            "correlation_intercept": conversion_params[1],
            "max_frames": max_frames,
            "fps": fps,
            "rotation": rotation,
            "cameras": cameras
        },
        "frame_sequence": []
    }
    
    print("\nStep 3: Determining video dimensions")
    sample_frame = None
    for reader in camera_readers:
        if reader.right_frame is not None:
            sample_frame = reader.right_frame
            break
    
    if sample_frame is None:
        print("Error: Could not find any sample frame to determine video dimensions")
        return
    
    # Apply rotation to sample frame to get correct output size
    sample_frame_rot = rotate_frame(sample_frame, rotation)
    height, width = sample_frame_rot.shape[:2]
    print(f"Video dimensions after rotation: {width}x{height}")
    
    print("\nStep 4: Creating video writers")
    # Mosaic layout: 3x3 grid
    mosaic_width = width * 3
    mosaic_height = height * 3
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    mosaic_output_file = os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4')
    mosaic_video = cv2.VideoWriter(mosaic_output_file, fourcc, fps, (mosaic_width, mosaic_height))
    
    if not mosaic_video.isOpened():
        print("H.264 failed, trying MJPG codec for mosaic...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        mosaic_video = cv2.VideoWriter(mosaic_output_file, fourcc, fps, (mosaic_width, mosaic_height))
        if not mosaic_video.isOpened():
            print("Error: Could not create mosaic video writer")
            return
    
    # Per-camera writers
    camera_writers = {}
    for camera_id in cameras:
        cam_file = os.path.join(per_camera_dir, f"{camera_id}_enhanced_sync.mp4")
        writer = cv2.VideoWriter(cam_file, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"H.264 failed, trying MJPG for {camera_id}...")
            writer = cv2.VideoWriter(cam_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        camera_writers[camera_id] = writer
    
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    print(f"\nStep 5: Processing {len(master_timeline)} enhanced timestamps")
    print(f"Using threshold: {threshold_ms} ms = {threshold_global_units:.2f} global units")
    
    frame_count = 0
    stats = {
        'total_frames': 0,
        'synchronized_frames': 0,
        'black_frames': 0,
        'camera_stats': {camera_id: {'actual': 0, 'black': 0} for camera_id in cameras}
    }
    
    for global_timestamp in master_timeline:
        if frame_count % 30 == 0:
            print(f"Progress: {frame_count}/{len(master_timeline)} ({frame_count/len(master_timeline)*100:.1f}%)")
        
        # Advance all camera buffers to current timestamp
        for reader in camera_readers:
            reader.advance_to_timestamp(global_timestamp, threshold_global_units)
        
        output_frames = []
        timestamp_frame_info = {
            "global_timestamp": global_timestamp,
            "frame_number": frame_count,
            "timestamp_seconds": global_timestamp * seconds_per_global_unit,
            "camera_frames": {}
        }
        
        # Process each camera
        for reader in camera_readers:
            frame, frame_info = reader.get_best_frame_for_timestamp(global_timestamp, threshold_global_units)
            
            if frame is not None:
                # Process frame
                if frame.shape[:2] != (height, width):
                    frame = resize_function(frame, (width, height))
                frame = rotate_frame(frame, rotation)
                
                output_frames.append(frame.copy())
                
                # Record frame info
                frame_info['frame_type'] = 'actual_frame'
                timestamp_frame_info["camera_frames"][reader.camera_id] = frame_info
                stats['camera_stats'][reader.camera_id]['actual'] += 1
            else:
                # Use black frame
                output_frames.append(black_frame.copy())
                
                # Record black frame info
                black_info = {
                    'frame_type': 'black_frame',
                    'reason': 'no_frame_within_threshold'
                }
                if frame_info:
                    black_info.update({
                        'nearest_frame_global_timestamp': frame_info['picked_global_timestamp'],
                        'nearest_frame_distance_global_units': frame_info['distance_global_units'],
                        'nearest_frame_distance_ms': frame_info['distance_ms']
                    })
                
                timestamp_frame_info["camera_frames"][reader.camera_id] = black_info
                stats['camera_stats'][reader.camera_id]['black'] += 1
            
            # Write to individual camera video
            camera_writers[reader.camera_id].write(output_frames[-1])
        
        # Add frame info to tracking sequence
        frame_tracking["frame_sequence"].append(timestamp_frame_info)
        
        # Create mosaic layout (3x3 grid with info panel)
        info_frame = black_frame.copy()
        cv2.putText(info_frame, f'Global TS: {global_timestamp}', (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_frame, f'Time: {global_timestamp * seconds_per_global_unit:.3f}s', (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_frame, f'Frame: {frame_count}', (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Arrange in 3x3 grid: info + 8 cameras
        output_frames_with_info = [info_frame] + output_frames
        
        row1 = np.concatenate(output_frames_with_info[0:3], axis=1)
        row2 = np.concatenate(output_frames_with_info[3:6], axis=1)
        row3 = np.concatenate(output_frames_with_info[6:9], axis=1)
        mosaic_frame = np.concatenate((row1, row2, row3), axis=0)
        
        mosaic_video.write(mosaic_frame)
        frame_count += 1
        stats['total_frames'] += 1
    
    # Step 6: Clean up and save results
    mosaic_video.release()
    for writer in camera_writers.values():
        writer.release()
    for reader in camera_readers:
        reader.cleanup()
    
    # Calculate final statistics
    total_actual = sum(stats['camera_stats'][cam]['actual'] for cam in cameras)
    total_black = sum(stats['camera_stats'][cam]['black'] for cam in cameras)
    
    frame_tracking["final_statistics"] = {
        "total_frames_processed": stats['total_frames'],
        "total_actual_frames": total_actual,
        "total_black_frames": total_black,
        "synchronization_rate": total_actual / (total_actual + total_black) if (total_actual + total_black) > 0 else 0,
        "camera_utilization": {
            cam: {
                'actual_frames': stats['camera_stats'][cam]['actual'],
                'black_frames': stats['camera_stats'][cam]['black'],
                'utilization_rate': stats['camera_stats'][cam]['actual'] / (stats['camera_stats'][cam]['actual'] + stats['camera_stats'][cam]['black']) if (stats['camera_stats'][cam]['actual'] + stats['camera_stats'][cam]['black']) > 0 else 0
            }
            for cam in cameras
        }
    }
    
    print("\nStep 6: Saving results and statistics")
    tracking_file = os.path.join(mosaic_dir, 'enhanced_sync_tracking.json')
    with open(tracking_file, 'w') as f:
        json.dump(frame_tracking, f, indent=2)
    
    print(f"\n=== ENHANCED SYNCHRONIZATION RESULTS ===")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total actual frames used: {total_actual}")
    print(f"Total black frames used: {total_black}")
    print(f"Overall synchronization rate: {frame_tracking['final_statistics']['synchronization_rate']*100:.1f}%")
    
    print(f"\nCamera utilization rates:")
    for camera_id in cameras:
        util = frame_tracking['final_statistics']['camera_utilization'][camera_id]
        print(f"  {camera_id}: {util['utilization_rate']*100:.1f}% ({util['actual_frames']}/{util['actual_frames'] + util['black_frames']} frames)")
    
    print(f"\nOutput files:")
    print(f"  Mosaic video: {mosaic_output_file}")
    print(f"  Individual videos: {per_camera_dir}/")
    print(f"  Tracking data: {tracking_file}")
    print(f"  Master timeline: {os.path.join(mosaic_dir, 'master_timeline_enhanced.json')}")

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize videos using enhanced timestamp analysis with fix_timestamp.py logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  python sync_video.py <data_path> [--output-dir OUTPUT_DIR] [--threshold THRESHOLD] [--max-frames MAX_FRAMES] [--fps FPS] [--rotation ROTATION]

ENHANCEMENTS:
  - Integrates fix_timestamp.py timestamp analysis logic
  - Detects and corrects frame drops and timing irregularities
  - Uses cross-timestamp correlation for accurate threshold conversion
  - Provides robust period calculation and frame ID correction
  - Handles network delays and clock drift automatically
        """
    )
    parser.add_argument("data_path", help="Path to data directory containing camera folders")
    parser.add_argument("--output-dir", default="synced_output_enhanced", help="Output directory for videos and tracking info")
    parser.add_argument("--threshold", type=int, default=50, help="Threshold for frame selection (milliseconds)")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument("--rotation", type=int, default=0, help="Rotation angle (0, 90, 180, 270)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} not found")
        return
    
    if args.rotation not in [0, 90, 180, 270]:
        print("Error: --rotation must be 0, 90, 180, or 270")
        return
    
    build_synchronized_videos_enhanced(
        args.data_path, 
        args.output_dir, 
        args.threshold, 
        args.max_frames, 
        args.fps, 
        args.rotation
    )

if __name__ == "__main__":
    main() 