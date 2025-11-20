#!/usr/bin/env python3
"""
Comprehensive Video Synchronization Script
Combines logic from merge.py, render.py, fix_timestamp.py, and gst_video.py
Updated for 4K video and 20 FPS
"""

import os
import json
import glob
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt

# Video parameters for 4K
W, H, FPS = 3840, 2160, 20  # 4K resolution, 20 FPS
RADIO_FREQ = 1200  # global units per second
DEFAULT_BITRATE = 50000  # kbps for 4K

# GStreamer imports (if available)
try:
    from gstreamer import Gst, GstApp, GstContext, GstPipeline
    import gstreamer.utils as utils
    GSTREAMER_AVAILABLE = True
except ImportError:
    GSTREAMER_AVAILABLE = False
    print("Warning: GStreamer not available, using OpenCV fallback")


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class GstVideoWriter:
    """GStreamer-based video writer for high-quality 4K encoding"""
    
    def __init__(self, filename, width, height, fps, format="BGR", bitrate=DEFAULT_BITRATE, 
                 variable=True, codec="h265", gpu=0):
        if not GSTREAMER_AVAILABLE:
            raise ImportError("GStreamer not available")
            
        self.context = GstContext()
        self.context.startup()
        self.filename = filename
        p = filename.rfind(".")
        self.temp_filename = filename[:p] + "_temp" + filename[p:]
        print(f"Temporary video file: {self.temp_filename}")

        assert codec in ["h264", "h265"], f"Unsupported codec {codec}"

        if isinstance(fps, float):
            self.caps = f"video/x-raw,format={format},width={width},height={height},framerate=(fraction){int(round(fps * 100_000))}/100000"
        else:
            self.caps = f"video/x-raw,format={format},width={width},height={height},framerate={fps}/1"

        self.command = f"appsrc emit-signals=True is-live=False caps={self.caps} ! queue ! videoconvert ! " \
                      f"nv{codec}enc preset=hq bitrate={bitrate} rc-mode={'vbr' if variable else 'cbr'} " \
                      f"gop-size=45 cuda-device-id={gpu} ! {codec}parse ! matroskamux ! filesink location={self.temp_filename}"
        
        self.pts = 0  # frame timestamp
        self.duration = 10**9 / fps  # frame duration
        self.pipeline, self.appsrc = GstPipeline(self.command), None
        self.pipeline._on_pipeline_init = self.on_pipeline_init
        self.pipeline.startup()

    def on_pipeline_init(self):
        """Setup AppSrc element"""
        self.appsrc = self.pipeline.get_by_cls(GstApp.AppSrc)[0]
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", True)
        self.appsrc.set_property("max-bytes", 200_000_000)
        self.appsrc.set_caps(Gst.Caps.from_string(self.caps))

    def write(self, frame):
        gst_buffer = utils.ndarray_to_gst_buffer(frame)
        gst_buffer.pts = self.pts
        gst_buffer.duration = self.duration
        self.pts += self.duration
        self.appsrc.emit("push-buffer", gst_buffer)

    def close(self):
        self.appsrc.emit("end-of-stream")
        while not self.pipeline.is_done:
            time.sleep(.1)
        self.pipeline.shutdown()
        self.context.shutdown()
        
        if os.path.exists(self.filename):
            print(f"Overwriting {self.filename}")
            os.remove(self.filename)
        os.rename(self.temp_filename, self.filename)


class OpenCVVideoWriter:
    """OpenCV fallback video writer"""
    
    def __init__(self, filename, width, height, fps, codec="mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            print(f"Failed to create video writer with {codec}, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create video writer for {filename}")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


def analyze_timestamps(ts, title, fps, n=10, debug=True, verbose=False):
    """
    Analyze timestamps to detect frame drops, duplicates, and calculate accurate periods.
    Updated for 20 FPS default.
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
    """Find linear correlation between two timestamp arrays."""
    if exclude_outliers:
        m, std = np.mean(y), np.std(y)
        idx = np.nonzero(np.abs(y - m) / std < 3)[0]
        n_out = y.shape[0] - idx.shape[0]
        if n_out > 0:
            print(f"{n_out} outlier(s) removed")
        x, y = x[idx], y[idx]

    mb = np.polyfit(x, y, 1)  # fit a line
    return mb[0], mb[1]


# These functions are now handled within the CameraReader class


class CameraReader:
    """Enhanced camera reader with 4K support and 20 FPS analysis"""
    
    def __init__(self, camera_id, data_path, output_dir):
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
        """Preprocess and merge JSON files."""
        time_path = os.path.join(self.data_path, self.ip, "time")
        
        if not os.path.exists(time_path):
            print(f"Time path not found: {time_path}")
            return False
        
        # Create merged file path in output directory
        merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        
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
        """Sort merged timeline data."""
        merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        
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
                if isinstance(dictionary, dict):
                    # Look for buffer entries in the dictionary
                    for key, value in natsorted(dictionary.items()):
                        if key.startswith('buffer_') and isinstance(value, dict):
                            sorted_data.append(value)
                        elif key == 'global_timestamp' and isinstance(dictionary, dict):
                            # If this is already a timeline entry, add it directly
                            sorted_data.append(dictionary)
                            break
            
            # If no buffer entries found, check if the data is already in timeline format
            if len(sorted_data) == 0:
                for item in merged_data:
                    if isinstance(item, dict) and 'global_timestamp' in item:
                        sorted_data.append(item)
            
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
        merged_file = os.path.join(self.output_dir, f"{self.ip}_{self.cam_num}.json")
        
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
        
        # Analyze timestamps with 20 FPS
        gstreamer_ts = np.array(gstreamer_timestamps)
        corrected_ts, period, (gaps, lost), (per, off) = analyze_timestamps(
            gstreamer_ts, f"{self.camera_id}_gstreamer", 20.0, debug=True, verbose=False
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
        if frame_id is None:
            return None
            
        # Calculate cumulative frame counts to map global frame_id to correct video file
        cumulative_frames = 0
        target_video_index = -1
        local_frame_id = frame_id
        
        for i, video_file in enumerate(self.video_files):
            if self.current_video is None or self.current_video_index != i:
                if self.current_video is not None:
                    self.current_video.release()
                self.current_video = cv2.VideoCapture(str(video_file))
                self.current_video_index = i
            
            if not self.current_video.isOpened():
                continue
            
            total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Check if frame_id falls within this video file's range
            if frame_id >= cumulative_frames and frame_id < cumulative_frames + total_frames:
                target_video_index = i
                local_frame_id = frame_id - cumulative_frames
                break
            
            cumulative_frames += total_frames
        
        # If we found the target video, read the frame
        if target_video_index >= 0 and self.current_video is not None and self.current_video.isOpened():
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, local_frame_id)
            ret, frame = self.current_video.read()
            
            if ret:
                # Debug: Log video segment transitions (only for first few frames of each segment)
                if local_frame_id < 5:  # Only log first 5 frames of each segment
                    video_name = os.path.basename(str(self.video_files[target_video_index]))
                    print(f"  {self.camera_id}: Global frame {frame_id} -> {video_name} frame {local_frame_id}")
                return frame
            else:
                # Debug: Log failed frame reads
                if target_video_index < len(self.video_files):
                    video_name = os.path.basename(str(self.video_files[target_video_index]))
                    print(f"  {self.camera_id}: Failed to read frame {local_frame_id} from {video_name}")
        
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
        """Find the best frame from left/right buffer for the target global timestamp."""
        left_timestamp, right_timestamp = self.get_current_frame_timestamps()
        left_frame, right_frame = self.get_current_frames()
        
        best_frame = None
        best_distance = float('inf')
        best_info = None
        
        # Check both left and right frames
        for idx, (frame, ts) in enumerate(zip([left_frame, right_frame], [left_timestamp, right_timestamp])):
            if frame is not None and ts is not None:
                distance = abs(ts - target_global_timestamp)
                if distance <= threshold_global_units and distance < best_distance:
                    best_frame = frame
                    best_distance = distance
                    best_info = {
                        'picked_global_timestamp': ts,
                        'picked_gstreamer_timestamp': self.timeline[self.timeline_index - 2 + idx].get('gstreamer_timestamp'),
                        'picked_frame_id': self.timeline[self.timeline_index - 2 + idx].get('frame_id'),
                        'distance_global_units': distance,
                        'distance_ms': distance / RADIO_FREQ * 1000,
                        'buffer_position': 'left' if idx == 0 else 'right',
                        'frame_type': 'actual_frame'
                    }
        
        if best_frame is None:
            # Return black frame info
            best_info = {
                'frame_type': 'black_frame',
                'reason': 'no_frame_within_threshold'
            }
            if left_timestamp is not None:
                best_info['nearest_frame_global_timestamp'] = left_timestamp
                best_info['nearest_frame_distance_global_units'] = abs(left_timestamp - target_global_timestamp)
                best_info['nearest_frame_distance_ms'] = best_info['nearest_frame_distance_global_units'] / RADIO_FREQ * 1000
        
        return best_frame, best_info
    
    def advance_frame(self):
        """Advance to the next frame."""
        return self.load_next_frame()
    
    def should_advance_for_timestamp(self, target_global_timestamp, threshold_global_units):
        """Check if we should advance frames for the target timestamp."""
        left_timestamp, right_timestamp = self.get_current_frame_timestamps()
        
        if left_timestamp is None or right_timestamp is None:
            return False
        
        # Advance if target is past both buffered frames or significantly ahead
        return (target_global_timestamp > max(left_timestamp, right_timestamp) or 
                target_global_timestamp > right_timestamp + threshold_global_units)
    
    def advance_to_timestamp(self, target_global_timestamp, threshold_global_units):
        """Advance frames until we have frames around the target timestamp."""
        while self.should_advance_for_timestamp(target_global_timestamp, threshold_global_units):
            if not self.advance_frame():
                break
    
    def cleanup(self):
        """Clean up resources."""
        if self.current_video is not None:
            self.current_video.release()


def create_master_timeline(camera_readers, fps=20):
    """Create a master timeline for synchronization."""
    print("Creating enhanced master timeline with timestamp analysis...")
    
    # Collect all global timestamps
    all_global_timestamps = []
    for reader in camera_readers:
        if reader.timeline:
            timestamps = [entry['global_timestamp'] for entry in reader.timeline]
            all_global_timestamps.extend(timestamps)
    
    if not all_global_timestamps:
        print("No global timestamps found!")
        return [], 1.0 / RADIO_FREQ
    
    # Find timeline range
    min_ts = min(all_global_timestamps)
    max_ts = max(all_global_timestamps)
    
    # Calculate frame interval in global units
    seconds_per_global_unit = 1.0 / RADIO_FREQ
    frame_interval_global = int(1.0 / fps / seconds_per_global_unit)
    
    print(f"Global timeline range: {min_ts} to {max_ts}")
    print(f"Total duration: {max_ts - min_ts} global units")
    print(f"Duration in seconds: {(max_ts - min_ts) * seconds_per_global_unit:.2f}")
    print(f"Frame interval: {frame_interval_global} global units ({1000/fps:.2f} ms)")
    
    # Create master timeline
    master_timeline = list(range(min_ts, max_ts + 1, frame_interval_global))
    print(f"Created enhanced master timeline with {len(master_timeline)} points")
    
    return master_timeline, seconds_per_global_unit


def build_synchronized_videos(data_path, output_dir, threshold_ms=100, max_frames=None, fps=20, rotation=0):
    """Build synchronized videos using the comprehensive approach."""
    print("Building Synchronized Videos with Comprehensive Analysis")
    
    # Create output directories
    per_camera_dir = os.path.join(output_dir, "per_camera")
    mosaic_dir = os.path.join(output_dir, "mosaic")
    os.makedirs(per_camera_dir, exist_ok=True)
    os.makedirs(mosaic_dir, exist_ok=True)
    
    # Find camera directories
    camera_dirs = []
    for item in os.listdir(data_path):
        if item.startswith("192.168."):
            camera_dirs.append(item)
    
    if not camera_dirs:
        print("No camera directories found!")
        return
    
    # Create camera readers
    print("\nStep 1: Initializing camera readers with enhanced timeline analysis")
    camera_readers = []
    cameras = []
    
    for camera_dir in camera_dirs:
        # Dynamically discover available camera numbers by checking time directory
        time_path = os.path.join(data_path, camera_dir, "time")
        if os.path.exists(time_path):
            # Find all unique camera numbers from timestamp files
            timestamp_files = glob.glob(os.path.join(time_path, "*_*_*.json"))
            cam_numbers = set()
            for file_path in timestamp_files:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if len(parts) >= 3:
                    cam_numbers.add(parts[0])  # First part is camera number
            
            print(f"Found camera numbers for {camera_dir}: {sorted(cam_numbers)}")
            
            for cam_num in sorted(cam_numbers):
                camera_id = f"{camera_dir}_{cam_num}"
                print(f"Initializing {camera_id}")
                try:
                    reader = CameraReader(camera_id, data_path, output_dir)
                    if reader.timeline:  # Only add if timeline was loaded successfully
                        camera_readers.append(reader)
                        cameras.append(camera_id)
                except Exception as e:
                    print(f"Error initializing {camera_id}: {e}")
        else:
            print(f"Time directory not found for {camera_dir}")
    
    if not camera_readers:
        print("No camera readers initialized successfully!")
        return
    
    print("\nStep 2: Creating enhanced master timeline")
    master_timeline, seconds_per_global_unit = create_master_timeline(camera_readers, fps)
    
    if max_frames:
        master_timeline = master_timeline[:max_frames]
        print(f"Processing first {max_frames} frames")
    
    # Convert threshold using correct 1200 global units per second relationship
    threshold_sec = threshold_ms / 1000.0
    threshold_global_units = threshold_sec * RADIO_FREQ
    
    print(f"Using correct conversion: 1200 global units = 1 second")
    print(f"Seconds per global unit: {seconds_per_global_unit:.6f}")
    print(f"Threshold conversion: {threshold_ms} ms = {threshold_global_units:.2f} global units")
    
    # Save master timeline
    with open(os.path.join(mosaic_dir, 'master_timeline_enhanced.json'), 'w') as f:
        json.dump(master_timeline, f, indent=2)
    print(f"Saved enhanced master timeline to {os.path.join(mosaic_dir, 'master_timeline_enhanced.json')}")
    
    # Frame tracking with enhanced metadata
    frame_tracking = {
        "synchronization_info": {
            "method": "comprehensive_analysis",
            "threshold_ms": threshold_ms,
            "threshold_global_units": threshold_global_units,
            "seconds_per_global_unit": seconds_per_global_unit,
            "conversion_method": "radio_frequency",
            "radio_freq": RADIO_FREQ,
            "max_frames": max_frames,
            "fps": fps,
            "rotation": rotation,
            "cameras": cameras,
            "resolution": f"{W}x{H}"
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
    if rotation == 90:
        sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_180)
    elif rotation == 270:
        sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    height, width = sample_frame.shape[:2]
    print(f"Video dimensions after rotation: {width}x{height}")
    
    print("\nStep 4: Creating video writers")
    
    # Try GStreamer first, fallback to OpenCV
    try:
        if GSTREAMER_AVAILABLE:
            mosaic_video = GstVideoWriter(
                os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
                width * 3, height * 3, fps, codec="h265"
            )
        else:
            mosaic_video = OpenCVVideoWriter(
                os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
                width * 3, height * 3, fps
            )
    except Exception as e:
        print(f"GStreamer failed: {e}, using OpenCV fallback")
        mosaic_video = OpenCVVideoWriter(
            os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
            width * 3, height * 3, fps
        )
    
    # Per-camera writers
    camera_writers = {}
    for camera_id in cameras:
        try:
            if GSTREAMER_AVAILABLE:
                writer = GstVideoWriter(
                    os.path.join(per_camera_dir, f"{camera_id}_enhanced_sync.mp4"),
                    width, height, fps, codec="h265"
                )
            else:
                writer = OpenCVVideoWriter(
                    os.path.join(per_camera_dir, f"{camera_id}_enhanced_sync.mp4"),
                    width, height, fps
                )
            camera_writers[camera_id] = writer
        except Exception as e:
            print(f"Failed to create writer for {camera_id}: {e}")
    
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
        
        # Collect frames from all cameras
        for reader in camera_readers:
            frame, frame_info = reader.get_best_frame_for_timestamp(global_timestamp, threshold_global_units)
            
            if frame is not None:
                # Apply rotation
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                output_frames.append(frame)
                stats['synchronized_frames'] += 1
                stats['camera_stats'][reader.camera_id]['actual'] += 1
            else:
                output_frames.append(black_frame)
                stats['black_frames'] += 1
                stats['camera_stats'][reader.camera_id]['black'] += 1
            
            timestamp_frame_info["camera_frames"][reader.camera_id] = frame_info
        
        # Create mosaic (3x3 grid)
        if len(output_frames) >= 9:
            # Create 3x3 mosaic
            mosaic = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
            for i, frame in enumerate(output_frames[:9]):
                row = i // 3
                col = i % 3
                mosaic[row*height:(row+1)*height, col*width:(col+1)*width] = frame
        else:
            # Pad with black frames if less than 9 cameras
            while len(output_frames) < 9:
                output_frames.append(black_frame)
            mosaic = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
            for i, frame in enumerate(output_frames):
                row = i // 3
                col = i % 3
                mosaic[row*height:(row+1)*height, col*width:(col+1)*width] = frame
        
        # Write mosaic frame
        mosaic_video.write(mosaic)
        
        # Write individual camera frames
        for i, (camera_id, writer) in enumerate(camera_writers.items()):
            if i < len(output_frames):
                writer.write(output_frames[i])
        
        # Store frame tracking info
        frame_tracking["frame_sequence"].append(timestamp_frame_info)
        
        frame_count += 1
        stats['total_frames'] += 1
    
    # Close video writers
    mosaic_video.close()
    for writer in camera_writers.values():
        writer.close()
    
    # Clean up camera readers
    for reader in camera_readers:
        reader.cleanup()
    
    print("\nStep 6: Saving results and statistics")
    
    # Calculate final statistics
    total_actual = stats['synchronized_frames']
    total_black = stats['black_frames']
    total_processed = stats['total_frames']
    sync_rate = (total_actual / total_processed * 100) if total_processed > 0 else 0
    
    print("\n=== COMPREHENSIVE SYNCHRONIZATION RESULTS ===")
    print(f"Total frames processed: {total_processed}")
    print(f"Total actual frames used: {total_actual}")
    print(f"Total black frames used: {total_black}")
    print(f"Overall synchronization rate: {sync_rate:.1f}%")
    
    print("\nCamera utilization rates:")
    for camera_id, camera_stats in stats['camera_stats'].items():
        actual = camera_stats['actual']
        rate = (actual / total_processed * 100) if total_processed > 0 else 0
        print(f"  {camera_id}: {rate:.1f}% ({actual}/{total_processed} frames)")
    
    # Save tracking data
    tracking_file = os.path.join(mosaic_dir, 'enhanced_sync_tracking.json')
    with open(tracking_file, 'w') as f:
        json.dump(frame_tracking, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nOutput files:")
    print(f"  Mosaic video: {os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4')}")
    print(f"  Individual videos: {per_camera_dir}/")
    print(f"  Tracking data: {tracking_file}")
    print(f"  Master timeline: {os.path.join(mosaic_dir, 'master_timeline_enhanced.json')}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Video Synchronization for 4K Multi-Camera Setup")
    parser.add_argument("data_path", help="Path to the data directory containing camera folders")
    parser.add_argument("--output-dir", default="synchronized_output", help="Output directory for synchronized videos")
    parser.add_argument("--threshold", type=int, default=100, help="Synchronization threshold in milliseconds")
    parser.add_argument("--fps", type=int, default=20, help="Output video frame rate")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270], help="Rotation angle for output videos")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return
    
    build_synchronized_videos(
        args.data_path,
        args.output_dir,
        threshold_ms=args.threshold,
        max_frames=args.max_frames,
        fps=args.fps,
        rotation=args.rotation
    )


if __name__ == "__main__":
    main() 