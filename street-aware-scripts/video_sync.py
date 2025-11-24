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
from concurrent.futures import ThreadPoolExecutor

# Video parameters for 4K
W, H, FPS = 3840, 2160, 20  # 4K resolution, 20 FPS
RADIO_FREQ = 1200  # global units per second
DEFAULT_BITRATE = 50000  # kbps for 4K
VERBOSE = False

# Allow OpenCV to leverage all CPU threads when possible
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except AttributeError:
    pass
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
            fps_caps = f"(fraction){int(round(fps * 100_000))}/100000"
        else:
            fps_caps = f"{fps}/1"

        self.input_caps = f"video/x-raw,format={format},width={width},height={height},framerate={fps_caps}"
        # Force conversion to NV12 before feeding NVENC to avoid caps negotiation issues
        self.encoder_caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps_caps}"

        self.command = (
            f"appsrc emit-signals=True is-live=False caps={self.input_caps} ! "
            f"queue ! videoconvert ! {self.encoder_caps} ! "
            f"nv{codec}enc preset=hq bitrate={bitrate} rc-mode={'vbr' if variable else 'cbr'} "
            f"gop-size=45 cuda-device-id={gpu} ! {codec}parse ! matroskamux ! "
            f"filesink location={self.temp_filename}"
        )
        
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
        self.appsrc.set_caps(Gst.Caps.from_string(self.input_caps))

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


def apply_rotation_if_needed(frame, rotation):
    """Rotate frame based on requested output orientation."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def debug_log(message):
    if VERBOSE:
        print(message)


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
        self.video_file_map = self._build_video_file_map(self.video_files)
        self.file_frame_offsets = self._compute_file_frame_offsets()
        self.video_frame_counts = {}
        self.current_video_key = None
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
                        'entry_index': i,
                        'file_index': entry.get('file_index'),
                        'file_template': entry.get('file_template')
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
        if self.timeline_index < len(self.timeline):
            frame_hint = self.timeline[self.timeline_index]
            debug_log(f"{self.camera_id}: load_next_frame at timeline index "
                      f"{self.timeline_index + 1}/{len(self.timeline)} "
                      f"(frame_id={frame_hint.get('frame_id')})")
        while self.timeline_index < len(self.timeline):
            frame_info = self.timeline[self.timeline_index]
            frame = self.get_frame_from_video_segments(frame_info)
            self.timeline_index += 1
            if frame is not None:
                # Update frame buffer (left -> right -> new)
                self.left_frame = self.right_frame
                self.right_frame = frame
                return True
        return False
    
    def get_frame_from_video_segments(self, frame_info):
        """Get a specific frame by frame ID from video segments."""
        if frame_info is None:
            return None

        frame_id = frame_info.get('frame_id')
        if frame_id is None:
            return None
        
        target_file_index = frame_info.get('file_index')
        if target_file_index is not None and target_file_index in self.video_file_map:
            video_file = self.video_file_map[target_file_index]
            video_key = str(video_file)
            video_name = os.path.basename(str(video_file))
            if self.current_video is None or self.current_video_key != video_key:
                if self.current_video is not None:
                    self.current_video.release()
                debug_log(f"{self.camera_id}: Opening {video_name} (index {target_file_index}) for frame lookup")
                self.current_video = cv2.VideoCapture(str(video_file))
                self.current_video_key = video_key
            
            if not self.current_video.isOpened():
                print(f"{self.camera_id}: Failed to open {video_name}")
                return None
            
            local_frame_id = self._compute_local_frame_id(target_file_index, frame_id)
            if local_frame_id is None:
                local_frame_id = 0
            
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, local_frame_id)
            ret, frame = self.current_video.read()
            
            if ret:
                if local_frame_id < 5:
                    debug_log(f"  {self.camera_id}: Global frame {frame_id} -> {video_name} frame {local_frame_id}")
                return frame
            else:
                print(f"  {self.camera_id}: Failed to read frame {local_frame_id} from {video_name}")
                return None
        
        # Fallback to sequential search if metadata is missing
        return self._fallback_frame_lookup(frame_id)

    def _fallback_frame_lookup(self, frame_id):
        """Fallback: sequential scan through video files (slow)."""
        cumulative_frames = 0
        total_videos = len(self.video_files)
        
        for idx, video_file in enumerate(self.video_files):
            video_name = os.path.basename(str(video_file))
            video_key = str(video_file)
            if self.current_video is None or self.current_video_key != video_key:
                if self.current_video is not None:
                    self.current_video.release()
                debug_log(f"{self.camera_id}: Opening {video_name} "
                          f"({idx + 1}/{total_videos}) for frame lookup")
                self.current_video = cv2.VideoCapture(str(video_file))
                self.current_video_key = video_key
            
            if not self.current_video.isOpened():
                print(f"{self.camera_id}: Failed to open {video_name}")
                continue
            
            if video_key not in self.video_frame_counts:
                debug_log(f"{self.camera_id}: Counting frames in {video_name}...")
                total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_frame_counts[video_key] = total_frames
                debug_log(f"{self.camera_id}: {video_name} has {total_frames} frames")
            else:
                total_frames = self.video_frame_counts[video_key]
            
            if total_frames <= 0:
                continue
            
            if frame_id >= cumulative_frames and frame_id < cumulative_frames + total_frames:
                local_frame_id = frame_id - cumulative_frames
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, local_frame_id)
                ret, frame = self.current_video.read()
                
                if ret:
                    if local_frame_id < 5:
                        debug_log(f"  {self.camera_id}: Global frame {frame_id} -> {video_name} frame {local_frame_id}")
                    return frame
                else:
                    print(f"  {self.camera_id}: Failed to read frame {local_frame_id} from {video_name}")
                    return None
            
            cumulative_frames += total_frames
        
        return None

    def _build_video_file_map(self, video_files):
        """Map file indices derived from filenames to actual paths."""
        video_map = {}
        for video_file in video_files:
            name = os.path.basename(str(video_file))
            parts = os.path.splitext(name)[0].split('_')
            if parts:
                try:
                    idx = int(parts[-1])
                    video_map[idx] = video_file
                except ValueError:
                    continue
        return video_map
    
    def _compute_file_frame_offsets(self):
        """Compute the first global frame id for each file index."""
        offsets = {}
        for entry in self.timeline:
            file_idx = entry.get('file_index')
            frame_id = entry.get('frame_id')
            if file_idx is None or frame_id is None:
                continue
            if file_idx not in offsets:
                offsets[file_idx] = frame_id
        return offsets
    
    def _compute_local_frame_id(self, file_index, frame_id):
        """Convert global frame id to local frame number inside the file."""
        offset = self.file_frame_offsets.get(file_index)
        if offset is None:
            return None
        local_frame_id = frame_id - offset
        return local_frame_id if local_frame_id >= 0 else 0
    
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


def build_synchronized_videos(data_path, output_dir, threshold_ms=100, max_frames=None, fps=20, rotation=0,
                              workers=None, nvenc_camera_limit=2):
    """Build synchronized videos using the comprehensive approach."""
    print("Building Synchronized Videos with Comprehensive Analysis")
    
    # Determine worker usage
    if workers is None:
        workers = min(max(1, (os.cpu_count() or 4)), 32)
    else:
        workers = max(1, workers)
    use_thread_pool = workers > 1
    print(f"Using up to {workers} worker thread(s) for parallel stages")
    executor = ThreadPoolExecutor(max_workers=workers) if use_thread_pool else None
    
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
        if executor:
            executor.shutdown(wait=True)
        return
    
    # Create camera readers
    print("\nStep 1: Initializing camera readers with enhanced timeline analysis")
    camera_readers = []
    cameras = []
    camera_ids = []
    
    for camera_dir in camera_dirs:
        time_path = os.path.join(data_path, camera_dir, "time")
        if os.path.exists(time_path):
            timestamp_files = glob.glob(os.path.join(time_path, "*_*_*.json"))
            cam_numbers = set()
            for file_path in timestamp_files:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if len(parts) >= 3:
                    cam_numbers.add(parts[0])
            
            sorted_nums = sorted(cam_numbers)
            if sorted_nums:
                print(f"Found camera numbers for {camera_dir}: {sorted_nums}")
            for cam_num in sorted_nums:
                camera_id = f"{camera_dir}_{cam_num}"
                camera_ids.append(camera_id)
        else:
            print(f"Time directory not found for {camera_dir}")
    
    def _init_reader(camera_id):
        print(f"Initializing {camera_id}")
        try:
            reader = CameraReader(camera_id, data_path, output_dir)
            return reader
        except Exception as e:
            print(f"Error initializing {camera_id}: {e}")
        return None
    
    if executor:
        future_pairs = []
        for camera_id in camera_ids:
            future_pairs.append((camera_id, executor.submit(_init_reader, camera_id)))
        for camera_id, future in future_pairs:
            reader = future.result()
            if reader and reader.timeline:
                camera_readers.append(reader)
                cameras.append(reader.camera_id)
    else:
        for camera_id in camera_ids:
            reader = _init_reader(camera_id)
            if reader and reader.timeline:
                camera_readers.append(reader)
                cameras.append(reader.camera_id)
    
    if not camera_readers:
        print("No camera readers initialized successfully!")
        if executor:
            executor.shutdown(wait=True)
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
        if executor:
            executor.shutdown(wait=True)
        return
    
    # Apply rotation to sample frame to get correct output size
    sample_frame = apply_rotation_if_needed(sample_frame, rotation)
    
    height, width = sample_frame.shape[:2]
    print(f"Video dimensions after rotation: {width}x{height}")

    # Mosaic tiles target resolution (2K per camera tile) else nvenc will fail
    tile_target_width = 2048
    tile_target_height = 1152
    tile_size = (tile_target_width, tile_target_height)
    print(f"Mosaic tiles will be resized to {tile_target_width}x{tile_target_height}")
    mosaic_width = tile_target_width * 3
    mosaic_height = tile_target_height * 3
    
    print("\nStep 4: Creating video writers")
    
    # Try GStreamer first, fallback to OpenCV
    try:
        if GSTREAMER_AVAILABLE:
            mosaic_video = GstVideoWriter(
                os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
                mosaic_width, mosaic_height, fps, codec="h265"
            )
        else:
            mosaic_video = OpenCVVideoWriter(
                os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
                mosaic_width, mosaic_height, fps
            )
    except Exception as e:
        print(f"GStreamer failed: {e}, using OpenCV fallback")
        mosaic_video = OpenCVVideoWriter(
            os.path.join(mosaic_dir, 'mosaic_enhanced_sync.mp4'),
            mosaic_width, mosaic_height, fps
        )
    
    camera_writers = {}
    nvenc_camera_ids = set()
    if GSTREAMER_AVAILABLE:
        if nvenc_camera_limit is None:
            nvenc_camera_limit = 2
        nvenc_camera_ids = set(cameras[:max(0, nvenc_camera_limit)])
        print(f"Using NVENC for {len(nvenc_camera_ids)} per-camera streams: {sorted(nvenc_camera_ids)}")
    else:
        nvenc_camera_limit = 0
    
    for camera_id in cameras:
        try:
            output_path = os.path.join(per_camera_dir, f"{camera_id}_enhanced_sync.mp4")
            if camera_id in nvenc_camera_ids:
                writer = GstVideoWriter(output_path, width, height, fps, codec="h265")
            else:
                writer = OpenCVVideoWriter(output_path, width, height, fps)
            camera_writers[camera_id] = writer
        except Exception as e:
            print(f"Failed to create writer for {camera_id}: {e}")
    
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    black_tile = np.zeros((tile_target_height, tile_target_width, 3), dtype=np.uint8)
    camera_positions = {reader.camera_id: idx for idx, reader in enumerate(camera_readers)}
    
    def fetch_frame_payload(reader, target_timestamp):
        frame, frame_info = reader.get_best_frame_for_timestamp(target_timestamp, threshold_global_units)
        is_actual = frame is not None
        if is_actual:
            frame = apply_rotation_if_needed(frame, rotation)
        else:
            frame = black_frame
        return reader.camera_id, frame, frame_info, is_actual
    
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
        if executor:
            advance_futures = [
                executor.submit(reader.advance_to_timestamp, global_timestamp, threshold_global_units)
                for reader in camera_readers
            ]
            for future in advance_futures:
                future.result()
        else:
            for reader in camera_readers:
                reader.advance_to_timestamp(global_timestamp, threshold_global_units)
        
        if executor:
            frame_futures = [
                executor.submit(fetch_frame_payload, reader, global_timestamp)
                for reader in camera_readers
            ]
            frame_results = [future.result() for future in frame_futures]
        else:
            frame_results = [fetch_frame_payload(reader, global_timestamp) for reader in camera_readers]
        
        ordered_results = [None] * len(camera_readers)
        for cam_id, frame, frame_info, is_actual in frame_results:
            idx = camera_positions.get(cam_id)
            if idx is not None:
                ordered_results[idx] = (frame, frame_info, is_actual)
        
        per_camera_frames = []
        mosaic_frames = []
        timestamp_frame_info = {
            "global_timestamp": global_timestamp,
            "frame_number": frame_count,
            "timestamp_seconds": global_timestamp * seconds_per_global_unit,
            "camera_frames": {}
        }
        
        # Collect frames from all cameras
        for idx, camera_id in enumerate(cameras):
            frame_data = ordered_results[idx]
            if frame_data is None:
                frame = black_frame
                frame_info = {
                    'frame_type': 'black_frame',
                    'reason': 'missing_frame_data'
                }
                is_actual = False
            else:
                frame, frame_info, is_actual = frame_data
            
            per_camera_frames.append(frame)
            if is_actual:
                if frame.shape[1] != tile_target_width or frame.shape[0] != tile_target_height:
                    mosaic_tile = cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA)
                else:
                    mosaic_tile = frame
            else:
                mosaic_tile = black_tile
            mosaic_frames.append(mosaic_tile)
            if is_actual:
                stats['synchronized_frames'] += 1
                stats['camera_stats'][camera_id]['actual'] += 1
            else:
                stats['black_frames'] += 1
                stats['camera_stats'][camera_id]['black'] += 1
            
            timestamp_frame_info["camera_frames"][camera_id] = frame_info
        
        # Create mosaic (3x3 grid)
        if len(mosaic_frames) >= 9:
            # Create 3x3 mosaic
            mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
            for i, frame in enumerate(mosaic_frames[:9]):
                row = i // 3
                col = i % 3
                mosaic[row*tile_target_height:(row+1)*tile_target_height,
                       col*tile_target_width:(col+1)*tile_target_width] = frame
        else:
            # Pad with black frames if less than 9 cameras
            while len(mosaic_frames) < 9:
                mosaic_frames.append(black_tile)
            mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
            for i, frame in enumerate(mosaic_frames):
                row = i // 3
                col = i % 3
                mosaic[row*tile_target_height:(row+1)*tile_target_height,
                       col*tile_target_width:(col+1)*tile_target_width] = frame
        
        # Write mosaic frame
        mosaic_video.write(mosaic)
        
        # Write individual camera frames
        for idx, camera_id in enumerate(cameras):
            frame = per_camera_frames[idx]
            writer = camera_writers.get(camera_id)
            if writer:
                writer.write(frame)
        
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
    
    if executor:
        executor.shutdown(wait=True)
    
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
    parser.add_argument("--workers", type=int, help="Number of worker threads to parallelize decoding/selection")
    parser.add_argument("--nvenc-camera-limit", type=int, default=2,
                        help="Number of per-camera videos that should use NVENC (requires GStreamer)")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return
    
    global VERBOSE
    VERBOSE = args.verbose

    build_synchronized_videos(
        args.data_path,
        args.output_dir,
        threshold_ms=args.threshold,
        max_frames=args.max_frames,
        fps=args.fps,
        rotation=args.rotation,
        workers=args.workers,
        nvenc_camera_limit=args.nvenc_camera_limit
    )


if __name__ == "__main__":
    main() 
