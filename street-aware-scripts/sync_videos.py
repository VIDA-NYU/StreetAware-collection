#!/usr/bin/env python3
# Mosaic Video Synchronization using Previous Work's Logic
# Two-frame buffer per camera, master timeline with regular intervals, single mosaic output

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from natsort import natsorted

# GPU support with graceful fallback
def check_gpu_availability():
    # Check if CUDA is available
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
    # Return appropriate processing device
    if check_gpu_availability():
        return "gpu"
    else:
        return "cpu"

def resize_frame_gpu(frame, target_size):
    # GPU-accelerated frame resizing
    try:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
        return gpu_resized.download()
    except:
        # Fallback to CPU
        return cv2.resize(frame, target_size)

def resize_frame_cpu(frame, target_size):
    # CPU frame resizing
    return cv2.resize(frame, target_size)

class CameraReader:
    def __init__(self, camera_id, data_path):
        self.camera_id = camera_id
        self.data_path = data_path
        self.ip, self.cam_num = camera_id.split('_')
        
        # Load timeline
        self.timeline = self.load_timeline()
        self.timeline_index = 0
        
        # Initialize video reader
        self.video_files = self.get_video_files()
        self.current_video_index = 0
        self.current_video = None
        self.left_frame = None
        self.right_frame = None
        
        # Load first two frames
        self.load_next_frame()
        self.load_next_frame()
        
        print(f"Initialized {camera_id}: {len(self.timeline)} timestamps, {len(self.video_files)} video files")
    
    def load_timeline(self):
        # Load timeline for this camera
        time_path = os.path.join(self.data_path, self.ip, "time")
        
        if not os.path.exists(time_path):
            print(f"Time path not found: {time_path}")
            return []
        
        # Load all timestamp files for this camera
        pattern = f"{self.cam_num}_*.json"
        timestamp_files = list(Path(time_path).glob(pattern))
        timestamp_files = natsorted(timestamp_files)
        
        all_timestamps = []
        for file_path in timestamp_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Handle the buffer format (buffer_0, buffer_1, etc.)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key.startswith('buffer_') and isinstance(value, dict):
                                if 'global_timestamp' in value and 'frame_id' in value:
                                    timestamp_info = {
                                        'timestamp': value['global_timestamp'],
                                        'frame_id': value['frame_id'],
                                        'source_file': str(file_path)
                                    }
                                    all_timestamps.append(timestamp_info)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Sort by timestamp
        all_timestamps.sort(key=lambda x: x['timestamp'])
        return all_timestamps
    
    def get_video_files(self):
        # Get list of video files for this camera
        video_path = os.path.join(self.data_path, self.ip, "video")
        
        if not os.path.exists(video_path):
            return []
        
        pattern = f"{self.cam_num}_*.avi"
        video_files = list(Path(video_path).glob(pattern))
        return natsorted(video_files)
    
    def load_next_frame(self):
        # Load the next frame from timeline
        if self.timeline_index >= len(self.timeline):
            return False
        
        # Get frame info
        frame_info = self.timeline[self.timeline_index]
        frame_id = frame_info['frame_id']
        
        # Find the video file containing this frame
        frame = self.get_frame_from_video_segments(frame_id)
        
        if frame is not None:
            # Update frame buffer
            self.left_frame = self.right_frame
            self.right_frame = frame
            self.timeline_index += 1
            return True
        else:
            # Skip this frame and try next
            self.timeline_index += 1
            return self.load_next_frame()
    
    def get_frame_from_video_segments(self, frame_id):
        # Get a specific frame from video segments
        for video_file in self.video_files:
            if self.current_video is None or self.current_video_index != self.video_files.index(video_file):
                if self.current_video is not None:
                    self.current_video.release()
                self.current_video = cv2.VideoCapture(str(video_file))
                self.current_video_index = self.video_files.index(video_file)
            
            if not self.current_video.isOpened():
                continue
            
            total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_id < total_frames:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = self.current_video.read()
                
                if ret:
                    return frame
            else:
                continue
        
        return None
    
    def get_current_frame_timestamps(self):
        # Get timestamps for the currently loaded frames
        if self.timeline_index >= len(self.timeline):
            return None, None
        
        # Get timestamps for the currently loaded frames
        left_timestamp = self.timeline[self.timeline_index - 2]['timestamp'] if self.timeline_index > 1 else None
        right_timestamp = self.timeline[self.timeline_index - 1]['timestamp'] if self.timeline_index > 0 else None
        
        return left_timestamp, right_timestamp
    
    def get_current_frames(self):
        # Get current left and right frames
        return self.left_frame, self.right_frame
    
    def advance_frame(self):
        # Advance to next frame
        return self.load_next_frame()
    
    def cleanup(self):
        # Clean up video reader
        if self.current_video is not None:
            self.current_video.release()

def create_master_timeline_previous_style(all_camera_readers, fps=30):
    # Create master timeline using previous work's approach
    print("Creating master timeline using previous work's approach...")
    
    # Collect all timestamps from all cameras
    all_timestamps = []
    for reader in all_camera_readers:
        for entry in reader.timeline:
            all_timestamps.append(entry['timestamp'])
    
    if not all_timestamps:
        print("No timestamps found!")
        return []
    
    # Find start and end times
    start_time = min(all_timestamps)
    end_time = max(all_timestamps)
    
    print(f"Timeline range: {start_time} to {end_time}")
    print(f"Total duration: {end_time - start_time} timestamp units")
    
    # Calculate frame interval (previous work uses regular intervals)
    frame_interval = 1.0 / fps  # For 30 FPS, interval = 0.033 seconds
    
    # Convert to timestamp units (assuming timestamps are in milliseconds)
    # If timestamps are in different units, adjust this calculation
    timestamp_interval = int(frame_interval * 1000)  # Convert to milliseconds
    
    # Create master timeline with regular intervals
    master_timeline = []
    current_time = start_time
    while current_time <= end_time:
        master_timeline.append(current_time)
        current_time += timestamp_interval
    
    print(f"Created master timeline with {len(master_timeline)} points")
    print(f"Frame interval: {timestamp_interval} timestamp units")
    
    return master_timeline

def build_mosaic_video_previous_logic(data_path, output_file, threshold=50, max_frames=300, fps=30):
    # Build mosaic video using previous work's logic
    
    print("Building Mosaic Video - Previous Work Logic")
    
    # Determine processing device
    processing_device = get_processing_device()
    print(f"Using {processing_device.upper()} for processing")
    
    # Set resize function based on device
    if processing_device == "gpu":
        resize_function = resize_frame_gpu
    else:
        resize_function = resize_frame_cpu
    
    # Camera configurations
    cameras = [
        '192.168.0.108_0', '192.168.0.108_2',
        '192.168.0.122_0', '192.168.0.122_2',
        '192.168.0.184_0', '192.168.0.184_2',
        '192.168.0.227_0', '192.168.0.227_2'
    ]
    
    # Initialize frame tracking
    frame_tracking = {
        "synchronization_info": {
            "method": "previous_work_mosaic_logic",
            "threshold": threshold,
            "max_frames": max_frames,
            "fps": fps,
            "cameras": cameras
        },
        "frame_sequence": []
    }
    
    # Step 1: Initialize camera readers
    print("\nStep 1: Initializing camera readers")
    camera_readers = []
    
    for camera_id in cameras:
        print(f"\nInitializing {camera_id}")
        reader = CameraReader(camera_id, data_path)
        camera_readers.append(reader)
    
    # Step 2: Create master timeline using previous work's approach
    print("\nStep 2: Creating master timeline")
    master_timeline = create_master_timeline_previous_style(camera_readers, fps)
    
    if max_frames:
        master_timeline = master_timeline[:max_frames]
        print(f"Processing first {max_frames} frames")
    
    # Save master timeline
    with open('master_timeline_mosaic.json', 'w') as f:
        json.dump(master_timeline, f, indent=2)
    print(f"Saved master timeline to master_timeline_mosaic.json")
    
    # Step 3: Get video dimensions from first available frame
    print("\nStep 3: Determining video dimensions")
    sample_frame = None
    for reader in camera_readers:
        if reader.right_frame is not None:
            sample_frame = reader.right_frame
            break
    
    if sample_frame is None:
        print("Error: Could not find any sample frame to determine video dimensions")
        return
    
    height, width = sample_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Step 4: Create mosaic video writer
    print("\nStep 4: Creating mosaic video writer")
    
    # Mosaic layout: 3x3 grid (9 cameras, but we have 8, so one empty slot)
    mosaic_width = width * 3
    mosaic_height = height * 3
    
    # Use H.264 codec which can handle large dimensions
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (mosaic_width, mosaic_height))
    
    if not output_video.isOpened():
        # Fallback to MJPG if H.264 fails
        print("H.264 failed, trying MJPG codec...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_video = cv2.VideoWriter(output_file, fourcc, fps, (mosaic_width, mosaic_height))
        
        if not output_video.isOpened():
            print("Error: Could not create video writer")
            return
    
    # Create black frame for empty slots
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Step 5: Process each global timestamp using previous work's logic
    print(f"\nStep 5: Processing {len(master_timeline)} global timestamps")
    print(f"Using threshold: {threshold}")
    
    frame_count = 0
    for global_timestamp in master_timeline:
        if frame_count % 30 == 0:  # Progress every second
            print(f"Progress: {frame_count}/{len(master_timeline)} ({frame_count/len(master_timeline)*100:.1f}%)")
        
        # Array to store all camera frames for this timestamp
        output_array = []
        
        # Track frame usage for this timestamp
        timestamp_frame_info = {
            "global_timestamp": global_timestamp,
            "frame_number": frame_count,
            "camera_frames": {}
        }
        
        # For each camera, use previous work's logic
        for i, reader in enumerate(camera_readers):
            # Get current timestamps and frames
            left_timestamp, right_timestamp = reader.get_current_frame_timestamps()
            left_frame, right_frame = reader.get_current_frames()
            
            if left_timestamp is None or right_timestamp is None:
                # No frames available - add black frame
                output_array.append(black_frame.copy())
                timestamp_frame_info["camera_frames"][reader.camera_id] = {
                    "frame_type": "black_frame",
                    "reason": "no_timeline_data",
                    "left_timestamp": left_timestamp,
                    "right_timestamp": right_timestamp
                }
                continue
            
            # Previous work's logic: compare left vs right timestamp
            if abs(global_timestamp - left_timestamp) < abs(global_timestamp - right_timestamp):
                # Left frame is closer
                if abs(global_timestamp - left_timestamp) < threshold:
                    # Use left frame
                    if left_frame is not None:
                        # Resize if necessary
                        if left_frame.shape[:2] != (height, width):
                            left_frame = resize_function(left_frame, (width, height))
                        output_array.append(left_frame.copy())
                        
                        # Track frame usage
                        timestamp_frame_info["camera_frames"][reader.camera_id] = {
                            "frame_type": "actual_frame",
                            "source_timestamp": left_timestamp,
                            "frame_id": reader.timeline[reader.timeline_index - 1]['frame_id'],
                            "source_file": reader.timeline[reader.timeline_index - 1]['source_file'],
                            "distance": abs(global_timestamp - left_timestamp),
                            "frame_loaded": True
                        }
                    else:
                        output_array.append(black_frame.copy())
                        timestamp_frame_info["camera_frames"][reader.camera_id] = {
                            "frame_type": "black_frame",
                            "reason": "frame_loading_failed",
                            "source_timestamp": left_timestamp,
                            "frame_id": reader.timeline[reader.timeline_index - 1]['frame_id'],
                            "frame_loaded": False
                        }
                else:
                    # Use black frame
                    output_array.append(black_frame.copy())
                    timestamp_frame_info["camera_frames"][reader.camera_id] = {
                        "frame_type": "black_frame",
                        "reason": "outside_threshold",
                        "source_timestamp": left_timestamp,
                        "distance": abs(global_timestamp - left_timestamp),
                        "threshold": threshold
                    }
            else:
                # Right frame is closer
                if abs(global_timestamp - right_timestamp) < threshold:
                    # Use right frame
                    if right_frame is not None:
                        # Resize if necessary
                        if right_frame.shape[:2] != (height, width):
                            right_frame = resize_function(right_frame, (width, height))
                        output_array.append(right_frame.copy())
                        
                        # Track frame usage
                        timestamp_frame_info["camera_frames"][reader.camera_id] = {
                            "frame_type": "actual_frame",
                            "source_timestamp": right_timestamp,
                            "frame_id": reader.timeline[reader.timeline_index]['frame_id'],
                            "source_file": reader.timeline[reader.timeline_index]['source_file'],
                            "distance": abs(global_timestamp - right_timestamp),
                            "frame_loaded": True
                        }
                        
                        # Advance frame (previous work's logic)
                        reader.advance_frame()
                    else:
                        output_array.append(black_frame.copy())
                        timestamp_frame_info["camera_frames"][reader.camera_id] = {
                            "frame_type": "black_frame",
                            "reason": "frame_loading_failed",
                            "source_timestamp": right_timestamp,
                            "frame_id": reader.timeline[reader.timeline_index]['frame_id'],
                            "frame_loaded": False
                        }
                else:
                    # Use black frame
                    output_array.append(black_frame.copy())
                    timestamp_frame_info["camera_frames"][reader.camera_id] = {
                        "frame_type": "black_frame",
                        "reason": "outside_threshold",
                        "source_timestamp": right_timestamp,
                        "distance": abs(global_timestamp - right_timestamp),
                        "threshold": threshold
                    }
        
        # Add one more black frame to make 9 frames (3x3 grid)
        output_array.append(black_frame.copy())
        
        # Add frame tracking info to the sequence
        frame_tracking["frame_sequence"].append(timestamp_frame_info)
        
        # Create mosaic layout (3x3 grid)
        # Row 1: [Info] [Camera 1] [Camera 2]
        # Row 2: [Camera 3] [Camera 4] [Camera 5]
        # Row 3: [Camera 6] [Camera 7] [Camera 8]
        
        # Create info frame
        info_frame = black_frame.copy()
        cv2.putText(info_frame, f'Timestamp: {global_timestamp}', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(info_frame, f'Frame: {frame_count}', (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create rows
        row1 = np.concatenate((info_frame, output_array[0], output_array[1]), axis=1)
        row2 = np.concatenate((output_array[2], output_array[3], output_array[4]), axis=1)
        row3 = np.concatenate((output_array[5], output_array[6], output_array[7]), axis=1)
        
        # Stack rows to create mosaic
        mosaic_frame = np.concatenate((row1, row2, row3), axis=0)
        
        # Write to video
        output_video.write(mosaic_frame)
        frame_count += 1
    
    # Step 6: Clean up
    output_video.release()
    for reader in camera_readers:
        reader.cleanup()
    
    # Step 6: Save frame tracking information
    print("\nStep 6: Saving frame tracking information")
    
    # Calculate statistics
    total_actual_frames = 0
    total_black_frames = 0
    camera_stats = {}
    
    for camera_id in cameras:
        camera_stats[camera_id] = {"actual_frames": 0, "black_frames": 0}
    
    for frame_info in frame_tracking["frame_sequence"]:
        for camera_id, camera_frame_info in frame_info["camera_frames"].items():
            if camera_frame_info["frame_type"] == "actual_frame":
                total_actual_frames += 1
                camera_stats[camera_id]["actual_frames"] += 1
            else:
                total_black_frames += 1
                camera_stats[camera_id]["black_frames"] += 1
    
    # Add statistics to tracking data
    frame_tracking["statistics"] = {
        "total_frames": len(frame_tracking["frame_sequence"]),
        "total_actual_frames": total_actual_frames,
        "total_black_frames": total_black_frames,
        "utilization_rate": total_actual_frames / (total_actual_frames + total_black_frames) if (total_actual_frames + total_black_frames) > 0 else 0,
        "camera_statistics": camera_stats
    }
    
    # Save tracking file
    tracking_file = output_file.replace('.mp4', '_frame_tracking.json')
    with open(tracking_file, 'w') as f:
        json.dump(frame_tracking, f, indent=2)
    
    print(f"Saved frame tracking to: {tracking_file}")
    print(f"Total frames processed: {len(frame_tracking['frame_sequence'])}")
    print(f"Actual frames used: {total_actual_frames}")
    print(f"Black frames used: {total_black_frames}")
    print(f"Utilization rate: {frame_tracking['statistics']['utilization_rate']*100:.1f}%")
    
    # Print camera statistics
    print("\nCamera frame usage:")
    for camera_id, stats in camera_stats.items():
        total = stats["actual_frames"] + stats["black_frames"]
        rate = stats["actual_frames"] / total * 100 if total > 0 else 0
        print(f"  {camera_id}: {stats['actual_frames']}/{total} frames ({rate:.1f}%)")
    
    print("\nMosaic Video Complete")
    print(f"Output file: {output_file}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize and create a mosaic video from multiple cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  python sync_videos.py <data_path> [--output OUTPUT] [--threshold THRESHOLD] [--max-frames MAX_FRAMES] [--fps FPS]

OPTIONS:
  data_path           Path to data directory containing camera folders
  --output            Output video file (default: mosaic_300_frames.mp4)
  --threshold         Threshold for frame selection in milliseconds (default: 50)
  --max-frames        Maximum number of frames to process (default: 300)
  --fps               Output video frames per second (default: 30)
        """
    )
    parser.add_argument("data_path", help="Path to data directory containing camera folders")
    parser.add_argument("--output", default="mosaic_300_frames.mp4", help="Output video file")
    parser.add_argument("--threshold", type=int, default=50, help="Threshold for frame selection (milliseconds)")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} not found")
        return
    
    build_mosaic_video_previous_logic(args.data_path, args.output, args.threshold, args.max_frames, args.fps)

if __name__ == "__main__":
    main() 