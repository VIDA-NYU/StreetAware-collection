#!/usr/bin/env python3
# 8-Camera Parallel CUDA Video Synchronization
# Each camera processes independently in parallel for maximum performance

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from natsort import natsorted
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import threading
from queue import Queue

# CUDA parallelization support
def check_cuda_availability():
    # Check if CUDA is available
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"CUDA GPU detected: {gpu_count} device(s) available")
            return True, gpu_count
        else:
            print("No CUDA GPU devices found")
            return False, 0
    except:
        print("CUDA not available")
        return False, 0

def get_gpu_memory_info():
    # Get GPU memory information
    try:
        gpu_memory = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_memory > 0:
            # Get memory info for first GPU
            device = cv2.cuda.getDevice()
            total_memory = cv2.cuda.getDevice().totalMemory()
            free_memory = cv2.cuda.getDevice().freeMemory()
            used_memory = total_memory - free_memory
            print(f"GPU Memory: {used_memory/1024**3:.1f}GB used, {free_memory/1024**3:.1f}GB free, {total_memory/1024**3:.1f}GB total")
            return total_memory, free_memory
    except:
        pass
    return 0, 0

class ParallelCameraProcessor:
    def __init__(self, camera_id, data_path, gpu_id=0):
        self.camera_id = camera_id
        self.data_path = data_path
        self.gpu_id = gpu_id
        self.ip, self.cam_num = camera_id.split('_')
        
        # CUDA setup
        self.cuda_available, _ = check_cuda_availability()
        if self.cuda_available:
            cv2.cuda.setDevice(gpu_id)
            self.gpu_stream = cv2.cuda.Stream()
        
        # Load timeline
        self.timeline = self.load_timeline()
        self.timeline_index = 0
        
        # Initialize video reader
        self.video_files = self.get_video_files()
        self.current_video_index = 0
        self.current_video = None
        self.frame_buffer = []
        self.buffer_size = 50  # Larger buffer for parallel processing
        
        # Load initial frames
        self.load_frame_batch()
        
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
    
    def load_frame_batch(self):
        # Load multiple frames in batch for parallel processing
        self.frame_buffer = []
        frames_to_load = min(self.buffer_size, len(self.timeline) - self.timeline_index)
        
        for i in range(frames_to_load):
            if self.timeline_index + i >= len(self.timeline):
                break
            
            frame_info = self.timeline[self.timeline_index + i]
            frame_id = frame_info['frame_id']
            frame = self.get_frame_from_video_segments(frame_id)
            
            if frame is not None:
                self.frame_buffer.append({
                    'frame': frame,
                    'timestamp': frame_info['timestamp'],
                    'frame_id': frame_info['frame_id'],
                    'source_file': frame_info['source_file']
                })
    
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
    
    def resize_frame_gpu(self, frame, target_size):
        # GPU-accelerated frame resizing
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame, self.gpu_stream)
            gpu_resized = cv2.cuda.resize(gpu_frame, target_size, stream=self.gpu_stream)
            result = gpu_resized.download(stream=self.gpu_stream)
            self.gpu_stream.waitForCompletion()
            return result
        except:
            # Fallback to CPU
            return cv2.resize(frame, target_size)
    
    def resize_frame_cpu(self, frame, target_size):
        # CPU frame resizing
        return cv2.resize(frame, target_size)
    
    def get_current_frames(self):
        # Get current frames from buffer
        return self.frame_buffer
    
    def advance_frame_batch(self):
        # Advance to next batch of frames
        self.timeline_index += len(self.frame_buffer)
        if self.timeline_index < len(self.timeline):
            self.load_frame_batch()
            return True
        return False
    
    def cleanup(self):
        # Clean up video reader
        if self.current_video is not None:
            self.current_video.release()

def process_camera_parallel(camera_id, data_path, gpu_id, master_timeline, threshold, height, width):
    # Process a single camera in parallel
    processor = ParallelCameraProcessor(camera_id, data_path, gpu_id)
    
    # Set resize function based on CUDA availability
    if processor.cuda_available:
        resize_function = processor.resize_frame_gpu
    else:
        resize_function = processor.resize_frame_cpu
    
    camera_frames = []
    frame_count = 0
    
    for global_timestamp in master_timeline:
        current_frames = processor.get_current_frames()
        
        if not current_frames:
            # No frames available - add black frame
            camera_frames.append(None)  # Will be replaced with black frame later
            continue
        
        # Find best matching frame for this timestamp
        best_frame = None
        best_distance = float('inf')
        
        for frame_info in current_frames:
            distance = abs(global_timestamp - frame_info['timestamp'])
            if distance < best_distance:
                best_distance = distance
                best_frame = frame_info['frame']
        
        if best_frame is not None and best_distance < threshold:
            # Use actual frame
            if best_frame.shape[:2] != (height, width):
                best_frame = resize_function(best_frame, (width, height))
            camera_frames.append(best_frame)
        else:
            # Use black frame
            camera_frames.append(None)  # Will be replaced with black frame later
        
        frame_count += 1
        
        # Advance frame batches periodically
        if frame_count % 50 == 0:
            processor.advance_frame_batch()
    
    processor.cleanup()
    return camera_id, camera_frames

def create_master_timeline_parallel(all_timestamps, fps=30):
    # Create master timeline using parallel approach
    print("Creating master timeline using parallel approach...")
    
    if not all_timestamps:
        print("No timestamps found!")
        return []
    
    # Find start and end times
    start_time = min(all_timestamps)
    end_time = max(all_timestamps)
    
    print(f"Timeline range: {start_time} to {end_time}")
    print(f"Total duration: {end_time - start_time} timestamp units")
    
    # Calculate frame interval
    frame_interval = 1.0 / fps
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

def build_mosaic_video_parallel_cuda(data_path, output_file, threshold=50, max_frames=300, fps=30):
    # Build mosaic video using 8-camera parallel CUDA processing
    
    print("Building Mosaic Video - 8-Camera Parallel CUDA Processing")
    
    # Check CUDA availability
    cuda_available, gpu_count = check_cuda_availability()
    if cuda_available:
        print(f"Using CUDA with {gpu_count} GPU(s)")
    else:
        print("CUDA not available, using CPU parallel processing")
    
    # Find all camera directories dynamically
    camera_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.replace('.', '').isdigit()]
    camera_nums = []
    
    # Find camera numbers by looking at timeline files
    for camera in camera_dirs:
        time_path = os.path.join(data_path, camera, "time")
        if os.path.exists(time_path):
            timeline_files = list(Path(time_path).glob('*.json'))
            for file_path in timeline_files:
                cam_num = file_path.stem.split('_')[0]
                if cam_num not in camera_nums:
                    camera_nums.append(cam_num)
    
    # Camera configurations
    cameras = []
    for camera in camera_dirs:
        for cam_num in camera_nums:
            cameras.append(f"{camera}_{cam_num}")
    
    print(f"Found {len(cameras)} cameras: {cameras}")
    
    # Initialize frame tracking
    frame_tracking = {
        "synchronization_info": {
            "method": "8_camera_parallel_cuda",
            "threshold": threshold,
            "max_frames": max_frames,
            "fps": fps,
            "cameras": cameras,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count
        },
        "frame_sequence": []
    }
    
    # Step 1: Collect all timestamps for master timeline
    print("\nStep 1: Collecting timestamps from all cameras")
    all_timestamps = []
    for camera_id in cameras:
        processor = ParallelCameraProcessor(camera_id, data_path)
        for entry in processor.timeline:
            all_timestamps.append(entry['timestamp'])
        processor.cleanup()
    
    # Step 2: Create master timeline
    print("\nStep 2: Creating master timeline")
    master_timeline = create_master_timeline_parallel(all_timestamps, fps)
    
    if max_frames:
        master_timeline = master_timeline[:max_frames]
        print(f"Processing first {max_frames} frames")
    
    # Save master timeline
    with open('master_timeline_parallel_cuda.json', 'w') as f:
        json.dump(master_timeline, f, indent=2)
    print(f"Saved master timeline to master_timeline_parallel_cuda.json")
    
    # Step 3: Get video dimensions
    print("\nStep 3: Determining video dimensions")
    sample_processor = ParallelCameraProcessor(cameras[0], data_path)
    sample_frame = None
    if sample_processor.frame_buffer:
        sample_frame = sample_processor.frame_buffer[0]['frame']
    sample_processor.cleanup()
    
    if sample_frame is None:
        print("Error: Could not find any sample frame to determine video dimensions")
        return
    
    height, width = sample_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Step 4: Create mosaic video writer
    print("\nStep 4: Creating mosaic video writer")
    
    # Mosaic layout: 3x3 grid
    mosaic_width = width * 3
    mosaic_height = height * 3
    
    # Use H.264 codec
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
    
    # Step 5: Process all cameras in parallel
    print(f"\nStep 5: Processing {len(cameras)} cameras in parallel")
    print(f"Using threshold: {threshold}")
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallel processing
    with ProcessPoolExecutor(max_workers=len(cameras)) as executor:
        # Submit all camera processing tasks
        future_to_camera = {}
        for i, camera_id in enumerate(cameras):
            gpu_id = i % gpu_count if cuda_available and gpu_count > 0 else 0
            future = executor.submit(
                process_camera_parallel, 
                camera_id, 
                data_path, 
                gpu_id, 
                master_timeline, 
                threshold, 
                height, 
                width
            )
            future_to_camera[future] = camera_id
        
        # Collect results as they complete
        camera_results = {}
        for future in as_completed(future_to_camera):
            camera_id = future_to_camera[future]
            try:
                result_camera_id, camera_frames = future.result()
                camera_results[result_camera_id] = camera_frames
                print(f"Completed processing {result_camera_id}")
            except Exception as e:
                print(f"Error processing {camera_id}: {e}")
                camera_results[camera_id] = [None] * len(master_timeline)
    
    parallel_time = time.time() - start_time
    print(f"Parallel processing completed in {parallel_time:.1f} seconds")
    
    # Step 6: Create mosaic video
    print(f"\nStep 6: Creating mosaic video from {len(master_timeline)} frames")
    
    frame_count = 0
    for frame_idx, global_timestamp in enumerate(master_timeline):
        if frame_count % 30 == 0:  # Progress every second
            print(f"Progress: {frame_count}/{len(master_timeline)} ({frame_count/len(master_timeline)*100:.1f}%)")
        
        # Collect frames from all cameras for this timestamp
        camera_frames = []
        timestamp_frame_info = {
            "global_timestamp": global_timestamp,
            "frame_number": frame_count,
            "camera_frames": {}
        }
        
        # Process each camera
        for camera_id in cameras:
            if camera_id in camera_results and frame_idx < len(camera_results[camera_id]):
                frame = camera_results[camera_id][frame_idx]
                if frame is not None:
                    camera_frames.append(frame)
                    timestamp_frame_info["camera_frames"][camera_id] = {
                        "frame_type": "actual_frame",
                        "frame_loaded": True
                    }
                else:
                    camera_frames.append(black_frame.copy())
                    timestamp_frame_info["camera_frames"][camera_id] = {
                        "frame_type": "black_frame",
                        "reason": "no_frame_available"
                    }
            else:
                camera_frames.append(black_frame.copy())
                timestamp_frame_info["camera_frames"][camera_id] = {
                    "frame_type": "black_frame",
                    "reason": "processing_error"
                }
        
        # Add one more black frame to make 9 frames (3x3 grid)
        camera_frames.append(black_frame.copy())
        
        # Add frame tracking info
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
        row1 = np.concatenate((info_frame, camera_frames[0], camera_frames[1]), axis=1)
        row2 = np.concatenate((camera_frames[2], camera_frames[3], camera_frames[4]), axis=1)
        row3 = np.concatenate((camera_frames[5], camera_frames[6], camera_frames[7]), axis=1)
        
        # Stack rows to create mosaic
        mosaic_frame = np.concatenate((row1, row2, row3), axis=0)
        
        # Write to video
        output_video.write(mosaic_frame)
        frame_count += 1
    
    # Step 7: Clean up and save results
    output_video.release()
    
    total_time = time.time() - start_time
    
    # Step 8: Save frame tracking information
    print("\nStep 8: Saving frame tracking information")
    
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
        "camera_statistics": camera_stats,
        "parallel_processing_time": parallel_time,
        "total_processing_time": total_time,
        "cuda_used": cuda_available
    }
    
    # Save tracking file
    tracking_file = output_file.replace('.mp4', '_parallel_cuda_tracking.json')
    with open(tracking_file, 'w') as f:
        json.dump(frame_tracking, f, indent=2)
    
    print(f"Saved frame tracking to: {tracking_file}")
    print(f"Total frames processed: {len(frame_tracking['frame_sequence'])}")
    print(f"Actual frames used: {total_actual_frames}")
    print(f"Black frames used: {total_black_frames}")
    print(f"Utilization rate: {frame_tracking['statistics']['utilization_rate']*100:.1f}%")
    print(f"Parallel processing time: {parallel_time:.1f} seconds")
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"CUDA acceleration: {'Yes' if cuda_available else 'No'}")
    
    # Print camera statistics
    print("\nCamera frame usage:")
    for camera_id, stats in camera_stats.items():
        total = stats["actual_frames"] + stats["black_frames"]
        rate = stats["actual_frames"] / total * 100 if total > 0 else 0
        print(f"  {camera_id}: {stats['actual_frames']}/{total} frames ({rate:.1f}%)")
    
    print(f"\n8-Camera Parallel CUDA Mosaic Video Complete")
    print(f"Output file: {output_file}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count/fps:.1f} seconds")
    print(f"Speedup: {total_time/3600:.1f} hours vs ~30 hours (original)")

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize and create a mosaic video from 8 cameras using parallel CUDA processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  python sync_videos_GPU.py <data_path> [--output OUTPUT] [--threshold THRESHOLD] [--max-frames MAX_FRAMES] [--fps FPS]

OPTIONS:
  data_path           Path to data directory containing camera folders
  --output            Output video file (default: mosaic_parallel_cuda.mp4)
  --threshold         Threshold for frame selection in milliseconds (default: 50)
  --max-frames        Maximum number of frames to process (default: 300)
  --fps               Output video frames per second (default: 30)
        """
    )
    parser.add_argument("data_path", help="Path to data directory containing camera folders")
    parser.add_argument("--output", default="mosaic_parallel_cuda.mp4", help="Output video file")
    parser.add_argument("--threshold", type=int, default=50, help="Threshold for frame selection (milliseconds)")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} not found")
        return
    
    build_mosaic_video_parallel_cuda(args.data_path, args.output, args.threshold, args.max_frames, args.fps)

if __name__ == "__main__":
    main() 