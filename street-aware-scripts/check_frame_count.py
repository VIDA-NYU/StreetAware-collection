# Analyze available frames and suggest optimal frame counts for sync_videos.py

import json
import os
from pathlib import Path
from natsort import natsorted

# GPU support with graceful fallback
def check_gpu_availability():
    # Check if CUDA is available
    try:
        import cv2
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

def analyze_camera_data(data_path):
    # Analyze timeline data for all cameras
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
    results = {}
    total_frames = 0
    
    print("Camera Timeline Analysis")
    print("-" * 40)
    
    for camera in camera_dirs:
        for cam_num in camera_nums:
            camera_id = f"{camera}_{cam_num}"
            time_path = os.path.join(data_path, camera, "time")
            
            if not os.path.exists(time_path):
                print(f"{camera_id}: No timeline data found")
                continue
            
            timeline_files = list(Path(time_path).glob(f'{cam_num}_*.json'))
            timeline_files = natsorted(timeline_files)
            
            if not timeline_files:
                print(f"{camera_id}: No timeline files found")
                continue
            
            # Count total frames across all timeline files
            frame_count = 0
            timestamps = []
            
            for file_path in timeline_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        buffer_count = len(data)
                        frame_count += buffer_count
                        
                        # Collect timestamps for range analysis
                        for key, value in data.items():
                            if key.startswith('buffer_') and isinstance(value, dict):
                                if 'global_timestamp' in value:
                                    timestamps.append(value['global_timestamp'])
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            
            if timestamps:
                min_ts = min(timestamps)
                max_ts = max(timestamps)
                duration_ms = max_ts - min_ts
                duration_sec = duration_ms / 1000  # Convert milliseconds to seconds
                
                results[camera_id] = {
                    'timeline_files': len(timeline_files),
                    'total_frames': frame_count,
                    'min_timestamp': min_ts,
                    'max_timestamp': max_ts,
                    'duration_sec': duration_sec,
                    'avg_fps': frame_count / duration_sec if duration_sec > 0 else 0
                }
                
                total_frames += frame_count
                
                print(f"{camera_id}:")
                print(f"  Timeline files: {len(timeline_files)}")
                print(f"  Total frames: {frame_count}")
                print(f"  Duration: {duration_sec:.1f} seconds")
                print(f"  Avg FPS: {results[camera_id]['avg_fps']:.1f}")
                print(f"  Timestamp range: {min_ts} - {max_ts}")
                print()
            else:
                print(f"{camera_id}: No valid timestamp data")
                print()
    
    return results, total_frames

def suggest_frame_counts(total_frames, duration_sec):
    # Suggest optimal frame counts for different use cases
    print("Frame Count Recommendations")
    print("-" * 40)
    
    # Common frame counts for different durations
    suggestions = [
        (total_frames, f"Full duration ({duration_sec:.1f} seconds)")
    ]
    
    print("Suggested --max-frames values:")
    for frames, description in suggestions:
        if frames <= total_frames:
            print(f"  {frames}: {description}")
    
    print(f"\nMaximum available: {total_frames} frames")
    print(f"Full duration: {duration_sec:.1f} seconds")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze available frames and timeline data for video synchronization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  python check_frame_count.py <data_path>

OPTIONS:
  data_path    Path to data directory containing camera folders
        """
    )
    parser.add_argument("data_path", help="Path to data directory containing camera folders")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} not found")
        return
    
    # Determine processing device
    processing_device = get_processing_device()
    print(f"Using {processing_device.upper()} for analysis")
    
    results, total_frames = analyze_camera_data(args.data_path)
    
    if not results:
        print("No valid camera data found")
        return
    
    # Calculate overall duration
    all_timestamps = []
    for camera_data in results.values():
        all_timestamps.extend([camera_data['min_timestamp'], camera_data['max_timestamp']])
    
    if all_timestamps:
        overall_min = min(all_timestamps)
        overall_max = max(all_timestamps)
        overall_duration = (overall_max - overall_min) / 1000  # Convert milliseconds to seconds
        
        print("Overall Analysis")
        print(f"Total frames across all cameras: {total_frames}")
        print(f"Overall timestamp range: {overall_min} - {overall_max}")
        print(f"Overall duration: {overall_duration:.1f} seconds")
        print()
        
        # Calculate average FPS across all cameras
        total_avg_fps = sum(camera_data['avg_fps'] for camera_data in results.values()) / len(results)
        avg_seconds_per_frame = 1.0 / total_avg_fps
        print(f"\nAverage FPS: {total_avg_fps:.1f}")
        print(f"Average seconds per frame: {avg_seconds_per_frame:.3f}")
        print(f"Total frames: {total_frames}")
        print(f"Full duration: {overall_duration:.1f} seconds")

if __name__ == "__main__":
    main() 