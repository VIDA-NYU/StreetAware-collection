#!/usr/bin/env python3
# Comprehensive AVI header fixer for 30fps videos
# Fixes corrupted headers and replaces original files with corrected versions

import os
import subprocess
import json
from pathlib import Path
from natsort import natsorted
import argparse
import shutil
import time

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

def check_ffmpeg():
    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def test_video_header(video_path):
    # Test if video header is correct
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return False, "Could not open video"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try to read first frame
    ret, frame = cap.read()
    first_frame_ok = ret
    
    # Try to read last frame
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        last_frame_ok = ret
    else:
        last_frame_ok = False
    
    cap.release()
    
    header_ok = first_frame_ok and last_frame_ok
    info = f"{width}x{height}, {fps}fps, {frame_count} frames, first_ok={first_frame_ok}, last_ok={last_frame_ok}"
    
    return header_ok, info

def method_1_reencode_30fps(input_path, output_path):
    # Method 1: Re-encode with 30fps and high quality
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'mjpeg',
        '-q:v', '1',
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        '-f', 'avi',
        '-fflags', '+genpts',
        '-avoid_negative_ts', 'make_zero',
        '-err_detect', 'ignore_err',
        '-max_muxing_queue_size', '1024',
        '-y',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def method_2_copy_30fps(input_path, output_path):
    # Method 2: Copy streams with 30fps
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-r', '30',
        '-f', 'avi',
        '-fflags', '+genpts',
        '-avoid_negative_ts', 'make_zero',
        '-err_detect', 'ignore_err',
        '-y',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def method_3_mp4_bridge(input_path, output_path):
    # Method 3: Convert to MP4 then back to AVI
    temp_path = str(output_path).replace('.avi', '_temp.mp4')
    
    # Step 1: Convert to MP4
    cmd1 = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-r', '30',
        '-fflags', '+genpts',
        '-avoid_negative_ts', 'make_zero',
        '-err_detect', 'ignore_err',
        '-y',
        temp_path
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        return False
    
    # Step 2: Convert back to AVI
    cmd2 = [
        'ffmpeg',
        '-i', temp_path,
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-r', '30',
        '-f', 'avi',
        '-y',
        str(output_path)
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return result2.returncode == 0

def method_4_aggressive_fix(input_path, output_path):
    # Method 4: Aggressive fix with different codec
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'mjpeg',
        '-q:v', '1',
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        '-f', 'avi',
        '-vcodec', 'mjpeg',
        '-fflags', '+genpts',
        '-avoid_negative_ts', 'make_zero',
        '-err_detect', 'ignore_err',
        '-max_muxing_queue_size', '2048',
        '-y',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def method_5_mencoder(input_path, output_path):
    # Method 5: Use mencoder if available
    try:
        cmd = [
            'mencoder',
            str(input_path),
            '-ovc', 'copy',
            '-oac', 'copy',
            '-o', str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def fix_single_video_comprehensive(input_path, backup=False):
    # Fix a single video using all available methods
    
    # Test original video
    original_ok, original_info = test_video_header(input_path)
    print(f"  Original: {'OK' if original_ok else 'ISSUE'} - {original_info}")
    
    if original_ok:
        print(f"  Video already has correct headers, skipping...")
        return True
    
    # Create backup if requested
    backup_path = None
    if backup:
        backup_path = str(input_path) + '.backup'
        shutil.copy2(input_path, backup_path)
        print(f"  Created backup: {os.path.basename(backup_path)}")
    
    # Create temporary output path
    temp_output = str(input_path) + '.temp'
    
    # Try all methods
    methods = [
        ("Re-encode 30fps", method_1_reencode_30fps),
        ("Copy 30fps", method_2_copy_30fps),
        ("MP4 Bridge", method_3_mp4_bridge),
        ("Aggressive Fix", method_4_aggressive_fix),
        ("Mencoder", method_5_mencoder)
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"  Trying {method_name}...")
            if method_func(input_path, temp_output):
                # Test if it worked
                fixed_ok, fixed_info = test_video_header(temp_output)
                print(f"  {method_name} result: {'OK' if fixed_ok else 'ISSUE'} - {fixed_info}")
                
                if fixed_ok:
                    # Replace original with fixed version
                    shutil.move(temp_output, input_path)
                    print(f"  SUCCESS: {method_name} fixed the headers!")
                    return True
                else:
                    print(f"  {method_name} failed to fix headers")
                    # Clean up temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
            else:
                print(f"  {method_name} failed to process")
                # Clean up temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
        except Exception as e:
            print(f"  {method_name} exception: {e}")
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    print(f"  All methods failed for {os.path.basename(input_path)}")
    
    # Restore backup if available
    if backup_path and os.path.exists(backup_path):
        shutil.move(backup_path, input_path)
        print(f"  Restored original from backup")
    
    return False

def fix_camera_videos_comprehensive(camera_ip, data_path, backup=False):
    # Fix all videos for a specific camera
    
    input_video_path = os.path.join(data_path, camera_ip, "video")
    
    if not os.path.exists(input_video_path):
        print(f"Input video path not found: {input_video_path}")
        return False
    
    # Get all AVI files
    video_files = list(Path(input_video_path).glob("*.avi"))
    video_files = natsorted(video_files)
    
    print(f"\nFixing AVI headers for {camera_ip}")
    print(f"Found {len(video_files)} video files")
    if backup:
        print(f"Creating backups before replacement...")
    
    success_count = 0
    error_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        if fix_single_video_comprehensive(video_file, backup):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nResults for {camera_ip}:")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    
    return success_count > 0

def analyze_video_comprehensive(video_path):
    # Comprehensive video analysis with detailed error reporting
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {
            "error": f"Could not open {video_path}",
            "file_path": str(video_path)
        }
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    # Try to read first frame
    ret, frame = cap.read()
    first_frame_ok = ret
    
    # Try to read last frame
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        last_frame_ok = ret
    else:
        last_frame_ok = False
    
    cap.release()
    
    return {
        "file_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "fourcc_str": fourcc_str,
        "first_frame_ok": first_frame_ok,
        "last_frame_ok": last_frame_ok,
        "file_size_mb": os.path.getsize(video_path) / (1024 * 1024)
    }

def verify_all_cameras(data_path, cameras):
    # Comprehensive verification with detailed analysis
    print("\nComprehensive Video Analysis")
    
    all_ok = True
    all_results = {}
    
    for camera_ip in cameras:
        input_video_path = os.path.join(data_path, camera_ip, "video")
        if not os.path.exists(input_video_path):
            print(f"\n{camera_ip}: Video path not found")
            continue
            
        video_files = list(Path(input_video_path).glob("*.avi"))
        video_files = natsorted(video_files)
        
        print(f"\nAnalyzing {camera_ip}")
        print(f"Found {len(video_files)} video files")
        
        camera_results = []
        ok_count = 0
        error_count = 0
        first_frame_issues = 0
        last_frame_issues = 0
        
        for video_file in video_files:
            result = analyze_video_comprehensive(video_file)
            camera_results.append(result)
            
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                error_count += 1
                all_ok = False
            else:
                print(f"  {video_file.name}: {result['width']}x{result['height']}, {result['fps']}fps, {result['frame_count']} frames, {result['fourcc_str']}")
                
                if not result['first_frame_ok']:
                    first_frame_issues += 1
                if not result['last_frame_ok']:
                    last_frame_issues += 1
                
                if result['first_frame_ok'] and result['last_frame_ok']:
                    ok_count += 1
                else:
                    error_count += 1
                    all_ok = False
        
        all_results[camera_ip] = {
            "camera_ip": camera_ip,
            "total_files": len(video_files),
            "files": camera_results
        }
        
        # Print summary for this camera
        print(f"\n{camera_ip} Summary")
        print(f"  Total files: {len(video_files)}")
        print(f"  Files with errors: {error_count}")
        print(f"  Resolution: {list(set(f['width'] for f in camera_results if 'error' not in f))} x {list(set(f['height'] for f in camera_results if 'error' not in f))}")
        print(f"  FPS: {list(set(f['fps'] for f in camera_results if 'error' not in f))}")
        print(f"  Codec: {list(set(f['fourcc_str'] for f in camera_results if 'error' not in f))}")
        print(f"  First frame issues: {first_frame_issues}")
        print(f"  Last frame issues: {last_frame_issues}")
    
    # Print overall summary
    print("\n" + "-"*60)
    print("OVERALL SUMMARY")
    print("-"*60)
    
    total_files = 0
    total_errors = 0
    
    for camera_ip, result in all_results.items():
        total_files += result['total_files']
        error_files = [f for f in result['files'] if "error" in f or not (f.get('first_frame_ok', False) and f.get('last_frame_ok', False))]
        total_errors += len(error_files)
        
        print(f"\n{camera_ip}")
        print(f"  Total files: {result['total_files']}")
        print(f"  Files with errors: {len(error_files)}")
        
        if result['files']:
            good_files = [f for f in result['files'] if "error" not in f and f.get('first_frame_ok', False) and f.get('last_frame_ok', False)]
            if good_files:
                widths = set(f['width'] for f in good_files)
                heights = set(f['height'] for f in good_files)
                fps_values = set(f['fps'] for f in good_files)
                fourcc_values = set(f['fourcc_str'] for f in good_files)
                
                print(f"  Resolution: {list(widths)} x {list(heights)}")
                print(f"  FPS: {list(fps_values)}")
                print(f"  Codec: {list(fourcc_values)}")
    
    print("\nTOTAL SUMMARY")
    print(f"  Total files: {total_files}")
    print(f"  Total errors: {total_errors}")
    print(f"  Success rate: {((total_files - total_errors) / total_files * 100):.1f}%")
    
    # Save detailed results
    with open("comprehensive_video_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: comprehensive_video_analysis.json")
    
    return all_ok

def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupted AVI video headers using multiple repair methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  python fix_avi_header.py <data_path> [--backup] [--cameras CAM1 CAM2 ...] [--test-only] [--verify-only] [--analyze-only]

OPTIONS:
  data_path        Path to data directory containing camera folders
  --backup         Create backup of original files before fixing
  --cameras        Specific camera IPs to process (default: all detected)
  --test-only      Only test headers, don't fix anything
  --verify-only    Only verify headers after previous fixes
  --analyze-only   Comprehensive analysis with detailed error reporting
        """
    )
    parser.add_argument("data_path", help="Path to data directory containing camera folders")
    parser.add_argument("--backup", action="store_true", help="Create backup of original files before fixing")
    parser.add_argument("--cameras", nargs="+", default=["192.168.0.108", "192.168.0.122", "192.168.0.184", "192.168.0.227"], 
                       help="Specific camera IPs to process")
    parser.add_argument("--test-only", action="store_true", help="Only test headers, don't fix anything")
    parser.add_argument("--verify-only", action="store_true", help="Only verify headers after previous fixes")
    parser.add_argument("--analyze-only", action="store_true", help="Comprehensive analysis with detailed error reporting")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("Error: ffmpeg is not available. Please install ffmpeg first.")
        return
    
    print("ffmpeg found, proceeding with comprehensive AVI header fixes...")
    
    # Determine processing device
    processing_device = get_processing_device()
    print(f"Using {processing_device.upper()} for video analysis")
    
    if args.test_only:
        print("TEST MODE: Only testing headers, not fixing...")
        # Test all cameras
        for camera_ip in args.cameras:
            input_video_path = os.path.join(args.data_path, camera_ip, "video")
            if os.path.exists(input_video_path):
                video_files = list(Path(input_video_path).glob("*.avi"))
                video_files = natsorted(video_files)
                
                print(f"\n=== Testing {camera_ip} ===")
                for video_file in video_files:  # Test all files
                    ok, info = test_video_header(video_file)
                    print(f"  {video_file.name}: {'OK' if ok else 'ISSUE'} - {info}")
        return
    
    if args.verify_only:
        verify_all_cameras(args.data_path, args.cameras)
        return
    
    if args.analyze_only:
        print("ANALYSIS MODE: Comprehensive video analysis with detailed error reporting...")
        verify_all_cameras(args.data_path, args.cameras)
        return
    
    # Process each camera
    start_time = time.time()
    all_success = True
    
    for camera_ip in args.cameras:
        success = fix_camera_videos_comprehensive(camera_ip, args.data_path, args.backup)
        if not success:
            all_success = False
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if all_success:
        print("\nAll cameras processed successfully")
        print(f"Processing time: {processing_time:.1f} seconds")
        print(f"Original video files have been replaced with corrected versions")
        
        # Verify results
        print("\nFinal verification")
        verify_all_cameras(args.data_path, args.cameras)
    else:
        print("\nSome cameras had issues")
        print(f"Processing time: {processing_time:.1f} seconds")

if __name__ == "__main__":
    main() 