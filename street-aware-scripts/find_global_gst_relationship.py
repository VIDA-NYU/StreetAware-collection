#!/usr/bin/env python3
"""
Find the linear relationship between global_timestamp and gstreamer_timestamp in your timeline files.
Prints the slope, intercept, and R^2 value for use in threshold conversion.
"""
import os
import json
import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def extract_pairs(data_path):
    pairs = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for value in data.values():
                                if isinstance(value, dict) and 'global_timestamp' in value and 'gstreamer_timestamp' in value:
                                    pairs.append((value['global_timestamp'], value['gstreamer_timestamp']))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return pairs

def fit_linear(pairs):
    if not pairs:
        return None, None, None, None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, intercept, r2, (x, y, y_pred)

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_global_gst_relationship.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    print(f"Scanning {data_path} for timeline JSON files...")
    pairs = extract_pairs(data_path)
    print(f"Found {len(pairs)} (global_timestamp, gstreamer_timestamp) pairs.")
    if not pairs:
        print("No valid pairs found.")
        sys.exit(1)
    slope, intercept, r2, (x, y, y_pred) = fit_linear(pairs)
    print("\nLinear relationship:")
    print(f"  gstreamer_timestamp = {slope:.8f} * global_timestamp + {intercept:.8f}")
    print(f"  R^2 = {r2:.6f}")
    print(f"  seconds_per_global_unit (slope) = {slope:.8f}")
    print(f"  global units per second = {1/slope if slope != 0 else float('inf'):.3f}")
    print("\nUse this slope for threshold conversion in your sync code.")
    if HAS_PLOT:
        plt.figure(figsize=(8,5))
        plt.scatter(x, y, s=1, label='Data')
        plt.plot(x, y_pred, color='red', label='Fit')
        plt.xlabel('global_timestamp')
        plt.ylabel('gstreamer_timestamp (seconds)')
        plt.title('Global Timestamp vs GStreamer Timestamp')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 