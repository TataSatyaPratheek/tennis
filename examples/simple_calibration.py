#!/usr/bin/env python3
"""Complete working example using professional libraries"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tennis.detection import CourtDetector
from tennis.calibration import TennisCalibrator
from tennis.video import VideoProcessor
from tennis.utils.geometry import create_court_polygon

def main():
    """Real implementation using professional libraries"""
    
    # Initialize components with proven libraries
    detector = CourtDetector()  # Uses YOLO + MediaPipe
    calibrator = TennisCalibrator()  # Uses OpenCV calibration
    processor = VideoProcessor()  # Uses video-reader-rs/decord + yt-dlp
    
    # Process video using proven pipeline
    video_url = "https://www.youtube.com/watch?v=ZlQK3w9_cXw"
    frames = processor.process_video_url(video_url, sample_rate=30)
    
    print(f"Extracted {len(frames)} frames using {processor.backend}")
    
    # Detect court keypoints using ML models
    keypoints_2d = []
    for i, frame in enumerate(frames[:10]):  # Process first 10 frames
        keypoints = detector.detect_court_keypoints(frame)
        if len(keypoints.points) >= 4:
            keypoints_2d.append(keypoints.points[:4])  # Use first 4 points
            print(f"Frame {i}: {len(keypoints.points)} keypoints detected using {keypoints.method}")
    
    if len(keypoints_2d) >= 3:
        # Calibrate using OpenCV's proven algorithms
        result = calibrator.calibrate_from_frames(frames[:len(keypoints_2d)], keypoints_2d)
        
        if result.success:
            print(f"Calibration successful!")
            print(f"Reprojection error: {result.reprojection_error:.3f} pixels")
            print(f"Camera matrix:\n{result.camera_matrix}")
        else:
            print(f"Calibration failed. Error: {result.reprojection_error:.3f}")
    else:
        print("Insufficient keypoints for calibration")

if __name__ == "__main__":
    main()
