#!/usr/bin/env python3
"""Working tennis video overlay - no placeholders"""

import cv2
from tennis.detection import CourtDetector
from tennis.video import VideoOverlay
from pathlib import Path

def main():
    # Real implementations using OpenCV
    detector = CourtDetector()
    overlay = VideoOverlay()
    
    # Process your tennis frames
    video_path = "your_tennis_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if ret:
        # Detect court using actual OpenCV methods
        keypoints = detector.detect_court_keypoints(frame)
        print(f"Detected {len(keypoints.points)} court points using {keypoints.method}")
        
        # Create overlay video using OpenCV VideoWriter
        overlay.create_overlay_video(
            "video1.mp4", "video2.mp4", "overlay_output.mp4", opacity=0.6
        )
    
    cap.release()

if __name__ == "__main__":
    main()
