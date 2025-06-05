"""Video overlay using OpenCV - actual implementation"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union

class VideoOverlay:
    """Real video overlay implementation using OpenCV"""
    
    def __init__(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    def create_overlay_video(self, video1_path: Union[str, Path], 
                           video2_path: Union[str, Path],
                           output_path: Union[str, Path],
                           opacity: float = 0.5) -> bool:
        """Create overlay video using OpenCV VideoWriter"""
        
        # Open both videos using OpenCV
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))
        
        # Get video properties
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        out = cv2.VideoWriter(str(output_path), self.fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Resize frame2 to match frame1 if needed
            if frame2.shape[:2] != frame1.shape[:2]:
                frame2 = cv2.resize(frame2, (width, height))
            
            # Create overlay using OpenCV's addWeighted
            overlay_frame = cv2.addWeighted(frame1, 1.0 - opacity, frame2, opacity, 0)
            
            # Write frame
            out.write(overlay_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release everything
        cap1.release()
        cap2.release()
        out.release()
        
        return True
    
    def create_side_by_side(self, video1_path: Union[str, Path], 
                          video2_path: Union[str, Path],
                          output_path: Union[str, Path]) -> bool:
        """Create side-by-side comparison using OpenCV hstack"""
        
        cap1 = cv2.VideoCapture(str(video1_path))
        cap2 = cv2.VideoCapture(str(video2_path))
        
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output will be twice the width
        out = cv2.VideoWriter(str(output_path), self.fourcc, fps, (width * 2, height))
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # Resize to match if needed
            frame2 = cv2.resize(frame2, (width, height))
            
            # Use OpenCV's hstack for side-by-side
            combined_frame = np.hstack((frame1, frame2))
            out.write(combined_frame)
        
        cap1.release()
        cap2.release()
        out.release()
        
        return True
