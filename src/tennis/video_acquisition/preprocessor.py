import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

class TennisVideoPreprocessor:
    """Tennis-specific video preprocessing optimized for court detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_court_features(self, frame: np.ndarray) -> np.ndarray:
        """Enhance tennis court lines for better detection"""[5]
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Non-local Means Denoising for cleaner lines
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def extract_court_region(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract court region using background subtraction"""[6]
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define tennis court color ranges (green/blue courts)
        # Tennis courts are typically green, blue, or red
        court_ranges = [
            # Green court range
            (np.array([35, 40, 40]), np.array([85, 255, 255])),
            # Blue court range  
            (np.array([100, 50, 50]), np.array([130, 255, 255]))
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in court_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original frame
        court_region = cv2.bitwise_and(frame, frame, mask=combined_mask)
        
        return court_region, combined_mask
    
    def reduce_noise_temporal(self, frames: List[np.ndarray], frame_index: int, 
                            window_size: int = 5) -> np.ndarray:
        """Apply temporal denoising using multiple frames"""[5]
        
        if len(frames) < window_size or frame_index < window_size // 2:
            return frames[frame_index]
        
        # Select frames for temporal window
        start_idx = max(0, frame_index - window_size // 2)
        end_idx = min(len(frames), frame_index + window_size // 2 + 1)
        temporal_frames = frames[start_idx:end_idx]
        
        # Convert to numpy array for efficient processing
        frame_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) 
                               for f in temporal_frames])
        
        # Apply temporal median filter
        denoised = np.median(frame_stack, axis=0).astype(np.uint8)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
