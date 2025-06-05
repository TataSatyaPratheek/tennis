import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json
from dataclasses import dataclass

@dataclass
class CameraMetadata:
    """Data class for camera metadata information"""
    resolution: Tuple[int, int]
    fps: float
    total_frames: int
    duration: float
    codec: str
    bitrate: Optional[int] = None
    aspect_ratio: float = 1.0

class VideoMetadataExtractor:
    """Extract comprehensive metadata from tennis video for camera analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_basic_metadata(self, video_path: Path) -> CameraMetadata:
        """Extract basic camera and video metadata"""
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Extract basic video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate derived properties
        duration = total_frames / fps if fps > 0 else 0
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Get codec information
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        metadata = CameraMetadata(
            resolution=(width, height),
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            aspect_ratio=aspect_ratio
        )
        
        self.logger.info(f"Video metadata: {width}x{height} @ {fps}fps, {total_frames} frames")
        return metadata
    
    def analyze_dynamic_parameters(self, video_path: Path, sample_frames: int = 50) -> Dict[str, Any]:
        """Analyze dynamic camera parameters throughout the video"""
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        brightness_values = []
        contrast_values = []
        sharpness_values = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate brightness (mean intensity)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Calculate contrast (standard deviation)
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                # Calculate sharpness (variance of Laplacian)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_values.append(sharpness)
        
        cap.release()
        
        return {
            'brightness': {
                'mean': float(np.mean(brightness_values)),
                'std': float(np.std(brightness_values)),
                'min': float(np.min(brightness_values)),
                'max': float(np.max(brightness_values))
            },
            'contrast': {
                'mean': float(np.mean(contrast_values)),
                'std': float(np.std(contrast_values)),
                'min': float(np.min(contrast_values)),
                'max': float(np.max(contrast_values))
            },
            'sharpness': {
                'mean': float(np.mean(sharpness_values)),
                'std': float(np.std(sharpness_values)),
                'min': float(np.min(sharpness_values)),
                'max': float(np.max(sharpness_values))
            }
        }
