import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
import logging
from tqdm import tqdm

class M1OptimizedFrameExtractor:
    """Memory-efficient frame extraction optimized for M1 Mac with 8GB RAM"""
    
    def __init__(self, target_resolution: Tuple[int, int] = (1280, 720)):
        self.target_resolution = target_resolution
        self.logger = logging.getLogger(__name__)
    
    def extract_frames_batch(self, 
                           video_path: Path,
                           output_dir: Path,
                           sample_rate: int = 30,
                           max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """Extract frames with memory management for 8GB constraint"""
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Video info: {total_frames} frames at {fps} FPS")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 0
        saved_count = 0
        
        # Memory-efficient processing with progress bar
        with tqdm(total=min(total_frames//sample_rate, max_frames or float('inf'))) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and saved_count >= max_frames):
                    break
                
                # Sample frames to reduce memory usage
                if frame_count % sample_rate == 0:
                    # Resize frame to target resolution
                    frame_resized = cv2.resize(frame, self.target_resolution)
                    
                    # Save frame
                    frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame_resized, 
                              [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compress to save space
                    
                    saved_count += 1
                    pbar.update(1)
                    
                    yield frame_resized
                
                frame_count += 1
        
        cap.release()
        self.logger.info(f"Extracted {saved_count} frames")
