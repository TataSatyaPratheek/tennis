from pathlib import Path
from typing import List, Generator
import logging

from .downloader import YouTubeVideoDownloader
from .frame_extractor import M1OptimizedFrameExtractor
from .preprocessor import TennisVideoPreprocessor
from .memory_manager import M1MemoryManager

class TennisVideoAcquisitionPipeline:
    """Complete video acquisition and preprocessing pipeline"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.downloader = YouTubeVideoDownloader(data_dir / "raw_videos")
        self.extractor = M1OptimizedFrameExtractor()
        self.preprocessor = TennisVideoPreprocessor()
        self.memory_manager = M1MemoryManager()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_tennis_video(self, 
                           video_url: str = "https://www.youtube.com/watch?v=ZlQK3w9_cXw",
                           sample_rate: int = 30,
                           max_frames: int = 200) -> Path:
        """Complete processing pipeline for tennis video"""
        
        self.logger.info("Starting tennis video acquisition pipeline")
        
        # Step 1: Download video
        self.logger.info("Downloading video...")
        video_path = self.downloader.download_tennis_video(video_url)
        
        # Step 2: Extract frames with memory management
        self.logger.info("Extracting frames...")
        frames_dir = self.data_dir / "processed_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        processed_frames = []
        for frame in self.extractor.extract_frames_batch(
            video_path, frames_dir, sample_rate, max_frames
        ):
            # Check memory usage
            self.memory_manager.check_memory_usage()
            
            # Preprocess frame
            enhanced_frame = self.preprocessor.enhance_court_features(frame)
            court_region, _ = self.preprocessor.extract_court_region(enhanced_frame)
            
            processed_frames.append(court_region)
            
            # Process in batches to manage memory
            if len(processed_frames) >= 10:
                # Save batch and clear memory
                self._save_frame_batch(processed_frames, frames_dir)
                processed_frames.clear()
        
        # Save remaining frames
        if processed_frames:
            self._save_frame_batch(processed_frames, frames_dir)
        
        self.logger.info(f"Processing complete. Frames saved to: {frames_dir}")
        return frames_dir
    
    def _save_frame_batch(self, frames: List, output_dir: Path) -> None:
        """Save a batch of processed frames"""
        import cv2
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"processed_frame_{len(list(output_dir.glob('*.jpg')))+i:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
