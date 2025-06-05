"""Video processing using modern libraries"""
from decord import VideoReader
import yt_dlp
import numpy as np
from pathlib import Path
from typing import List, Union

class VideoProcessor:
    """Video processing using decord + yt-dlp"""
    
    def __init__(self):
        # Use yt-dlp for downloading - handles all formats
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Reasonable quality
            'outtmpl': 'data/%(title)s.%(ext)s'
        }
    
    def download_video(self, url: str) -> Path:
        """Download video using yt-dlp"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    
    def extract_frames(self, video_path: Union[str, Path], 
                      sample_rate: int = 30) -> List[np.ndarray]:
        """Extract frames using decord (fastest video library)"""
        
        # Use decord - optimized for ML workloads
        vr = VideoReader(str(video_path))
        
        # Sample frames evenly
        indices = range(0, len(vr), sample_rate)
        frames = vr.get_batch(list(indices)).asnumpy()
        
        # Convert from RGB to BGR for OpenCV compatibility
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    
    def process_video_url(self, url: str, sample_rate: int = 30) -> List[np.ndarray]:
        """Complete pipeline: download + extract frames"""
        video_path = self.download_video(url)
        return self.extract_frames(video_path, sample_rate)
