"""Video processing using best-in-class libraries"""
try:
    from video_reader import PyVideoReader  # Correct import for video_reader-rs
    VIDEO_BACKEND = 'video_reader_rs'
except ImportError:
    try:
        from decord import VideoReader  # Fallback
        VIDEO_BACKEND = 'decord'
    except ImportError:
        import cv2
        VIDEO_BACKEND = 'opencv'

import yt_dlp
import numpy as np
from pathlib import Path
from typing import List, Union

class VideoProcessor:
    """Video processing using proven libraries - no custom implementations"""
    
    def __init__(self):
        self.backend = VIDEO_BACKEND
        print(f"Using video backend: {self.backend}")
        # Use yt-dlp - industry standard for video downloading
        self.ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': 'data/%(title)s.%(ext)s'
        }
    
    def download_video(self, url: str) -> Path:
        """Download using yt-dlp - don't reinvent video downloading"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    
    def extract_frames(self, video_path: Union[str, Path], 
                      sample_rate: int = 30) -> List[np.ndarray]:
        """Use the best available video reader"""
        
        if self.backend == 'video_reader_rs':
            return self._extract_with_video_reader_rs(video_path, sample_rate)
        elif self.backend == 'decord':
            return self._extract_with_decord(video_path, sample_rate)
        else:
            return self._extract_with_opencv(video_path, sample_rate)
    
    def _extract_with_video_reader_rs(self, video_path, sample_rate):
        """Use video-reader-rs for optimal performance"""
        vr = PyVideoReader(str(video_path))  # Use PyVideoReader, not VideoReader
        
        # Get total frame count and create indices
        total_frames = vr.get_shape()[0]
        indices = list(range(0, total_frames, sample_rate))
        
        # Use get_batch method for extracting specific frames
        frames = vr.get_batch(indices)
        
        # Convert to list of numpy arrays (frames is already numpy array)
        return [frames[i] for i in range(len(frames))]
    
    def _extract_with_decord(self, video_path, sample_rate):
        """Use decord as fallback"""
        from decord import VideoReader
        vr = VideoReader(str(video_path))
        indices = range(0, len(vr), sample_rate)
        frames = vr.get_batch(list(indices)).asnumpy()
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    
    def _extract_with_opencv(self, video_path, sample_rate):
        """Use OpenCV as final fallback"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
