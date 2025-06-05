"""Simplified pipeline using existing libraries"""
import yt_dlp
from video_reader import PyVideoReader as VideoReader
import cv2
from pathlib import Path
from typing import List

class SimpleVideoAcquisitionPipeline:
    """Minimal pipeline using proven libraries"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
    def process_tennis_video(self, video_url: str, sample_rate: int = 30) -> Path:
        """Download and extract frames using standard libraries"""
        
        # Download with yt-dlp
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': str(self.data_dir / 'video.%(ext)s')
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Extract frames with video-reader-rs
        video_files = list(self.data_dir.glob('video.*'))
        if not video_files:
            raise ValueError("Video download failed")
            
        video_path = video_files[0]
        reader = VideoReader(str(video_path))
        
        # Save frames
        frames_dir = self.data_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for i in range(0, len(reader), sample_rate):
            frame = reader[i]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), frame_bgr)
        
        return frames_dir
