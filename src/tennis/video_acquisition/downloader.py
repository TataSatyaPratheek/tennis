import subprocess
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any

class YouTubeVideoDownloader:
    """Download tennis video using yt-dlp with M1 optimization"""
    
    def __init__(self, output_dir: Path = Path("data/raw_videos")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_ytdlp_installed()
    
    def _ensure_ytdlp_installed(self) -> None:
        """Install yt-dlp if not available using uv"""[7]
        try:
            import yt_dlp
        except ImportError:
            subprocess.run([sys.executable, "-m", "uv", "pip", "install", "yt-dlp"], check=True)
    
    def download_tennis_video(self, url: str = "https://www.youtube.com/watch?v=ZlQK3w9_cXw") -> Path:
        """Download M3 tennis prodigy video with optimized settings for M1"""
        import yt_dlp
        
        # M1-optimized download options
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for 8GB RAM constraint
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'writeinfojson': True,  # Save metadata for analysis
            'extractaudio': False,
            'merge_output_format': 'mp4'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return Path(filename)
