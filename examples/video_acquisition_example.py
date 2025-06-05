#!/usr/bin/env python3

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.video_acquisition.pipeline import TennisVideoAcquisitionPipeline

def main():
    """Example usage of the video acquisition pipeline"""
    
    # Initialize pipeline
    pipeline = TennisVideoAcquisitionPipeline(data_dir=Path("data"))
    
    # Process the tennis prodigy M3 video
    processed_frames_dir = pipeline.process_tennis_video(
        video_url="https://www.youtube.com/watch?v=ZlQK3w9_cXw",
        sample_rate=30,  # Extract every 30th frame
        max_frames=100   # Limit for memory management
    )
    
    print(f"Video processing complete!")
    print(f"Processed frames available at: {processed_frames_dir}")

if __name__ == "__main__":
    main()
