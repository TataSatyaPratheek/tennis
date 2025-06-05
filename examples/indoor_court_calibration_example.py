#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.calibration.pattern_manager import CalibrationPatternManager
from tennis.calibration.calibration_engine import TennisCourtCalibrationEngine

def main():
    """Test indoor court calibration with enhanced detector"""
    
    print("Indoor Tennis Court Calibration Test")
    print("="*50)
    
    # Initialize with indoor detector
    pattern_manager = CalibrationPatternManager()
    calibration_engine = TennisCourtCalibrationEngine(
        pattern_manager, 
        use_indoor_detector=True  # Use specialized indoor detector
    )
    
    # Test with your frames
    frames_dir = Path("data/processed_frames")
    frame_files = sorted(frames_dir.glob("*.jpg"))[:10]  # Test with first 10 frames
    
    print(f"Testing with {len(frame_files)} frames")
    
    try:
        result = calibration_engine.calibrate_camera_from_frames(
            frame_files, 
            pattern_type='minimal'
        )
        
        print(f"Calibration Result:")
        print(f"  Quality: {result.calibration_quality}")
        print(f"  Reprojection Error: {result.reprojection_error:.3f} pixels")
        print(f"  Keypoints Used: {result.keypoint_count}")
        
    except Exception as e:
        print(f"Calibration failed: {e}")

if __name__ == "__main__":
    main()
