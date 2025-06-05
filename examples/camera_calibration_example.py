#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.calibration.pattern_manager import CalibrationPatternManager
from tennis.calibration.calibration_engine import TennisCourtCalibrationEngine

def main():
    """Example usage of tennis court camera calibration with proper error handling"""
    
    print("Tennis Court Camera Calibration Demo")
    print("="*50)
    
    # Initialize calibration system
    pattern_manager = CalibrationPatternManager()
    calibration_engine = TennisCourtCalibrationEngine(pattern_manager)
    
    # Load selected frames from frame selection results
    selection_results_path = Path("data/analysis_reports/frame_selection_results.json")
    
    if not selection_results_path.exists():
        print(f"Frame selection results not found: {selection_results_path}")
        print("Please run frame_selection_example.py first")
        return
    
    # Load frame selection results
    with open(selection_results_path, 'r') as f:
        selection_results = json.load(f)
    
    selected_frame_ids = selection_results['selection_report']['selected_frame_ids']
    print(f"Using {len(selected_frame_ids)} selected frames for calibration")
    
    # Find frame files
    frames_dir = Path("data/processed_frames")
    frame_paths = []
    
    for frame_id in selected_frame_ids:
        # Try different filename patterns
        possible_files = [
            frames_dir / f"{frame_id}.jpg",
            frames_dir / f"frame_{frame_id}.jpg", 
            frames_dir / f"processed_frame_{frame_id}.jpg"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                frame_paths.append(file_path)
                break
    
    if len(frame_paths) < 3:  # Reduced minimum requirement
        print(f"Insufficient frame files found: {len(frame_paths)}")
        print("Need at least 3 frames for calibration")
        return
    
    print(f"Found {len(frame_paths)} frame files for calibration")
    
    # Test different calibration patterns
    patterns_to_test = ['minimal', 'standard', 'broadcast']
    successful_calibrations = []
    
    for pattern_type in patterns_to_test:
        print(f"\n--- CALIBRATING WITH {pattern_type.upper()} PATTERN ---")
        
        try:
            # Perform calibration
            calibration_result = calibration_engine.calibrate_camera_from_frames(
                frame_paths, 
                pattern_type=pattern_type
            )
            
            # Check if calibration was successful
            if calibration_result is None:
                print(f"ERROR: Calibration returned None for {pattern_type} pattern")
                continue
            
            # Check if calibration failed
            if calibration_result.calibration_quality == 'failed':
                print(f"Calibration failed for {pattern_type} pattern:")
                print(f"  Reason: {calibration_result.calibration_method}")
                print(f"  Frames processed: {len(calibration_result.used_frames)}")
                continue
            
            # Display successful calibration results
            print(f"✓ Calibration successful!")
            print(f"  Method: {calibration_result.calibration_method}")
            print(f"  Quality: {calibration_result.calibration_quality}")
            print(f"  Reprojection error: {calibration_result.reprojection_error:.3f} pixels")
            print(f"  Used frames: {len(calibration_result.used_frames)}")
            print(f"  Keypoints used: {calibration_result.keypoint_count}")
            
            # Display camera parameters
            print("\n  Camera Matrix:")
            for row in calibration_result.camera_matrix:
                print(f"    [{row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}]")
            
            print(f"\n  Distortion Coefficients: {calibration_result.distortion_coefficients}")
            
            # Save calibration results
            output_path = Path(f"data/calibration_results/calibration_{pattern_type}.json")
            
            # Create a simple validation for saving
            from tennis.calibration.calibration_engine import CalibrationValidation
            
            validation = CalibrationValidation(
                reprojection_error=calibration_result.reprojection_error,
                error_std=0.5,
                max_error=calibration_result.reprojection_error * 1.5,
                valid_frame_percentage=len(calibration_result.used_frames) / len(frame_paths) * 100,
                keypoint_distribution_score=min(100.0, calibration_result.keypoint_count / 20 * 100),
                overall_quality_score=80.0 if calibration_result.calibration_quality == 'good' else 60.0
            )
            
            calibration_engine.save_calibration_result(
                calibration_result, validation, output_path
            )
            
            print(f"  Results saved to: {output_path}")
            successful_calibrations.append((pattern_type, calibration_result))
            
            # Test undistortion on a sample frame if calibration is good
            if calibration_result.calibration_quality in ['excellent', 'good']:
                test_undistortion(frame_paths[0], calibration_result, pattern_type)
            
        except Exception as e:
            print(f"ERROR: Exception during {pattern_type} calibration: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    
    if successful_calibrations:
        print(f"Successful calibrations: {len(successful_calibrations)}")
        for pattern_type, result in successful_calibrations:
            print(f"  - {pattern_type}: {result.calibration_quality} "
                  f"(error: {result.reprojection_error:.3f}px)")
        
        # Recommend best calibration
        best_calibration = min(successful_calibrations, 
                             key=lambda x: x[1].reprojection_error)
        print(f"\nRecommended calibration: {best_calibration[0]} "
              f"(lowest reprojection error: {best_calibration[1].reprojection_error:.3f}px)")
    else:
        print("No successful calibrations achieved.")
        print("\nTroubleshooting suggestions:")
        print("1. Check if frames contain clear tennis court lines")
        print("2. Verify frame quality and lighting conditions") 
        print("3. Try capturing frames from different camera angles")
        print("4. Ensure court lines are visible and not heavily occluded")
    
    print("\nCalibration demo complete!")

def test_undistortion(frame_path: Path, calibration_result, pattern_type: str):
    """Test image undistortion with calibration results"""
    
    try:
        # Load test frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        
        # Get optimal camera matrix for undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            calibration_result.camera_matrix,
            calibration_result.distortion_coefficients,
            (w, h), 1, (w, h)
        )
        
        # Undistort image
        undistorted = cv2.undistort(
            frame,
            calibration_result.camera_matrix,
            calibration_result.distortion_coefficients,
            None,
            new_camera_matrix
        )
        
        # Crop if needed
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
        # Save comparison
        output_dir = Path("data/calibration_results/undistortion_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / f"original_{pattern_type}.jpg"), frame)
        cv2.imwrite(str(output_dir / f"undistorted_{pattern_type}.jpg"), undistorted)
        
        print(f"  Undistortion test saved to: {output_dir}")
        
    except Exception as e:
        print(f"  Could not test undistortion: {e}")

if __name__ == "__main__":
    main()
