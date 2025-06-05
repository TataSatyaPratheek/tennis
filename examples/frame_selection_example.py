#!/usr/bin/env python3

from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.frame_selection.selector import CalibrationFrameSelector

def main():
    """Example usage of frame selection pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize frame selector
    selector = CalibrationFrameSelector(
        min_calibration_frames=8,
        max_calibration_frames=15
    )
    
    # Path to processed frames
    frames_dir = Path("data/processed_frames")
    
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return
    
    try:
        # Select calibration frames
        results = selector.select_calibration_frames(
            frames_dir=frames_dir,
            max_analyze=50  # Analyze maximum 50 frames for efficiency
        )
        
        # Save results
        output_path = Path("data/analysis_reports/frame_selection_results.json")
        selector.save_selection_results(results, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("FRAME SELECTION RESULTS")
        print("="*60)
        
        report = results['selection_report']
        print(f"Total frames analyzed: {report['total_frames_analyzed']}")
        print(f"Frames selected for calibration: {report['frames_selected']}")
        print(f"Selection rate: {report['selection_rate']:.1f}%")
        print(f"\nScore statistics (selected frames):")
        print(f"  Mean score: {report['score_statistics']['selected_frames']['mean']:.1f}")
        print(f"  Score range: {report['score_statistics']['selected_frames']['min']:.1f} - {report['score_statistics']['selected_frames']['max']:.1f}")
        
        print(f"\nQuality distribution:")
        for grade, count in report['quality_distribution'].items():
            print(f"  {grade.capitalize()}: {count} frames")
        
        print(f"\nSelected frame IDs:")
        for frame_id in report['selected_frame_ids']:
            print(f"  - {frame_id}")
        
        print(f"\nRecommendation: {report['recommendation']}")
        print(f"\nDetailed results saved to: {output_path}")
        
        # Visualize selected frames (optional)
        visualize_selected_frames = input("\nWould you like to visualize selected frames? (y/n): ")
        if visualize_selected_frames.lower() == 'y':
            visualize_frames(results['selected_frames'], frames_dir)
        
    except Exception as e:
        logger.error(f"Error during frame selection: {e}")
        import traceback
        traceback.print_exc()

def visualize_frames(selected_frames, frames_dir):
    """Optional visualization of selected frames"""
    import cv2
    
    print("\nDisplaying selected frames (press any key to continue, 'q' to quit)...")
    
    for i, frame_data in enumerate(selected_frames):
        frame_id = frame_data['frame_id']
        
        # Try to find the frame file
        possible_files = [
            frames_dir / f"{frame_id}.jpg",
            frames_dir / f"frame_{frame_id}.jpg",
            frames_dir / f"processed_frame_{frame_id}.jpg"
        ]
        
        frame_file = None
        for file_path in possible_files:
            if file_path.exists():
                frame_file = file_path
                break
        
        if frame_file:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                # Add frame information overlay
                score = frame_data['score']
                info_text = [
                    f"Frame: {frame_id} ({i+1}/{len(selected_frames)})",
                    f"Score: {score.total_score:.1f} ({score.calibration_suitability})",
                    f"Features: {frame_data['features'].feature_count}",
                    f"Quality: {score.quality_score:.1f}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Show frame
                cv2.imshow('Selected Calibration Frame', frame)
                key = cv2.waitKey(0)
                
                if key == ord('q'):
                    break
        else:
            print(f"Could not find frame file for: {frame_id}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
