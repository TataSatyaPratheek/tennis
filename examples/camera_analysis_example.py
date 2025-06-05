#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.camera_analysis.analyzer import ComprehensiveCameraAnalyzer

def load_processed_frames(frames_dir: Path, max_frames: int = 20) -> list:
    """Load processed frames as fallback option"""
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    
    if not frame_files:
        print("No processed frames found!")
        return []
    
    # Sample frames evenly
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files)-1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    frames = []
    for frame_path in frame_files:
        try:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames.append(frame)
                print(f"Loaded frame: {frame_path.name}")
        except Exception as e:
            print(f"Error loading frame {frame_path}: {e}")
    
    return frames

def create_dummy_video_metadata(frames: list) -> dict:
    """Create dummy video metadata when original video is unavailable"""
    
    if not frames:
        return {
            'resolution': (1280, 720),
            'fps': 30.0,
            'total_frames': 0,
            'duration': 0.0,
            'codec': 'unknown'
        }
    
    h, w = frames[0].shape[:2]
    return {
        'resolution': (w, h),
        'fps': 30.0,  # Estimated
        'total_frames': len(frames),
        'duration': len(frames) / 30.0,
        'codec': 'mp4v'
    }

def main():
    """Example usage of camera analysis pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize analyzer
    analyzer = ComprehensiveCameraAnalyzer()
    
    # Try multiple video path variations
    video_paths = [
        Path("data/raw_videos/7 YEAR OLD TENNIS PRODIGY- M3.mp4"),  # Actual filename
        Path("data/raw_videos/7_YEAR_OLD_TENNIS_PRODIGY_M3.mp4"),   # Script's original path
        Path("data/raw_videos").glob("*.mp4")  # Any mp4 in the directory
    ]
    
    video_path = None
    sample_frames = []
    
    # Try to find and open video file
    for path_option in video_paths[:2]:  # First two are direct paths
        if isinstance(path_option, Path) and path_option.exists():
            print(f"Found video file: {path_option}")
            cap = cv2.VideoCapture(str(path_option))
            
            if cap.isOpened():
                video_path = path_option
                print(f"Successfully opened video: {video_path}")
                
                # Extract sample frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    frame_indices = np.linspace(0, total_frames-1, 20, dtype=int)
                    
                    for frame_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            sample_frames.append(frame)
                
                cap.release()
                break
            else:
                cap.release()
                print(f"Could not open video file: {path_option}")
    
    # Fallback: Try to find any MP4 files
    if not video_path:
        video_dir = Path("data/raw_videos")
        if video_dir.exists():
            mp4_files = list(video_dir.glob("*.mp4"))
            if mp4_files:
                for mp4_file in mp4_files:
                    print(f"Trying MP4 file: {mp4_file}")
                    cap = cv2.VideoCapture(str(mp4_file))
                    if cap.isOpened():
                        video_path = mp4_file
                        
                        # Extract sample frames
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 0:
                            frame_indices = np.linspace(0, total_frames-1, 20, dtype=int)
                            
                            for frame_idx in frame_indices:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame = cap.read()
                                if ret:
                                    sample_frames.append(frame)
                        
                        cap.release()
                        break
                    cap.release()
    
    # If still no video, use processed frames
    if not sample_frames:
        print("No video file accessible, using processed frames...")
        frames_dir = Path("data/processed_frames")
        sample_frames = load_processed_frames(frames_dir, max_frames=20)
        
        if not sample_frames:
            print("No frames available for analysis!")
            return
        
        # Create a dummy video path for the metadata extractor
        video_path = Path("data/processed_frames")  # Use frames directory as placeholder
    
    print(f"Loaded {len(sample_frames)} frames for analysis")
    
    # Create custom analyzer that handles missing video file
    class ModifiedCameraAnalyzer(ComprehensiveCameraAnalyzer):
        def analyze_video_camera_characteristics(self, video_path, sample_frames):
            """Modified version that handles missing video files"""
            
            # Try to extract metadata normally
            try:
                metadata = self.metadata_extractor.extract_basic_metadata(video_path)
                dynamic_params = self.metadata_extractor.analyze_dynamic_parameters(video_path)
            except Exception as e:
                logger.warning(f"Could not extract video metadata: {e}")
                # Use dummy metadata based on frames
                dummy_metadata = create_dummy_video_metadata(sample_frames)
                metadata = type('obj', (object,), dummy_metadata)
                dynamic_params = {
                    'brightness': {'mean': 128, 'std': 30, 'min': 50, 'max': 200},
                    'contrast': {'mean': 50, 'std': 15, 'min': 20, 'max': 80},
                    'sharpness': {'mean': 100, 'std': 25, 'min': 50, 'max': 150}
                }
            
            # Continue with rest of analysis
            motion_data = self.motion_analyzer.analyze_camera_motion(sample_frames)
            lighting_conditions = self.lighting_analyzer.analyze_lighting_conditions(sample_frames)
            
            if sample_frames:
                camera_position = self.position_estimator.estimate_camera_position(sample_frames[0])
            else:
                camera_position = None
            
            # Compile analysis result
            analysis_result = {
                'metadata': {
                    'resolution': getattr(metadata, 'resolution', (1280, 720)),
                    'fps': getattr(metadata, 'fps', 30.0),
                    'total_frames': getattr(metadata, 'total_frames', len(sample_frames)),
                    'duration': getattr(metadata, 'duration', len(sample_frames) / 30.0),
                    'codec': getattr(metadata, 'codec', 'unknown'),
                    'aspect_ratio': getattr(metadata, 'aspect_ratio', 16/9)
                },
                'dynamic_parameters': dynamic_params,
                'motion_analysis': {
                    'average_motion_magnitude': float(np.mean([m.motion_magnitude for m in motion_data])) if motion_data else 0.0,
                    'stability_score': float(np.mean([m.stability_score for m in motion_data])) if motion_data else 1.0,
                    'dominant_motion_type': self._get_dominant_motion_type(motion_data),
                    'motion_frames': len(motion_data)
                },
                'lighting_conditions': {
                    'overall_brightness': lighting_conditions.overall_brightness,
                    'brightness_uniformity': lighting_conditions.brightness_uniformity,
                    'contrast_ratio': lighting_conditions.contrast_ratio,
                    'lighting_quality': lighting_conditions.lighting_quality,
                    'shadow_areas': lighting_conditions.shadow_areas,
                    'overexposed_areas': lighting_conditions.overexposed_areas
                },
                'camera_position': {
                    'height_estimate': camera_position.height_estimate if camera_position else 6.0,
                    'distance_estimate': camera_position.distance_estimate if camera_position else 20.0,
                    'angle_horizontal': camera_position.angle_horizontal if camera_position else 45.0,
                    'angle_vertical': camera_position.angle_vertical if camera_position else 0.0,
                    'court_coverage': camera_position.court_coverage if camera_position else 50.0,
                    'position_confidence': camera_position.position_confidence if camera_position else 0.5
                }
            }
            
            # Add analysis summary
            analysis_result['analysis_summary'] = self._generate_analysis_summary(analysis_result)
            
            return analysis_result
    
    # Use modified analyzer
    modified_analyzer = ModifiedCameraAnalyzer()
    
    # Perform comprehensive analysis
    try:
        analysis_result = modified_analyzer.analyze_video_camera_characteristics(
            video_path, sample_frames
        )
        
        # Save analysis report
        report_path = Path("data/analysis_reports/camera_analysis.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        modified_analyzer.save_analysis_report(analysis_result, report_path)
        
        # Print summary
        print("\n" + "="*50)
        print("CAMERA ANALYSIS COMPLETE!")
        print("="*50)
        print(f"Video Resolution: {analysis_result['metadata']['resolution']}")
        print(f"Frame Rate: {analysis_result['metadata']['fps']:.1f} fps")
        print(f"Total Frames Analyzed: {analysis_result['metadata']['total_frames']}")
        print(f"Video Quality: {analysis_result['analysis_summary']['overall_quality']}")
        print(f"Camera Setup: {analysis_result['analysis_summary']['camera_setup']}")
        print(f"Camera Stability: {analysis_result['analysis_summary']['camera_stability']}")
        print(f"Lighting Quality: {analysis_result['lighting_conditions']['lighting_quality']}")
        print(f"Estimated Camera Height: {analysis_result['camera_position']['height_estimate']:.1f}m")
        print(f"Estimated Distance: {analysis_result['camera_position']['distance_estimate']:.1f}m")
        print(f"Court Coverage: {analysis_result['camera_position']['court_coverage']:.1f}%")
        print(f"\nFull report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
