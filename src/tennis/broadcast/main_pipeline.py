# src/tennis/broadcast/main_pipeline.py
"""Main broadcast pipeline integrating all components"""

from ..analysis.gait_analyzer import TennisGaitAnalyzer # Changed to relative import
from .professional_overlay import ProfessionalBroadcastOverlay # Changed to relative import (sibling module)
from ..calibration.broadcast_calibrator import BroadcastCameraCalibrator # Changed to relative import
from ..tracking.ball_tracker import ProfessionalBallTracker # Changed to relative import
from ..scoring.broadcast_scorer import BroadcastScoreManager # Changed to relative import
import cv2

class TennisBroadcastPipeline:
    def __init__(self):
        self.ball_tracker = ProfessionalBallTracker()
        self.gait_analyzer = TennisGaitAnalyzer()
        self.calibrator = BroadcastCameraCalibrator()
        self.scorer = BroadcastScoreManager()
        self.overlay_system = ProfessionalBroadcastOverlay()
        
    def process_video_to_broadcast(self, input_video_path, output_path):
        """Convert raw tennis video to professional broadcast"""
        
        cap = cv2.VideoCapture(input_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 60.0, (1920, 1080))
        
        frame_count = 0
        print(f"Starting video processing: {input_video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or cannot read the frame.")
                break
                
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}...")
                
            # Track all elements
            ball_data = self.ball_tracker.track_ball_frame(frame)
            player_data = self.gait_analyzer.analyze_players(frame)
            calibration_data = self.calibrator.get_current_calibration()
            
            # Update score based on tracking
            score_update = self.scorer.analyze_point_from_tracking(
                ball_data['trajectory'], player_data
            )
            
            # Create broadcast frame
            tracking_data = {
                'ball': ball_data,
                'players': player_data,
                'calibration': calibration_data,
                'frame_number': frame_count
            }
            
            score_data = self.scorer.get_broadcast_data()
            
            broadcast_frame = self.overlay_system.create_broadcast_frame(
                frame, tracking_data, score_data
            )
            
            out.write(broadcast_frame)
            frame_count += 1
            
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved to: {output_path}")

def main():
    # Initialize broadcast pipeline
    pipeline = TennisBroadcastPipeline()
    
    # Process raw tennis video
    input_video = "/Users/vi/Documents/play/tennis/data/raw_videos/7 YEAR OLD TENNIS PRODIGY- M3.mp4"
    output_video = "data/processed_videos/broadcast_ready_match.mp4"
    
    print("Processing tennis video to broadcast quality...")
    pipeline.process_video_to_broadcast(input_video, output_video)
    
    print("Broadcast-ready video created!")
    print("Features included:")
    print("✓ Professional ball tracking (TrackNet accuracy)")
    print("✓ Player gait analysis and biomechanics") 
    print("✓ Automatic scoring and statistics")
    print("✓ Hawk-Eye style court visualization")
    print("✓ ATP Media style broadcast graphics")
    print("✓ Real-time line calling")
    print("✓ Player movement heatmaps")

if __name__ == "__main__":
    main()
