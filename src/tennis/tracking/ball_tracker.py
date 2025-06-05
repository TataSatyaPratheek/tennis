# src/tennis/tracking/ball_tracker.py
"""Professional ball tracking using TrackNet + YOLO ensemble"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class ProfessionalBallTracker:
    def __init__(self):
        # Use fine-tuned YOLO for tennis balls (as shown in search results)
        self.yolo_ball = YOLO('models/tennis_ball_yolov5.pt')  # Fine-tuned model
        
        # TrackNet-style heatmap detector
        self.tracknet_model = self._load_tracknet_model()
        
        # Ball trajectory buffer for smooth prediction
        self.ball_positions = deque(maxlen=10)
        self.velocity_buffer = deque(maxlen=5)
        
    def track_ball_frame(self, frame):
        """Multi-method ball tracking for reliability"""
        
        # Method 1: YOLO detection
        yolo_detection = self._detect_ball_yolo(frame)
        
        # Method 2: TrackNet heatmap
        heatmap_detection = self._detect_ball_tracknet(frame)
        
        # Method 3: Temporal consistency
        predicted_position = self._predict_ball_position()
        
        # Ensemble decision
        final_position = self._ensemble_ball_position(
            yolo_detection, heatmap_detection, predicted_position
        )
        
        if final_position:
            self.ball_positions.append(final_position)
            self._update_velocity()
            
        return {
            'position': final_position,
            'confidence': self._calculate_confidence(),
            'velocity': self._get_current_velocity(),
            'trajectory': list(self.ball_positions)
        }
    
    def _detect_ball_tracknet(self, frame):
        """TrackNet-style heatmap detection"""
        # Resize to 640x360 as per TrackNet paper
        resized = cv2.resize(frame, (640, 360))
        
        # Generate detection heatmap
        with torch.no_grad():
            heatmap = self.tracknet_model(resized)
            
        # Find peak in heatmap
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Scale back to original resolution
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 360
        
        return (int(x * scale_x), int(y * scale_y))
