# src/tennis/calibration/broadcast_calibrator.py
"""Broadcast-quality camera calibration using court models"""
import cv2
import numpy as np
from shapely.geometry import Polygon

class BroadcastCameraCalibrator:
    def __init__(self):
        # ITF official tennis court model (23.77m x 10.97m)
        self.court_3d_model = self._create_official_court_model()
        
        # Multiple calibration methods for robustness
        self.calibration_methods = [
            'court_lines', 'net_posts', 'service_boxes', 'baseline_corners'
        ]
        
    def _create_official_court_model(self):
        """Official ITF tennis court dimensions"""
        return np.array([
            # Main court corners
            [0, 0, 0], [23.77, 0, 0], [23.77, 10.97, 0], [0, 10.97, 0],
            # Net posts
            [11.885, -1.07, 0], [11.885, 11.97, 0],
            # Service lines
            [6.4, 0, 0], [6.4, 10.97, 0], [17.37, 0, 0], [17.37, 10.97, 0],
            # Service boxes
            [6.4, 5.485, 0], [17.37, 5.485, 0],
            # Center marks
            [11.885, 0, 0], [11.885, 10.97, 0]
        ], dtype=np.float32)
    
    def calibrate_broadcast_camera(self, frame, detected_keypoints):
        """Professional camera calibration for broadcast quality"""
        
        # Multi-method calibration
        calibration_results = []
        
        for method in self.calibration_methods:
            result = self._calibrate_with_method(frame, detected_keypoints, method)
            if result['success']:
                calibration_results.append(result)
        
        # Choose best calibration based on reprojection error
        best_calibration = min(calibration_results, 
                             key=lambda x: x['reprojection_error'])
        
        # Refine calibration using bundle adjustment
        refined_calibration = self._refine_calibration(best_calibration)
        
        return refined_calibration
