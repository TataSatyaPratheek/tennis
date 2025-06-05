"""Camera calibration using OpenCV standard methods"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    reprojection_error: float
    success: bool

class TennisCalibrator:
    """Camera calibration using OpenCV's proven algorithms"""
    
    def __init__(self):
        # Use OpenCV's standard calibration - battle-tested for 20+ years
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Tennis court dimensions (real world) - use official ITF standards
        self.court_3d_points = np.array([
            [0, 0, 0], [23.77, 0, 0],      # Baseline
            [0, 10.97, 0], [23.77, 10.97, 0]  # Service line
        ], dtype=np.float32)
    
    def calibrate_from_frames(self, frames: List[np.ndarray], 
                            keypoints_2d: List[np.ndarray]) -> CalibrationResult:
        """Calibrate camera using OpenCV's standard calibration"""
        
        # Prepare object points (3D) and image points (2D)
        object_points = [self.court_3d_points for _ in frames]
        image_points = [kp.astype(np.float32) for kp in keypoints_2d]
        
        # Use OpenCV's calibrateCamera - don't reinvent this wheel
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, frames[0].shape[:2][::-1], 
            None, None, criteria=self.criteria
        )
        
        return CalibrationResult(
            camera_matrix=mtx,
            distortion_coeffs=dist,
            reprojection_error=ret,
            success=ret < 2.0  # Good calibration threshold
        )
