import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

@dataclass
class CameraMotion:
    """Data class for camera motion analysis results"""
    motion_magnitude: float
    motion_direction: float
    motion_type: str  # 'static', 'pan', 'tilt', 'zoom', 'complex'
    stability_score: float

class CameraMotionAnalyzer:
    """Analyze camera movement and stability in tennis footage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Motion detection thresholds
        self.static_threshold = 2.0
        self.pan_threshold = 10.0
        self.tilt_threshold = 8.0
        self.zoom_threshold = 0.1
    
    def analyze_camera_motion(self, frames: List[np.ndarray]) -> List[CameraMotion]:
        """Analyze camera motion between consecutive frames"""
        
        if len(frames) < 2:
            return []
        
        motion_data = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Initialize feature detector
        orb = cv2.ORB_create(nfeatures=1000)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Detect and match features
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(curr_gray, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Extract matched points
                if len(matches) >= 8:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Estimate homography
                    try:
                        H, mask = cv2.findHomography(src_pts, dst_pts, 
                                                   cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            motion = self._analyze_homography(H, curr_gray.shape)
                            motion_data.append(motion)
                        else:
                            # Fallback to optical flow
                            motion = self._optical_flow_motion(prev_gray, curr_gray)
                            motion_data.append(motion)
                    except:
                        motion = self._optical_flow_motion(prev_gray, curr_gray)
                        motion_data.append(motion)
                else:
                    motion = self._optical_flow_motion(prev_gray, curr_gray)
                    motion_data.append(motion)
            else:
                motion = self._optical_flow_motion(prev_gray, curr_gray)
                motion_data.append(motion)
            
            prev_gray = curr_gray
        
        return motion_data
    
    def _analyze_homography(self, H: np.ndarray, frame_shape: Tuple[int, int]) -> CameraMotion:
        """Analyze camera motion from homography matrix"""
        
        h, w = frame_shape
        
        # Define corner points
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Calculate motion vectors
        motion_vectors = transformed_corners - corners
        
        # Calculate average motion
        avg_motion = np.mean(motion_vectors.reshape(-1, 2), axis=0)
        motion_magnitude = np.linalg.norm(avg_motion)
        motion_direction = np.arctan2(avg_motion[1], avg_motion[0]) * 180 / np.pi
        
        # Determine motion type
        motion_type = self._classify_motion_type(H, motion_magnitude)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(motion_vectors)
        
        return CameraMotion(
            motion_magnitude=float(motion_magnitude),
            motion_direction=float(motion_direction),
            motion_type=motion_type,
            stability_score=float(stability_score)
        )
    
    def _optical_flow_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> CameraMotion:
        """Fixed optical flow motion analysis using Lucas-Kanade method"""
        
        try:
            # Detect good features to track in previous frame
            prev_pts = cv2.goodFeaturesToTrack(
                prev_frame, 
                maxCorners=100, 
                qualityLevel=0.3, 
                minDistance=7, 
                blockSize=7
            )
            
            if prev_pts is None or len(prev_pts) == 0:
                # No points to track, assume static camera
                return CameraMotion(
                    motion_magnitude=0.0,
                    motion_direction=0.0,
                    motion_type='static',
                    stability_score=1.0
                )
            
            # Calculate optical flow using Lucas-Kanade method
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, curr_frame, prev_pts, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Filter valid points
            if curr_pts is None or status is None:
                return CameraMotion(
                    motion_magnitude=0.0,
                    motion_direction=0.0,
                    motion_type='static',
                    stability_score=1.0
                )
            
            # Select good points
            good_prev_pts = prev_pts[status == 1]
            good_curr_pts = curr_pts[status == 1]
            
            if len(good_prev_pts) == 0:
                return CameraMotion(
                    motion_magnitude=0.0,
                    motion_direction=0.0,
                    motion_type='static',
                    stability_score=1.0
                )
            
            # Calculate flow vectors
            flow_vectors = good_curr_pts - good_prev_pts
            magnitudes = np.linalg.norm(flow_vectors, axis=1)
            
            # Calculate average motion
            avg_magnitude = np.mean(magnitudes) if len(magnitudes) > 0 else 0.0
            avg_direction = np.arctan2(
                np.mean(flow_vectors[:, 1]), 
                np.mean(flow_vectors[:, 0])
            ) * 180 / np.pi if len(magnitudes) > 0 else 0.0
            
            # Determine motion type based on magnitude
            if avg_magnitude < self.static_threshold:
                motion_type = 'static'
            elif avg_magnitude < self.pan_threshold:
                motion_type = 'pan'
            else:
                motion_type = 'complex'
            
            # Calculate stability score (inverse of motion variance)
            motion_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0.0
            stability_score = max(0.0, min(1.0, 1.0 - motion_variance / 100.0))
            
            return CameraMotion(
                motion_magnitude=float(avg_magnitude),
                motion_direction=float(avg_direction),
                motion_type=motion_type,
                stability_score=float(stability_score)
            )
            
        except Exception as e:
            self.logger.warning(f"Error in optical flow calculation: {e}")
            # Return default values in case of error
            return CameraMotion(
                motion_magnitude=0.0,
                motion_direction=0.0,
                motion_type='unknown',
                stability_score=0.5
            )
    
    def _classify_motion_type(self, H: np.ndarray, magnitude: float) -> str:
        """Classify motion type based on homography decomposition"""
        
        if magnitude < self.static_threshold:
            return 'static'
        
        # Decompose homography to analyze motion components
        try:
            # Extract rotation and translation components
            _, R, t, _ = cv2.decomposeHomographyMat(H, np.eye(3))
            
            if len(R) > 0:
                rotation_angle = np.abs(cv2.Rodrigues(R[0])[0][2])
                translation_norm = np.linalg.norm(t[0]) if len(t) > 0 else 0
                
                # Classify based on dominant motion
                if rotation_angle > 0.1 and translation_norm > self.pan_threshold:
                    return 'complex'
                elif translation_norm > self.pan_threshold:
                    return 'pan'
                elif rotation_angle > 0.05:
                    return 'tilt'
                else:
                    return 'zoom'
        except:
            pass
        
        return 'complex'
    
    def _calculate_stability_score(self, motion_vectors: np.ndarray) -> float:
        """Calculate stability score based on motion consistency"""
        
        # Calculate variance of motion vectors
        motion_variance = np.var(motion_vectors.reshape(-1, 2), axis=0)
        total_variance = np.sum(motion_variance)
        
        # Convert to stability score (0-1, higher is more stable)
        stability_score = max(0, min(1, 1.0 - total_variance / 1000.0))
        
        return stability_score
