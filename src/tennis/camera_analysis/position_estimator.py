import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass

@dataclass
class CameraPosition:
    """Data class for camera position estimation"""
    height_estimate: float  # estimated height above court
    distance_estimate: float  # estimated distance from court center
    angle_horizontal: float  # horizontal viewing angle
    angle_vertical: float  # vertical viewing angle (tilt)
    court_coverage: float  # percentage of court visible
    position_confidence: float  # confidence in estimation (0-1)

class CameraPositionEstimator:
    """Estimate camera position and orientation for tennis court footage"""[2]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Tennis court dimensions (ITF standard)
        self.court_length = 23.77  # meters
        self.court_width = 10.97   # meters (doubles)
        self.singles_width = 8.23  # meters
        self.net_height = 0.914    # meters at center
    
    def estimate_camera_position(self, frame: np.ndarray, 
                               court_lines: Optional[List] = None) -> CameraPosition:
        """Estimate camera position based on court geometry"""[2]
        
        if court_lines is None:
            court_lines = self._detect_court_lines(frame)
        
        # Estimate position parameters
        height_estimate = self._estimate_camera_height(frame, court_lines)
        distance_estimate = self._estimate_distance_from_court(frame, court_lines)
        angle_horizontal = self._estimate_horizontal_angle(frame, court_lines)
        angle_vertical = self._estimate_vertical_angle(frame, court_lines)
        court_coverage = self._calculate_court_coverage(frame, court_lines)
        
        # Calculate confidence based on line detection quality
        position_confidence = self._calculate_position_confidence(court_lines)
        
        return CameraPosition(
            height_estimate=height_estimate,
            distance_estimate=distance_estimate,
            angle_horizontal=angle_horizontal,
            angle_vertical=angle_vertical,
            court_coverage=court_coverage,
            position_confidence=position_confidence
        )
    
    def _detect_court_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect tennis court lines for position estimation"""[2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Filter and group lines
        filtered_lines = self._filter_court_lines(lines, frame.shape)
        
        return filtered_lines
    
    def _filter_court_lines(self, lines: np.ndarray, frame_shape: Tuple[int, int]) -> List:
        """Filter detected lines to identify court-specific lines"""[2]
        
        h, w = frame_shape[:2]
        court_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            # Filter based on tennis court line characteristics
            # Horizontal lines (baselines, service lines)
            if abs(angle) < 15 or abs(angle) > 165:
                if length > w * 0.2:  # Minimum length threshold
                    court_lines.append((x1, y1, x2, y2))
            
            # Vertical lines (sidelines, center service line)
            elif 75 < abs(angle) < 105:
                if length > h * 0.3:  # Minimum length threshold
                    court_lines.append((x1, y1, x2, y2))
        
        return court_lines
    
    def _estimate_camera_height(self, frame: np.ndarray, court_lines: List) -> float:
        """Estimate camera height based on court perspective"""[2]
        
        if not court_lines:
            return 6.0  # Default estimate
        
        h, w = frame.shape[:2]
        
        # Find horizontal lines (likely baselines or service lines)
        horizontal_lines = []
        for x1, y1, x2, y2 in court_lines:
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if abs(angle) < 15 or abs(angle) > 165:
                # Calculate line position relative to frame
                line_y = (y1 + y2) / 2
                relative_position = line_y / h
                horizontal_lines.append(relative_position)
        
        if horizontal_lines:
            # Camera height estimation based on perspective
            # Lower lines in frame indicate higher camera position
            avg_line_position = np.mean(horizontal_lines)
            
            # Empirical relationship for tennis court cameras
            if avg_line_position > 0.7:  # Lines in lower part
                height_estimate = 3.0 + (avg_line_position - 0.7) * 10
            else:
                height_estimate = 3.0 + (0.7 - avg_line_position) * 5
            
            return min(15.0, max(2.0, height_estimate))
        
        return 6.0  # Default height
    
    def _estimate_distance_from_court(self, frame: np.ndarray, court_lines: List) -> float:
        """Estimate camera distance from court center"""[2]
        
        if not court_lines:
            return 20.0  # Default estimate
        
        h, w = frame.shape[:2]
        
        # Count lines in different regions
        left_lines = sum(1 for x1, y1, x2, y2 in court_lines 
                        if max(x1, x2) < w//3)
        right_lines = sum(1 for x1, y1, x2, y2 in court_lines 
                         if min(x1, x2) > 2*w//3)
        center_lines = sum(1 for x1, y1, x2, y2 in court_lines 
                          if w//3 <= min(x1, x2) and max(x1, x2) <= 2*w//3)
        
        # Estimate distance based on court visibility distribution
        total_lines = len(court_lines)
        if total_lines > 0:
            center_ratio = center_lines / total_lines
            
            # More center lines suggest closer position
            if center_ratio > 0.6:
                distance_estimate = 10.0 + (1.0 - center_ratio) * 30
            else:
                distance_estimate = 20.0 + (0.6 - center_ratio) * 40
            
            return min(100.0, max(5.0, distance_estimate))
        
        return 20.0
    
    def _estimate_horizontal_angle(self, frame: np.ndarray, court_lines: List) -> float:
        """Estimate horizontal viewing angle"""
        
        h, w = frame.shape[:2]
        
        # Analyze line convergence for perspective estimation
        if len(court_lines) >= 2:
            # Calculate average line orientations
            angles = []
            for x1, y1, x2, y2 in court_lines:
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                angles.append(angle)
            
            angle_std = np.std(angles)
            
            # Higher angle variation suggests wider viewing angle
            horizontal_angle = 30.0 + angle_std * 0.5
            return min(90.0, max(15.0, horizontal_angle))
        
        return 45.0  # Default angle
    
    def _estimate_vertical_angle(self, frame: np.ndarray, court_lines: List) -> float:
        """Estimate vertical viewing angle (camera tilt)"""[2]
        
        h, w = frame.shape[:2]
        
        # Find dominant horizontal line position
        horizontal_positions = []
        for x1, y1, x2, y2 in court_lines:
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if abs(angle) < 15 or abs(angle) > 165:
                avg_y = (y1 + y2) / 2
                horizontal_positions.append(avg_y / h)
        
        if horizontal_positions:
            avg_position = np.mean(horizontal_positions)
            
            # Convert position to tilt angle
            # Lower position in frame = higher tilt angle
            tilt_angle = (avg_position - 0.5) * 60  # -30 to +30 degrees
            return tilt_angle
        
        return 0.0  # No tilt
    
    def _calculate_court_coverage(self, frame: np.ndarray, court_lines: List) -> float:
        """Calculate percentage of court visible in frame"""
        
        if not court_lines:
            return 50.0  # Default estimate
        
        # Estimate court area based on detected lines
        line_coverage = len(court_lines) / 12.0  # Assuming ~12 major court lines
        court_coverage = min(100.0, line_coverage * 100)
        
        return court_coverage
    
    def _calculate_position_confidence(self, court_lines: List) -> float:
        """Calculate confidence in position estimation"""
        
        # Confidence based on number and quality of detected lines
        if len(court_lines) >= 8:
            return 0.9
        elif len(court_lines) >= 5:
            return 0.7
        elif len(court_lines) >= 3:
            return 0.5
        else:
            return 0.3
