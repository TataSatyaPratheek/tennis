import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class CourtFeatures:
    """Data class for detected court features"""
    lines: List[Tuple[int, int, int, int]]
    intersections: List[Tuple[int, int]]
    court_corners: List[Tuple[int, int]]
    service_lines: List[Tuple[int, int, int, int]]
    baselines: List[Tuple[int, int, int, int]]
    sidelines: List[Tuple[int, int, int, int]]
    center_line: Optional[Tuple[int, int, int, int]]
    net_line: Optional[Tuple[int, int, int, int]]
    feature_count: int
    detection_confidence: float

class TennisCourtDetector:
    """Detect tennis court features for calibration frame selection"""[3]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Tennis court standard dimensions (ITF regulations)
        self.court_length = 23.77  # meters
        self.court_width_doubles = 10.97  # meters
        self.court_width_singles = 8.23   # meters
        self.service_box_length = 6.40    # meters
        self.service_box_width = 4.115    # meters (singles width / 2)
        
        # Detection parameters
        self.line_detection_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
        self.angle_tolerance = 15  # degrees
    
    def detect_court_features(self, frame: np.ndarray) -> CourtFeatures:
        """Detect tennis court features for calibration assessment"""[3]
        
        if frame is None or frame.size == 0:
            return self._create_empty_features()
        
        # Preprocess frame for line detection
        preprocessed = self._preprocess_for_line_detection(frame)
        
        # Detect all lines in the frame
        all_lines = self._detect_lines(preprocessed)
        
        if not all_lines:
            return self._create_empty_features()
        
        # Classify lines into court features
        classified_lines = self._classify_court_lines(all_lines, frame.shape)
        
        # Find line intersections
        intersections = self._find_line_intersections(classified_lines['all_lines'])
        
        # Detect court corners
        court_corners = self._detect_court_corners(intersections, frame.shape)
        
        # Calculate feature count and confidence
        feature_count = self._count_features(classified_lines)
        detection_confidence = self._calculate_detection_confidence(classified_lines, intersections)
        
        return CourtFeatures(
            lines=classified_lines['all_lines'],
            intersections=intersections,
            court_corners=court_corners,
            service_lines=classified_lines['service_lines'],
            baselines=classified_lines['baselines'],
            sidelines=classified_lines['sidelines'],
            center_line=classified_lines['center_line'],
            net_line=classified_lines['net_line'],
            feature_count=feature_count,
            detection_confidence=detection_confidence
        )
    
    def _preprocess_for_line_detection(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame to enhance line detection"""[3]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply edge detection
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        
        return edges
    
    def _detect_lines(self, edge_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lines using Hough transform"""[3]
        
        lines = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi/180,
            threshold=self.line_detection_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
        
        # Filter and merge similar lines
        filtered_lines = self._filter_and_merge_lines(line_list)
        
        return filtered_lines
    
    def _filter_and_merge_lines(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Filter out noise and merge similar lines"""[3]
        
        if not lines:
            return []
        
        # Group lines by orientation (horizontal vs vertical)
        horizontal_lines = []
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < self.angle_tolerance or abs(angle) > 180 - self.angle_tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 90 - self.angle_tolerance < abs(angle) < 90 + self.angle_tolerance:
                vertical_lines.append((x1, y1, x2, y2))
        
        # Merge similar lines within each group
        merged_horizontal = self._merge_similar_lines(horizontal_lines, is_horizontal=True)
        merged_vertical = self._merge_similar_lines(vertical_lines, is_horizontal=False)
        
        return merged_horizontal + merged_vertical
    
    def _merge_similar_lines(self, lines: List[Tuple[int, int, int, int]], is_horizontal: bool) -> List[Tuple[int, int, int, int]]:
        """Merge lines that are close and parallel"""
        
        if len(lines) < 2:
            return lines
        
        merged_lines = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            # Find lines to merge with current line
            group = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue
                
                if self._should_merge_lines(line1, line2, is_horizontal):
                    group.append(line2)
                    used[j] = True
            
            # Merge the group into a single line
            if len(group) == 1:
                merged_lines.append(group[0])
            else:
                merged_line = self._merge_line_group(group, is_horizontal)
                merged_lines.append(merged_line)
        
        return merged_lines
    
    def _should_merge_lines(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int], is_horizontal: bool) -> bool:
        """Determine if two lines should be merged"""
        
        x1_1, y1_1, x2_1, y2_1 = line1
        x1_2, y1_2, x2_2, y2_2 = line2
        
        # Calculate distance between lines
        if is_horizontal:
            # For horizontal lines, check y-coordinate distance
            avg_y1 = (y1_1 + y2_1) / 2
            avg_y2 = (y1_2 + y2_2) / 2
            distance = abs(avg_y1 - avg_y2)
        else:
            # For vertical lines, check x-coordinate distance
            avg_x1 = (x1_1 + x2_1) / 2
            avg_x2 = (x1_2 + x2_2) / 2
            distance = abs(avg_x1 - avg_x2)
        
        # Merge if distance is small (within 20 pixels)
        return distance < 20
    
    def _merge_line_group(self, lines: List[Tuple[int, int, int, int]], is_horizontal: bool) -> Tuple[int, int, int, int]:
        """Merge a group of similar lines into one"""
        
        if is_horizontal:
            # For horizontal lines, find the extreme x coordinates
            min_x = min(min(x1, x2) for x1, y1, x2, y2 in lines)
            max_x = max(max(x1, x2) for x1, y1, x2, y2 in lines)
            avg_y = int(np.mean([y1 + y2 for x1, y1, x2, y2 in lines]) / 2)
            return (min_x, avg_y, max_x, avg_y)
        else:
            # For vertical lines, find the extreme y coordinates
            min_y = min(min(y1, y2) for x1, y1, x2, y2 in lines)
            max_y = max(max(y1, y2) for x1, y1, x2, y2 in lines)
            avg_x = int(np.mean([x1 + x2 for x1, y1, x2, y2 in lines]) / 2)
            return (avg_x, min_y, avg_x, max_y)
    
    def _classify_court_lines(self, lines: List[Tuple[int, int, int, int]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Classify detected lines into specific court features"""[3]
        
        h, w = frame_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        # Separate lines by orientation
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < self.angle_tolerance or abs(angle) > 180 - self.angle_tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 90 - self.angle_tolerance < abs(angle) < 90 + self.angle_tolerance:
                vertical_lines.append((x1, y1, x2, y2))
        
        # Sort horizontal lines by y-coordinate
        horizontal_lines.sort(key=lambda line: (line[1] + line[3]) / 2)
        
        # Sort vertical lines by x-coordinate
        vertical_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        
        # Classify lines based on position and length
        baselines = []
        service_lines = []
        sidelines = []
        center_line = None
        net_line = None
        
        # Identify baselines (top and bottom horizontal lines)
        if len(horizontal_lines) >= 2:
            baselines = [horizontal_lines[0], horizontal_lines[-1]]
        
        # Identify service lines (middle horizontal lines)
        if len(horizontal_lines) >= 4:
            middle_lines = horizontal_lines[1:-1]
            service_lines = middle_lines[:2] if len(middle_lines) >= 2 else middle_lines
        
        # Identify sidelines (left and right vertical lines)
        if len(vertical_lines) >= 2:
            sidelines = [vertical_lines[0], vertical_lines[-1]]
        
        # Identify center line (middle vertical line)
        if len(vertical_lines) >= 3:
            center_idx = len(vertical_lines) // 2
            center_line = vertical_lines[center_idx]
        
        # Identify net line (horizontal line in the middle third of the frame)
        middle_y = h // 2
        net_candidates = [line for line in horizontal_lines 
                         if abs((line[1] + line[3]) / 2 - middle_y) < h // 6]
        if net_candidates:
            net_line = net_candidates[0]
        
        return {
            'all_lines': lines,
            'baselines': baselines,
            'service_lines': service_lines,
            'sidelines': sidelines,
            'center_line': center_line,
            'net_line': net_line
        }
    
    def _find_line_intersections(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """Find intersection points between lines"""[2]
        
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                intersection = self._calculate_line_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)
        
        # Remove duplicate intersections
        unique_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing_point in unique_intersections:
                if abs(point[0] - existing_point[0]) < 10 and abs(point[1] - existing_point[1]) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(point)
        
        return unique_intersections
    
    def _calculate_line_intersection(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Calculate intersection point of two lines"""[2]
        
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate the direction vectors
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        # Calculate intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # Calculate the intersection coordinates
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))
    
    def _detect_court_corners(self, intersections: List[Tuple[int, int]], frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Detect the main corners of the tennis court"""[3]
        
        if len(intersections) < 4:
            return intersections
        
        h, w = frame_shape[:2]
        
        # Find corners in each quadrant of the frame
        corners = []
        quadrants = [
            (0, w//2, 0, h//2),      # Top-left
            (w//2, w, 0, h//2),      # Top-right
            (0, w//2, h//2, h),      # Bottom-left
            (w//2, w, h//2, h)       # Bottom-right
        ]
        
        for x_min, x_max, y_min, y_max in quadrants:
            quadrant_points = [
                (x, y) for x, y in intersections
                if x_min <= x < x_max and y_min <= y < y_max
            ]
            
            if quadrant_points:
                # Find the point closest to the quadrant corner
                if x_min == 0 and y_min == 0:  # Top-left
                    corner = min(quadrant_points, key=lambda p: p[0] + p[1])
                elif x_min == w//2 and y_min == 0:  # Top-right
                    corner = min(quadrant_points, key=lambda p: (w - p[0]) + p[1])
                elif x_min == 0 and y_min == h//2:  # Bottom-left
                    corner = min(quadrant_points, key=lambda p: p[0] + (h - p[1]))
                else:  # Bottom-right
                    corner = min(quadrant_points, key=lambda p: (w - p[0]) + (h - p[1]))
                
                corners.append(corner)
        
        return corners
    
    def _count_features(self, classified_lines: Dict[str, Any]) -> int:
        """Count the number of detected court features"""
        
        count = 0
        count += len(classified_lines['baselines'])
        count += len(classified_lines['service_lines'])
        count += len(classified_lines['sidelines'])
        count += 1 if classified_lines['center_line'] else 0
        count += 1 if classified_lines['net_line'] else 0
        
        return count
    
    def _calculate_detection_confidence(self, classified_lines: Dict[str, Any], intersections: List[Tuple[int, int]]) -> float:
        """Calculate confidence in court detection"""[3]
        
        # Expected features for a complete tennis court
        expected_features = {
            'baselines': 2,
            'service_lines': 2,
            'sidelines': 2,
            'center_line': 1,
            'net_line': 1
        }
        
        total_expected = sum(expected_features.values())
        detected_features = self._count_features(classified_lines)
        
        # Base confidence from feature detection
        feature_confidence = detected_features / total_expected
        
        # Bonus for intersections (indicates good line detection)
        intersection_bonus = min(0.2, len(intersections) / 20 * 0.2)
        
        # Total confidence
        confidence = min(1.0, feature_confidence + intersection_bonus)
        
        return float(confidence)
    
    def _create_empty_features(self) -> CourtFeatures:
        """Create empty court features for failed detection"""
        
        return CourtFeatures(
            lines=[],
            intersections=[],
            court_corners=[],
            service_lines=[],
            baselines=[],
            sidelines=[],
            center_line=None,
            net_line=None,
            feature_count=0,
            detection_confidence=0.0
        )
