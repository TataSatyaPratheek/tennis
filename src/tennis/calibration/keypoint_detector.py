import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

from .pattern_manager import CalibrationPatternManager
from .pattern_design import CourtKeypoint

@dataclass
class DetectedKeypoint:
    """Data class for detected tennis court keypoints"""
    image_coords: Tuple[float, float]
    world_coords: Tuple[float, float, float]
    keypoint_name: str
    confidence: float
    detection_method: str

@dataclass
class KeypointDetectionResult:
    """Result of keypoint detection in a frame"""
    frame_id: str
    detected_keypoints: List[DetectedKeypoint]
    detection_count: int
    avg_confidence: float
    detection_quality: str
    debug_info: Dict[str, Any]  # Added for debugging

class TennisCourtKeypointDetector:
    """Enhanced tennis court keypoint detector with better robustness"""
    
    def __init__(self, pattern_manager: CalibrationPatternManager):
        self.pattern_manager = pattern_manager
        self.logger = logging.getLogger(__name__)
        
        # Relaxed detection parameters for better detection rate
        self.line_detection_params = {
            'threshold': 50,  # Lowered from 80
            'min_line_length': 30,  # Lowered from 50
            'max_line_gap': 15,  # Increased from 10
            'canny_low': 30,  # Lowered from 50
            'canny_high': 100  # Lowered from 150
        }
        
        # Relaxed confidence thresholds
        self.confidence_thresholds = {
            'excellent': 0.7,  # Lowered from 0.8
            'good': 0.5,       # Lowered from 0.6
            'fair': 0.3,       # Lowered from 0.4
            'poor': 0.0
        }
        
        # Minimum keypoints for each quality level
        self.min_keypoints = {
            'excellent': 6,
            'good': 4,    # Lowered from 5
            'fair': 3,    # Lowered from 4
            'poor': 0
        }
    
    def detect_keypoints_in_frame(self, frame: np.ndarray, frame_id: str, 
                                pattern_type: str = 'minimal') -> KeypointDetectionResult:
        """Enhanced keypoint detection with multiple fallback methods"""
        
        if frame is None or frame.size == 0:
            return self._create_empty_result(frame_id)
        
        debug_info = {'methods_tried': [], 'line_counts': {}, 'intersection_counts': {}}
        
        try:
            # Method 1: Standard detection
            result = self._try_standard_detection(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 3:  # Lowered threshold
                return result
            
            # Method 2: Enhanced preprocessing
            result = self._try_enhanced_detection(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 3:
                return result
            
            # Method 3: Adaptive thresholding
            result = self._try_adaptive_detection(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 3:
                return result
            
            # Method 4: Manual feature estimation (fallback)
            result = self._try_estimation_fallback(frame, frame_id, pattern_type, debug_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting keypoints in frame {frame_id}: {e}")
            return self._create_empty_result(frame_id, debug_info)
    
    def _try_standard_detection(self, frame: np.ndarray, frame_id: str, 
                              pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Try standard detection method"""
        
        debug_info['methods_tried'].append('standard')
        
        pattern = self.pattern_manager.get_pattern(pattern_type)
        preprocessed = self._preprocess_frame(frame)
        lines = self._detect_court_lines(preprocessed)
        
        debug_info['line_counts']['standard'] = len(lines)
        
        if not lines:
            return self._create_empty_result(frame_id, debug_info)
        
        intersections = self._find_line_intersections(lines)
        debug_info['intersection_counts']['standard'] = len(intersections)
        
        court_features = self._detect_court_features(lines, intersections, frame.shape)
        detected_keypoints = self._match_features_to_pattern(court_features, pattern, frame.shape)
        
        avg_confidence = np.mean([kp.confidence for kp in detected_keypoints]) if detected_keypoints else 0.0
        detection_quality = self._determine_detection_quality(avg_confidence, len(detected_keypoints))
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=detected_keypoints,
            detection_count=len(detected_keypoints),
            avg_confidence=avg_confidence,
            detection_quality=detection_quality,
            debug_info=debug_info
        )
    
    def _try_enhanced_detection(self, frame: np.ndarray, frame_id: str,
                              pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Try enhanced preprocessing method"""
        
        debug_info['methods_tried'].append('enhanced')
        
        pattern = self.pattern_manager.get_pattern(pattern_type)
        
        # Enhanced preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple enhancement techniques
        enhanced = self._apply_enhancement_pipeline(gray)
        
        # More aggressive line detection
        lines = self._detect_court_lines_aggressive(enhanced)
        debug_info['line_counts']['enhanced'] = len(lines)
        
        if not lines:
            return self._create_empty_result(frame_id, debug_info)
        
        intersections = self._find_line_intersections(lines)
        debug_info['intersection_counts']['enhanced'] = len(intersections)
        
        court_features = self._detect_court_features(lines, intersections, frame.shape)
        detected_keypoints = self._match_features_to_pattern_relaxed(court_features, pattern, frame.shape)
        
        avg_confidence = np.mean([kp.confidence for kp in detected_keypoints]) if detected_keypoints else 0.0
        detection_quality = self._determine_detection_quality(avg_confidence, len(detected_keypoints))
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=detected_keypoints,
            detection_count=len(detected_keypoints),
            avg_confidence=avg_confidence,
            detection_quality=detection_quality,
            debug_info=debug_info
        )
    
    def _try_adaptive_detection(self, frame: np.ndarray, frame_id: str,
                              pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Try adaptive threshold method"""
        
        debug_info['methods_tried'].append('adaptive')
        
        pattern = self.pattern_manager.get_pattern(pattern_type)
        
        # Adaptive preprocessing based on frame characteristics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze frame brightness to adapt parameters
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:  # Dark frame
            canny_low, canny_high = 20, 80
            threshold = 30
        elif mean_brightness > 180:  # Bright frame
            canny_low, canny_high = 80, 200
            threshold = 80
        else:  # Normal frame
            canny_low, canny_high = 40, 120
            threshold = 50
        
        # Apply adaptive enhancement
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive edge detection
        edges = cv2.Canny(filtered, canny_low, canny_high)
        
        # Adaptive line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=25,
            maxLineGap=20
        )
        
        if lines is None:
            lines = []
        else:
            lines = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]
        
        debug_info['line_counts']['adaptive'] = len(lines)
        
        if not lines:
            return self._create_empty_result(frame_id, debug_info)
        
        # Filter and process lines
        filtered_lines = self._filter_court_lines_relaxed(lines, frame.shape[:2])
        intersections = self._find_line_intersections(filtered_lines)
        debug_info['intersection_counts']['adaptive'] = len(intersections)
        
        court_features = self._detect_court_features(filtered_lines, intersections, frame.shape)
        detected_keypoints = self._match_features_to_pattern_relaxed(court_features, pattern, frame.shape)
        
        avg_confidence = np.mean([kp.confidence for kp in detected_keypoints]) if detected_keypoints else 0.0
        detection_quality = self._determine_detection_quality(avg_confidence, len(detected_keypoints))
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=detected_keypoints,
            detection_count=len(detected_keypoints),
            avg_confidence=avg_confidence,
            detection_quality=detection_quality,
            debug_info=debug_info
        )
    
    def _try_estimation_fallback(self, frame: np.ndarray, frame_id: str,
                               pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Fallback method using geometric estimation"""
        
        debug_info['methods_tried'].append('estimation_fallback')
        
        pattern = self.pattern_manager.get_pattern(pattern_type)
        h, w = frame.shape[:2]
        
        # Create estimated keypoints based on typical tennis court layout
        estimated_keypoints = []
        
        # Estimate court corners based on image geometry
        margin_x = w * 0.1
        margin_y = h * 0.1
        
        corner_estimates = [
            ("bottom_left_corner", (margin_x, h - margin_y), 0.3),
            ("bottom_right_corner", (w - margin_x, h - margin_y), 0.3),
            ("top_left_corner", (margin_x, margin_y), 0.3),
            ("top_right_corner", (w - margin_x, margin_y), 0.3)
        ]
        
        for corner_name, image_pos, confidence in corner_estimates:
            keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
            if keypoint:
                estimated_keypoints.append(DetectedKeypoint(
                    image_coords=image_pos,
                    world_coords=keypoint.world_coords,
                    keypoint_name=corner_name,
                    confidence=confidence,
                    detection_method='geometric_estimation'
                ))
        
        debug_info['line_counts']['estimation'] = 0
        debug_info['intersection_counts']['estimation'] = 0
        
        avg_confidence = np.mean([kp.confidence for kp in estimated_keypoints]) if estimated_keypoints else 0.0
        detection_quality = self._determine_detection_quality(avg_confidence, len(estimated_keypoints))
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=estimated_keypoints,
            detection_count=len(estimated_keypoints),
            avg_confidence=avg_confidence,
            detection_quality=detection_quality,
            debug_info=debug_info
        )
    
    def _apply_enhancement_pipeline(self, gray: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement pipeline"""
        
        # 1. Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. Edge-preserving smoothing
        smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        return smooth
    
    def _detect_court_lines_aggressive(self, preprocessed: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """More aggressive line detection with multiple methods"""
        
        lines_all = []
        
        # Method 1: Standard Canny + Hough
        edges1 = cv2.Canny(preprocessed, 20, 80)
        lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 30, minLineLength=20, maxLineGap=20)
        if lines1 is not None:
            lines_all.extend([(x1, y1, x2, y2) for line in lines1 for x1, y1, x2, y2 in line])
        
        # Method 2: Higher threshold Canny + Hough
        edges2 = cv2.Canny(preprocessed, 50, 150)
        lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 50, minLineLength=30, maxLineGap=15)
        if lines2 is not None:
            lines_all.extend([(x1, y1, x2, y2) for line in lines2 for x1, y1, x2, y2 in line])
        
        # Method 3: Sobel edge detection + Hough
        sobelx = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        lines3 = cv2.HoughLinesP(sobel, 1, np.pi/180, 40, minLineLength=25, maxLineGap=18)
        if lines3 is not None:
            lines_all.extend([(x1, y1, x2, y2) for line in lines3 for x1, y1, x2, y2 in line])
        
        # Remove duplicates
        unique_lines = self._remove_duplicate_lines(lines_all)
        
        return unique_lines
    
    def _remove_duplicate_lines(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate and very similar lines"""
        
        if not lines:
            return []
        
        unique_lines = []
        threshold = 20  # pixels
        
        for line in lines:
            x1, y1, x2, y2 = line
            is_duplicate = False
            
            for existing_line in unique_lines:
                ex1, ey1, ex2, ey2 = existing_line
                
                # Calculate distance between line endpoints
                dist1 = np.sqrt((x1 - ex1)**2 + (y1 - ey1)**2)
                dist2 = np.sqrt((x2 - ex2)**2 + (y2 - ey2)**2)
                dist3 = np.sqrt((x1 - ex2)**2 + (y1 - ey2)**2)
                dist4 = np.sqrt((x2 - ex1)**2 + (y2 - ey1)**2)
                
                min_dist = min(dist1 + dist2, dist3 + dist4)
                
                if min_dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines
    
    def _filter_court_lines_relaxed(self, lines: List[Tuple[int, int, int, int]], 
                                   frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Relaxed line filtering for better detection"""
        
        if not lines:
            return []
        
        h, w = frame_shape
        filtered_lines = []
        
        for x1, y1, x2, y2 in lines:
            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Relaxed length filter
            if length < min(w, h) * 0.05:  # Very short lines
                continue
            
            # Keep lines that are roughly horizontal or vertical
            if (abs(angle) < 25 or abs(angle) > 155 or  # Horizontal-ish
                (65 < abs(angle) < 115)):  # Vertical-ish
                filtered_lines.append((x1, y1, x2, y2))
        
        return filtered_lines
    
    def _match_features_to_pattern_relaxed(self, court_features: Dict[str, Any], 
                                         pattern: Any, frame_shape: Tuple[int, ...]) -> List[DetectedKeypoint]:
        """Relaxed feature matching with lower confidence requirements"""
        
        detected_keypoints = []
        h, w = frame_shape[:2]
        
        # Extract available features
        corners = court_features.get('court_corners', [])
        horizontal_lines = court_features.get('horizontal_lines', [])
        vertical_lines = court_features.get('vertical_lines', [])
        intersections = court_features.get('intersections', [])
        
        # Strategy 1: Use any available corners
        if len(corners) >= 2:
            # Sort corners by position
            sorted_corners = sorted(corners, key=lambda p: (p[1], p[0]))  # Sort by y, then x
            
            corner_names = ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner']
            
            for i, corner_name in enumerate(corner_names):
                if i < len(sorted_corners):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
                    if keypoint:
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=sorted_corners[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=corner_name,
                            confidence=0.6,  # Medium confidence
                            detection_method='relaxed_corner_detection'
                        ))
        
        # Strategy 2: Estimate corners from line intersections
        elif len(intersections) >= 2:
            # Use intersections as corner approximations
            sorted_intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
            
            corner_names = ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner']
            
            for i, corner_name in enumerate(corner_names):
                if i < len(sorted_intersections):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
                    if keypoint:
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=sorted_intersections[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=corner_name,
                            confidence=0.5,  # Lower confidence
                            detection_method='intersection_approximation'
                        ))
        
        # Strategy 3: Estimate from line endpoints
        elif len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Use line endpoints to estimate corners
            h_line1, h_line2 = horizontal_lines[0], horizontal_lines[-1]
            v_line1, v_line2 = vertical_lines[0], vertical_lines[-1]
            
            # Estimate corner positions
            estimated_corners = [
                (v_line1[0], h_line1[1]),  # Bottom-left
                (v_line2[0], h_line1[1]),  # Bottom-right
                (v_line1[0], h_line2[1]),  # Top-left
                (v_line2[0], h_line2[1])   # Top-right
            ]
            
            corner_names = ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner']
            
            for i, corner_name in enumerate(corner_names):
                if i < len(estimated_corners):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
                    if keypoint:
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=estimated_corners[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=corner_name,
                            confidence=0.4,  # Lower confidence
                            detection_method='line_endpoint_estimation'
                        ))
        
        return detected_keypoints
    
    def _determine_detection_quality(self, avg_confidence: float, keypoint_count: int) -> str:
        """Determine detection quality with relaxed thresholds"""
        
        for quality in ['excellent', 'good', 'fair', 'poor']:
            if (avg_confidence >= self.confidence_thresholds[quality] and 
                keypoint_count >= self.min_keypoints[quality]):
                return quality
        
        return 'poor'
    
    def _create_empty_result(self, frame_id: str, debug_info: Dict = None) -> KeypointDetectionResult:
        """Create empty detection result"""
        
        if debug_info is None:
            debug_info = {'methods_tried': [], 'line_counts': {}, 'intersection_counts': {}}
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=[],
            detection_count=0,
            avg_confidence=0.0,
            detection_quality='poor',
            debug_info=debug_info
        )
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal line detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    def _detect_court_lines(self, preprocessed_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect court lines using edge detection and Hough transform"""
        edges = cv2.Canny(preprocessed_frame, self.line_detection_params['canny_low'],
                         self.line_detection_params['canny_high'], apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                               threshold=self.line_detection_params['threshold'],
                               minLineLength=self.line_detection_params['min_line_length'],
                               maxLineGap=self.line_detection_params['max_line_gap'])
        
        if lines is None:
            return []
        
        line_list = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]
        return self._filter_court_lines_relaxed(line_list, preprocessed_frame.shape)
    
    def _find_line_intersections(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """Find intersection points between detected lines"""
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                intersection = self._calculate_line_intersection(line1, line2)
                if intersection:
                    x, y = intersection
                    if 0 <= x <= 10000 and 0 <= y <= 10000:
                        intersections.append((x, y))
        
        return self._remove_duplicate_points(intersections, threshold=15)  # Increased threshold
    
    def _calculate_line_intersection(self, line1: Tuple[int, int, int, int], 
                                   line2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if -1.0 <= t <= 2.0 and -1.0 <= u <= 2.0:  # More relaxed bounds
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            return (x, y)
        
        return None
    
    def _remove_duplicate_points(self, points: List[Tuple[int, int]], 
                               threshold: int = 15) -> List[Tuple[int, int]]:
        """Remove duplicate points within threshold distance"""
        if not points:
            return []
        
        unique_points = []
        for point in points:
            is_duplicate = False
            for existing_point in unique_points:
                distance = np.sqrt((point[0] - existing_point[0])**2 + 
                                 (point[1] - existing_point[1])**2)
                if distance < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points
    
    def _detect_court_features(self, lines: List[Tuple[int, int, int, int]], 
                             intersections: List[Tuple[int, int]], 
                             frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Detect specific court features from lines and intersections"""
        h, w = frame_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 25 or abs(angle) > 155:  # More relaxed angle threshold
                horizontal_lines.append(line)
            elif 65 < abs(angle) < 115:  # More relaxed angle threshold
                vertical_lines.append(line)
        
        horizontal_lines.sort(key=lambda line: (line[1] + line[3]) / 2)
        vertical_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        
        # Find corners from intersections using relaxed criteria
        court_corners = self._find_court_corners_relaxed(intersections, frame_shape)
        
        return {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'intersections': intersections,
            'baselines': horizontal_lines[:2] if len(horizontal_lines) >= 2 else horizontal_lines,
            'sidelines': vertical_lines[:2] if len(vertical_lines) >= 2 else vertical_lines,
            'service_lines': horizontal_lines[1:-1] if len(horizontal_lines) >= 3 else [],
            'net_line': horizontal_lines[len(horizontal_lines)//2] if horizontal_lines else None,
            'court_corners': court_corners
        }
    
    def _find_court_corners_relaxed(self, intersections: List[Tuple[int, int]], 
                                   frame_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """Find court corners with relaxed criteria"""
        if len(intersections) < 2:
            return intersections
        
        h, w = frame_shape[:2]
        
        # If we have enough intersections, try to find corners in quadrants
        if len(intersections) >= 4:
            corners = []
            quadrants = [(0, w//2, 0, h//2), (w//2, w, 0, h//2), 
                        (0, w//2, h//2, h), (w//2, w, h//2, h)]
            
            for x_min, x_max, y_min, y_max in quadrants:
                quadrant_points = [(x, y) for x, y in intersections
                                 if x_min <= x < x_max and y_min <= y < y_max]
                
                if quadrant_points:
                    # Use the point closest to the quadrant corner
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
        else:
            # Just return the intersections we have
            return intersections

    def _match_features_to_pattern(self, court_features: Dict[str, Any], 
                                pattern: Any, frame_shape: Tuple[int, ...]) -> List[DetectedKeypoint]:
        """Match detected court features to calibration pattern keypoints"""
        
        detected_keypoints = []
        h, w = frame_shape[:2]
        
        # Extract available features
        corners = court_features.get('court_corners', [])
        horizontal_lines = court_features.get('horizontal_lines', [])
        vertical_lines = court_features.get('vertical_lines', [])
        intersections = court_features.get('intersections', [])
        
        # Strategy 1: Use corners if available
        if len(corners) >= 2:
            sorted_corners = sorted(corners, key=lambda p: (p[1], p[0]))
            corner_names = ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner']
            
            for i, corner_name in enumerate(corner_names):
                if i < len(sorted_corners):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
                    if keypoint:
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=sorted_corners[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=corner_name,
                            confidence=0.8,
                            detection_method='corner_detection'
                        ))
        
        # Strategy 2: Use intersections as corners
        elif len(intersections) >= 2:
            sorted_intersections = sorted(intersections, key=lambda p: (p[1], p[0]))
            corner_names = ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner']
            
            for i, corner_name in enumerate(corner_names):
                if i < len(sorted_intersections):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(corner_name)
                    if keypoint:
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=sorted_intersections[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=corner_name,
                            confidence=0.6,
                            detection_method='intersection_approximation'
                        ))
        
        return detected_keypoints

    def _sort_corners_clockwise(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort corner points in clockwise order starting from top-left"""
        
        if len(corners) < 4:
            return corners
        
        # Find centroid
        cx = sum(x for x, y in corners) / len(corners)
        cy = sum(y for x, y in corners) / len(corners)
        
        # Sort by angle from centroid
        def angle_from_center(point):
            x, y = point
            return np.arctan2(y - cy, x - cx)
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Ensure we start from top-left (minimum x+y)
        top_left_idx = min(range(len(sorted_corners)), 
                        key=lambda i: sorted_corners[i][0] + sorted_corners[i][1])
        
        # Rotate list to start from top-left
        reordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
        
        return reordered
