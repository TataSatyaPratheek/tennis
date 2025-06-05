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
    debug_info: Dict[str, Any]

class IndoorTennisCourtKeypointDetector:
    """Specialized detector for indoor tennis courts with colored surfaces and white lines"""
    
    def __init__(self, pattern_manager: CalibrationPatternManager):
        self.pattern_manager = pattern_manager
        self.logger = logging.getLogger(__name__)
        
        # Indoor court color ranges (HSV)
        self.court_colors = {
            'blue_court': {
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            },
            'green_court': {
                'lower': np.array([40, 40, 40]),
                'upper': np.array([80, 255, 255])
            },
            'purple_court': {
                'lower': np.array([130, 50, 50]),
                'upper': np.array([160, 255, 255])
            }
        }
        
        # White line detection parameters
        self.white_line_params = {
            'lower_white': np.array([0, 0, 200]),
            'upper_white': np.array([180, 30, 255]),
            'min_area': 100,
            'min_line_length': 40,
            'max_line_gap': 8
        }
        
        # Enhanced detection parameters for indoor courts
        self.detection_params = {
            'gaussian_blur': (3, 3),
            'morphology_kernel': (3, 3),
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 60,
            'hough_min_length': 50,
            'hough_max_gap': 10
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'excellent': 0.85,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
        
        # Service area focus for your video type
        self.service_area_weights = {
            'service_box_corners': 1.0,
            'baseline_intersections': 0.9,
            'center_service_line': 0.8,
            'net_area': 0.6
        }
    
    def detect_keypoints_in_frame(self, frame: np.ndarray, frame_id: str, 
                                pattern_type: str = 'minimal') -> KeypointDetectionResult:
        """Enhanced keypoint detection optimized for indoor tennis courts"""
        
        if frame is None or frame.size == 0:
            return self._create_empty_result(frame_id)
        
        debug_info = {
            'methods_tried': [],
            'court_color_detected': None,
            'white_line_pixels': 0,
            'detected_lines_count': 0,
            'service_area_coverage': 0
        }
        
        try:
            # Method 1: Color-based white line detection
            result = self._detect_using_color_segmentation(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 4 and result.avg_confidence > 0.6:
                return result
            
            # Method 2: Enhanced edge detection with court mask
            result = self._detect_using_enhanced_edges(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 4 and result.avg_confidence > 0.5:
                return result
            
            # Method 3: Template-based service area detection
            result = self._detect_service_area_focused(frame, frame_id, pattern_type, debug_info)
            if result.detection_count >= 3:
                return result
            
            # Method 4: Fallback with geometric estimation
            result = self._geometric_estimation_fallback(frame, frame_id, pattern_type, debug_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in indoor court detection for frame {frame_id}: {e}")
            return self._create_empty_result(frame_id, debug_info)
    
    def _detect_using_color_segmentation(self, frame: np.ndarray, frame_id: str,
                                       pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Primary method: Color-based detection for white lines on colored court"""
        
        debug_info['methods_tried'].append('color_segmentation')
        
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect court surface color
        court_color = self._detect_court_surface_color(hsv)
        debug_info['court_color_detected'] = court_color
        
        # Create court mask
        court_mask = self._create_court_surface_mask(hsv, court_color)
        
        # Detect white lines
        white_line_mask = self._detect_white_lines(hsv)
        debug_info['white_line_pixels'] = np.sum(white_line_mask > 0)
        
        # Combine masks to isolate court lines
        court_lines_mask = cv2.bitwise_and(white_line_mask, court_mask)
        
        # Enhance line detection
        enhanced_lines = self._enhance_line_mask(court_lines_mask)
        
        # Extract line segments
        lines = self._extract_precise_lines(enhanced_lines)
        debug_info['detected_lines_count'] = len(lines)
        
        if not lines:
            return self._create_empty_result(frame_id, debug_info)
        
        # Find intersections with sub-pixel accuracy
        intersections = self._find_sub_pixel_intersections(lines, enhanced_lines)
        
        # Detect court features specific to service area
        court_features = self._analyze_service_area_features(lines, intersections, frame.shape)
        
        # Match to calibration pattern with confidence scoring
        detected_keypoints = self._match_to_pattern_with_confidence(
            court_features, self.pattern_manager.get_pattern(pattern_type), frame.shape
        )
        
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
    
    def _detect_court_surface_color(self, hsv: np.ndarray) -> str:
        """Detect the primary court surface color"""
        
        best_match = 'blue_court'  # Default for your video
        max_pixels = 0
        
        for color_name, color_range in self.court_colors.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            pixel_count = np.sum(mask > 0)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_match = color_name
        
        return best_match
    
    def _create_court_surface_mask(self, hsv: np.ndarray, court_color: str) -> np.ndarray:
        """Create mask for court surface area"""
        
        if court_color in self.court_colors:
            color_range = self.court_colors[court_color]
            court_mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        else:
            # Fallback: create mask from central area
            h, w = hsv.shape[:2]
            court_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(court_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel)
        
        return court_mask
    
    def _detect_white_lines(self, hsv: np.ndarray) -> np.ndarray:
        """Detect white lines with optimized parameters for indoor courts"""
        
        # Create white mask
        white_mask = cv2.inRange(hsv, self.white_line_params['lower_white'], 
                               self.white_line_params['upper_white'])
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small gaps in lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        return white_mask
    
    def _enhance_line_mask(self, line_mask: np.ndarray) -> np.ndarray:
        """Enhance detected line mask for better line extraction"""
        
        # Apply morphological operations to connect broken lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Enhance horizontal lines
        horizontal_lines = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Enhance vertical lines
        vertical_lines = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Combine enhanced lines
        enhanced = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Final cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        return enhanced
    
    def _extract_precise_lines(self, line_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract line segments with high precision"""
        
        # Apply edge detection on the mask
        edges = cv2.Canny(line_mask, 50, 150)
        
        # Use probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.detection_params['hough_threshold'],
            minLineLength=self.detection_params['hough_min_length'],
            maxLineGap=self.detection_params['hough_max_gap']
        )
        
        if lines is None:
            return []
        
        # Convert to list format
        line_list = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]
        
        # Filter and merge similar lines
        filtered_lines = self._filter_and_merge_lines(line_list, line_mask.shape)
        
        return filtered_lines
    
    def _filter_and_merge_lines(self, lines: List[Tuple[int, int, int, int]], 
                               frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Filter and merge lines specific to tennis court geometry"""
        
        if not lines:
            return []
        
        h, w = frame_shape
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Filter by minimum length
            if length < min(w, h) * 0.08:
                continue
            
            # Classify by angle (stricter tolerance for better accuracy)
            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 80 < abs(angle) < 100:
                vertical_lines.append((x1, y1, x2, y2))
        
        # Merge similar lines within each group
        merged_horizontal = self._merge_parallel_lines(horizontal_lines, is_horizontal=True)
        merged_vertical = self._merge_parallel_lines(vertical_lines, is_horizontal=False)
        
        return merged_horizontal + merged_vertical
    
    def _merge_parallel_lines(self, lines: List[Tuple[int, int, int, int]], 
                            is_horizontal: bool) -> List[Tuple[int, int, int, int]]:
        """Merge parallel lines that are close to each other"""
        
        if len(lines) < 2:
            return lines
        
        merged = []
        used = [False] * len(lines)
        threshold = 15  # pixels
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            
            group = [line1]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue
                
                if self._lines_are_similar(line1, line2, threshold, is_horizontal):
                    group.append(line2)
                    used[j] = True
            
            # Merge the group into a single representative line
            merged_line = self._merge_line_group(group, is_horizontal)
            merged.append(merged_line)
        
        return merged
    
    def _lines_are_similar(self, line1: Tuple[int, int, int, int], 
                         line2: Tuple[int, int, int, int],
                         threshold: int, is_horizontal: bool) -> bool:
        """Check if two lines are similar enough to merge"""
        
        x1_1, y1_1, x2_1, y2_1 = line1
        x1_2, y1_2, x2_2, y2_2 = line2
        
        if is_horizontal:
            # Check y-coordinate distance for horizontal lines
            avg_y1 = (y1_1 + y2_1) / 2
            avg_y2 = (y1_2 + y2_2) / 2
            return abs(avg_y1 - avg_y2) < threshold
        else:
            # Check x-coordinate distance for vertical lines
            avg_x1 = (x1_1 + x2_1) / 2
            avg_x2 = (x1_2 + x2_2) / 2
            return abs(avg_x1 - avg_x2) < threshold
    
    def _merge_line_group(self, lines: List[Tuple[int, int, int, int]], 
                        is_horizontal: bool) -> Tuple[int, int, int, int]:
        """Merge a group of similar lines into one representative line"""
        
        if len(lines) == 1:
            return lines[0]
        
        if is_horizontal:
            # For horizontal lines, extend to cover all x-coordinates
            min_x = min(min(x1, x2) for x1, y1, x2, y2 in lines)
            max_x = max(max(x1, x2) for x1, y1, x2, y2 in lines)
            avg_y = int(np.mean([y1 + y2 for x1, y1, x2, y2 in lines]) / 2)
            return (min_x, avg_y, max_x, avg_y)
        else:
            # For vertical lines, extend to cover all y-coordinates
            min_y = min(min(y1, y2) for x1, y1, x2, y2 in lines)
            max_y = max(max(y1, y2) for x1, y1, x2, y2 in lines)
            avg_x = int(np.mean([x1 + x2 for x1, y1, x2, y2 in lines]) / 2)
            return (avg_x, min_y, avg_x, max_y)
    
    def _find_sub_pixel_intersections(self, lines: List[Tuple[int, int, int, int]], 
                                    line_mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find line intersections with sub-pixel accuracy"""
        
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                intersection = self._calculate_precise_intersection(line1, line2)
                
                if intersection:
                    x, y = intersection
                    h, w = line_mask.shape
                    
                    # Validate intersection is within frame and on actual lines
                    if (0 <= x < w and 0 <= y < h and
                        self._verify_intersection_on_lines(intersection, line_mask)):
                        intersections.append(intersection)
        
        # Remove duplicate intersections
        unique_intersections = self._remove_duplicate_intersections(intersections, threshold=8)
        
        return unique_intersections
    
    def _calculate_precise_intersection(self, line1: Tuple[int, int, int, int], 
                                      line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """Calculate intersection with sub-pixel precision"""
        
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Convert to float for precision
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x3, y3, x4, y4 = float(x3), float(y3), float(x4), float(y4)
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Allow intersection slightly outside line segments for robustness
        if -0.2 <= t <= 1.2 and -0.2 <= u <= 1.2:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def _verify_intersection_on_lines(self, intersection: Tuple[float, float], 
                                    line_mask: np.ndarray) -> bool:
        """Verify intersection point is on actual detected lines"""
        
        x, y = intersection
        h, w = line_mask.shape
        
        # Check a small neighborhood around the intersection
        radius = 3
        x_min = max(0, int(x) - radius)
        x_max = min(w, int(x) + radius + 1)
        y_min = max(0, int(y) - radius)
        y_max = min(h, int(y) + radius + 1)
        
        neighborhood = line_mask[y_min:y_max, x_min:x_max]
        
        # Intersection is valid if there are line pixels nearby
        return np.sum(neighborhood > 0) >= 4
    
    def _remove_duplicate_intersections(self, intersections: List[Tuple[float, float]], 
                                      threshold: float = 8.0) -> List[Tuple[float, float]]:
        """Remove duplicate intersections within threshold distance"""
        
        if not intersections:
            return []
        
        unique = []
        
        for point in intersections:
            is_duplicate = False
            for existing in unique:
                distance = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(point)
        
        return unique
    
    def _analyze_service_area_features(self, lines: List[Tuple[int, int, int, int]], 
                                     intersections: List[Tuple[float, float]], 
                                     frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze court features focusing on visible service area"""
        
        h, w = frame_shape[:2]
        
        # Classify lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_lines.append(line)
            elif 75 < abs(angle) < 105:
                vertical_lines.append(line)
        
        # Sort lines by position
        horizontal_lines.sort(key=lambda line: (line[1] + line[3]) / 2)
        vertical_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        
        # Identify specific court features
        service_box_corners = self._find_service_box_corners(intersections, frame_shape)
        baseline_intersections = self._find_baseline_intersections(horizontal_lines, vertical_lines)
        center_service_intersections = self._find_center_service_intersections(lines, intersections)
        
        return {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'intersections': intersections,
            'service_box_corners': service_box_corners,
            'baseline_intersections': baseline_intersections,
            'center_service_intersections': center_service_intersections,
            'court_coverage': self._calculate_court_coverage(lines, frame_shape)
        }
    
    def _find_service_box_corners(self, intersections: List[Tuple[float, float]], 
                                frame_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Find service box corners which are most visible in your video"""
        
        if len(intersections) < 2:
            return intersections
        
        h, w = frame_shape[:2]
        
        # Focus on the near service area (bottom 60% of frame)
        near_intersections = [(x, y) for x, y in intersections if y > h * 0.4]
        
        if not near_intersections:
            return intersections[:4] if len(intersections) >= 4 else intersections
        
        # Sort by distance from bottom center (camera perspective)
        bottom_center = (w // 2, h)
        near_intersections.sort(key=lambda p: np.sqrt((p[0] - bottom_center[0])**2 + (p[1] - bottom_center[1])**2))
        
        return near_intersections[:4] if len(near_intersections) >= 4 else near_intersections
    
    def _find_baseline_intersections(self, horizontal_lines: List, vertical_lines: List) -> List[Tuple[float, float]]:
        """Find baseline intersections with sidelines"""
        
        baseline_intersections = []
        
        if horizontal_lines and vertical_lines:
            # Use bottom-most horizontal line as baseline
            baseline = horizontal_lines[-1] if horizontal_lines else None
            
            if baseline:
                for sideline in vertical_lines:
                    intersection = self._calculate_precise_intersection(baseline, sideline)
                    if intersection:
                        baseline_intersections.append(intersection)
        
        return baseline_intersections
    
    def _find_center_service_intersections(self, lines: List, intersections: List) -> List[Tuple[float, float]]:
        """Find center service line intersections"""
        
        # This would require more sophisticated analysis of line arrangements
        # For now, return intersections in the central area
        if not intersections:
            return []
        
        # Find intersections near the center of the frame
        center_intersections = []
        for x, y in intersections:
            # Center service line typically in middle third of frame width
            if 0.3 < x / 640 < 0.7:  # Assuming 640 width, adjust for your frame size
                center_intersections.append((x, y))
        
        return center_intersections
    
    def _calculate_court_coverage(self, lines: List, frame_shape: Tuple[int, int]) -> float:
        """Calculate how much of the court is covered by detected lines"""
        
        if not lines:
            return 0.0
        
        h, w = frame_shape[:2]
        
        # Create a mask of detected lines
        line_mask = np.zeros((h, w), dtype=np.uint8)
        
        for x1, y1, x2, y2 in lines:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        
        # Calculate coverage as percentage of frame
        coverage = np.sum(line_mask > 0) / (h * w) * 100
        
        return min(100.0, coverage)
    
    def _match_to_pattern_with_confidence(self, court_features: Dict[str, Any], 
                                        pattern: Any, frame_shape: Tuple[int, int]) -> List[DetectedKeypoint]:
        """Match detected features to calibration pattern with confidence scoring"""
        
        detected_keypoints = []
        h, w = frame_shape[:2]
        
        # Prioritize service box corners (most reliable in your video)
        service_corners = court_features.get('service_box_corners', [])
        baseline_intersections = court_features.get('baseline_intersections', [])
        center_intersections = court_features.get('center_service_intersections', [])
        
        # Map service box corners to tennis court keypoints
        if len(service_corners) >= 4:
            corner_mappings = [
                ('bottom_left_corner', 0.9),
                ('bottom_right_corner', 0.9),
                ('service_line_left', 0.85),
                ('service_line_right', 0.85)
            ]
            
            for i, (keypoint_name, base_confidence) in enumerate(corner_mappings):
                if i < len(service_corners):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(keypoint_name)
                    if keypoint:
                        # Adjust confidence based on detection quality
                        confidence = base_confidence * self._calculate_point_confidence(
                            service_corners[i], court_features, frame_shape
                        )
                        
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=service_corners[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=keypoint_name,
                            confidence=confidence,
                            detection_method='service_box_detection'
                        ))
        
        # Add baseline intersections if available
        if len(baseline_intersections) >= 2:
            baseline_mappings = [
                ('bottom_left_corner', 0.8),
                ('bottom_right_corner', 0.8)
            ]
            
            for i, (keypoint_name, base_confidence) in enumerate(baseline_mappings):
                if i < len(baseline_intersections):
                    keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(keypoint_name)
                    if keypoint:
                        confidence = base_confidence * self._calculate_point_confidence(
                            baseline_intersections[i], court_features, frame_shape
                        )
                        
                        detected_keypoints.append(DetectedKeypoint(
                            image_coords=baseline_intersections[i],
                            world_coords=keypoint.world_coords,
                            keypoint_name=keypoint_name,
                            confidence=confidence,
                            detection_method='baseline_intersection'
                        ))
        
        return detected_keypoints
    
    def _calculate_point_confidence(self, point: Tuple[float, float], 
                                  court_features: Dict[str, Any], 
                                  frame_shape: Tuple[int, int]) -> float:
        """Calculate confidence score for a detected point"""
        
        # Base confidence factors
        factors = []
        
        # Factor 1: Distance from frame edges (avoid boundary artifacts)
        h, w = frame_shape[:2]
        x, y = point
        edge_distance = min(x, y, w - x, h - y)
        edge_factor = min(1.0, edge_distance / 20)  # Points too close to edges are less reliable
        factors.append(edge_factor)
        
        # Factor 2: Density of nearby intersections (more intersections = more reliable)
        nearby_intersections = 0
        for other_x, other_y in court_features.get('intersections', []):
            distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
            if 5 < distance < 50:  # Nearby but not the same point
                nearby_intersections += 1
        
        density_factor = min(1.0, nearby_intersections / 3)
        factors.append(density_factor)
        
        # Factor 3: Court coverage (better detection = higher confidence)
        coverage = court_features.get('court_coverage', 0)
        coverage_factor = min(1.0, coverage / 5)  # 5% coverage as baseline
        factors.append(coverage_factor)
        
        # Combine factors (geometric mean for balanced weighting)
        combined_confidence = np.power(np.prod(factors), 1.0 / len(factors))
        
        return float(combined_confidence)
    
    def _detect_using_enhanced_edges(self, frame: np.ndarray, frame_id: str,
                                   pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Fallback method using enhanced edge detection"""
        
        debug_info['methods_tried'].append('enhanced_edges')
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Apply Canny edge detection
        edges = cv2.Canny(enhanced, self.detection_params['canny_low'], 
                         self.detection_params['canny_high'])
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.detection_params['hough_threshold'],
            minLineLength=self.detection_params['hough_min_length'],
            maxLineGap=self.detection_params['hough_max_gap']
        )
        
        if lines is None:
            return self._create_empty_result(frame_id, debug_info)
        
        line_list = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]
        filtered_lines = self._filter_and_merge_lines(line_list, frame.shape[:2])
        
        intersections = self._find_sub_pixel_intersections(filtered_lines, edges)
        court_features = self._analyze_service_area_features(filtered_lines, intersections, frame.shape)
        
        detected_keypoints = self._match_to_pattern_with_confidence(
            court_features, self.pattern_manager.get_pattern(pattern_type), frame.shape
        )
        
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
    
    def _detect_service_area_focused(self, frame: np.ndarray, frame_id: str,
                                   pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Method focused specifically on service area detection"""
        
        debug_info['methods_tried'].append('service_area_focused')
        
        h, w = frame.shape[:2]
        
        # Focus on service area (bottom 60% of frame where service boxes are visible)
        service_area = frame[int(h * 0.4):, :]
        
        # Apply color-based detection to service area
        hsv_service = cv2.cvtColor(service_area, cv2.COLOR_BGR2HSV)
        white_lines = self._detect_white_lines(hsv_service)
        
        # Extract lines from service area
        lines = self._extract_precise_lines(white_lines)
        
        # Adjust line coordinates back to full frame
        adjusted_lines = []
        y_offset = int(h * 0.4)
        for x1, y1, x2, y2 in lines:
            adjusted_lines.append((x1, y1 + y_offset, x2, y2 + y_offset))
        
        if not adjusted_lines:
            return self._create_empty_result(frame_id, debug_info)
        
        # Find intersections
        intersections = self._find_sub_pixel_intersections(adjusted_lines, white_lines)
        
        # Focus on service box detection
        service_features = {
            'horizontal_lines': [line for line in adjusted_lines if abs(np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi) < 15],
            'vertical_lines': [line for line in adjusted_lines if 75 < abs(np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi) < 105],
            'intersections': intersections,
            'service_box_corners': intersections[:4] if len(intersections) >= 4 else intersections
        }
        
        detected_keypoints = self._match_to_pattern_with_confidence(
            service_features, self.pattern_manager.get_pattern(pattern_type), frame.shape
        )
        
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
    
    def _geometric_estimation_fallback(self, frame: np.ndarray, frame_id: str,
                                     pattern_type: str, debug_info: Dict) -> KeypointDetectionResult:
        """Final fallback using geometric estimation based on typical court layout"""
        
        debug_info['methods_tried'].append('geometric_estimation')
        
        h, w = frame.shape[:2]
        pattern = self.pattern_manager.get_pattern(pattern_type)
        
        # Estimate keypoints based on typical service area layout in broadcast view
        estimated_keypoints = []
        
        # Estimated positions based on your video characteristics
        estimates = [
            ("bottom_left_corner", (w * 0.15, h * 0.85), 0.4),
            ("bottom_right_corner", (w * 0.85, h * 0.85), 0.4),
            ("service_line_left", (w * 0.25, h * 0.65), 0.3),
            ("service_line_right", (w * 0.75, h * 0.65), 0.3)
        ]
        
        for keypoint_name, (x, y), confidence in estimates:
            keypoint = self.pattern_manager.base_pattern.get_keypoint_by_name(keypoint_name)
            if keypoint:
                estimated_keypoints.append(DetectedKeypoint(
                    image_coords=(x, y),
                    world_coords=keypoint.world_coords,
                    keypoint_name=keypoint_name,
                    confidence=confidence,
                    detection_method='geometric_estimation'
                ))
        
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
    
    def _determine_detection_quality(self, avg_confidence: float, keypoint_count: int) -> str:
        """Determine detection quality based on confidence and keypoint count"""
        
        if keypoint_count >= 6 and avg_confidence >= self.confidence_thresholds['excellent']:
            return 'excellent'
        elif keypoint_count >= 4 and avg_confidence >= self.confidence_thresholds['good']:
            return 'good'
        elif keypoint_count >= 3 and avg_confidence >= self.confidence_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _create_empty_result(self, frame_id: str, debug_info: Dict = None) -> KeypointDetectionResult:
        """Create empty detection result"""
        
        if debug_info is None:
            debug_info = {'methods_tried': [], 'court_color_detected': None, 
                         'white_line_pixels': 0, 'detected_lines_count': 0}
        
        return KeypointDetectionResult(
            frame_id=frame_id,
            detected_keypoints=[],
            detection_count=0,
            avg_confidence=0.0,
            detection_quality='poor',
            debug_info=debug_info
        )
