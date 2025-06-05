import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class FrameQuality:
    """Data class for frame quality metrics"""
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    line_visibility_score: float
    occlusion_score: float
    overall_score: float
    quality_grade: str  # 'excellent', 'good', 'fair', 'poor'

class FrameQualityAnalyzer:
    """Analyze frame quality for camera calibration suitability"""[5]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds for tennis court calibration
        self.sharpness_threshold = 100.0  # Variance of Laplacian
        self.brightness_range = (80, 180)  # Optimal brightness range
        self.contrast_threshold = 50.0     # Standard deviation threshold
        self.line_density_threshold = 0.02 # Minimum line pixel ratio
    
    def analyze_frame_quality(self, frame: np.ndarray, frame_id: str = "") -> FrameQuality:
        """Comprehensive frame quality analysis for calibration"""[5]
        
        if frame is None or frame.size == 0:
            return self._create_poor_quality()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate individual quality metrics
        sharpness_score = self._calculate_sharpness(gray)
        brightness_score = self._calculate_brightness_quality(gray)
        contrast_score = self._calculate_contrast_quality(gray)
        line_visibility_score = self._calculate_line_visibility(gray)
        occlusion_score = self._calculate_occlusion_score(frame)
        
        # Calculate weighted overall score
        weights = {
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.20,
            'line_visibility': 0.30,
            'occlusion': 0.10
        }
        
        overall_score = (
            sharpness_score * weights['sharpness'] +
            brightness_score * weights['brightness'] +
            contrast_score * weights['contrast'] +
            line_visibility_score * weights['line_visibility'] +
            occlusion_score * weights['occlusion']
        )
        
        # Determine quality grade
        quality_grade = self._determine_quality_grade(overall_score)
        
        return FrameQuality(
            sharpness_score=sharpness_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            line_visibility_score=line_visibility_score,
            occlusion_score=occlusion_score,
            overall_score=overall_score,
            quality_grade=quality_grade
        )
    
    def _calculate_sharpness(self, gray_frame: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""[5]
        
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-100 scale
        normalized_score = min(100.0, variance / self.sharpness_threshold * 100)
        
        return float(normalized_score)
    
    def _calculate_brightness_quality(self, gray_frame: np.ndarray) -> float:
        """Calculate brightness quality based on optimal range"""[4]
        
        mean_brightness = np.mean(gray_frame)
        
        # Check if brightness is in optimal range
        if self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]:
            # Calculate how close to the center of the range
            center = sum(self.brightness_range) / 2
            deviation = abs(mean_brightness - center)
            max_deviation = (self.brightness_range[1] - self.brightness_range[0]) / 2
            score = (1 - deviation / max_deviation) * 100
        else:
            # Penalize severely for out-of-range brightness
            if mean_brightness < self.brightness_range[0]:
                score = max(0, mean_brightness / self.brightness_range[0] * 50)
            else:
                score = max(0, (255 - mean_brightness) / (255 - self.brightness_range[1]) * 50)
        
        return float(score)
    
    def _calculate_contrast_quality(self, gray_frame: np.ndarray) -> float:
        """Calculate contrast quality using standard deviation"""[5]
        
        contrast = np.std(gray_frame)
        
        # Normalize to 0-100 scale
        normalized_score = min(100.0, contrast / self.contrast_threshold * 100)
        
        return float(normalized_score)
    
    def _calculate_line_visibility(self, gray_frame: np.ndarray) -> float:
        """Calculate line visibility score for tennis court"""[3]
        
        # Apply edge detection to find lines
        edges = cv2.Canny(gray_frame, 50, 150)
        
        # Calculate line pixel density
        total_pixels = gray_frame.size
        line_pixels = np.sum(edges > 0)
        line_density = line_pixels / total_pixels
        
        # Normalize to 0-100 scale
        score = min(100.0, line_density / self.line_density_threshold * 100)
        
        return float(score)
    
    def _calculate_occlusion_score(self, frame: np.ndarray) -> float:
        """Calculate occlusion score (higher = less occlusion)"""
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define tennis court color ranges (green/blue courts)
        court_ranges = [
            # Green court range
            (np.array([35, 40, 40]), np.array([85, 255, 255])),
            # Blue court range
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Hard court (gray/white) range
            (np.array([0, 0, 100]), np.array([180, 30, 255]))
        ]
        
        court_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in court_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            court_mask = cv2.bitwise_or(court_mask, mask)
        
        # Calculate percentage of court visible
        court_percentage = np.sum(court_mask > 0) / court_mask.size
        
        # Score based on court visibility (less occlusion = higher score)
        occlusion_score = court_percentage * 100
        
        return float(occlusion_score)
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """Determine quality grade based on overall score"""[5]
        
        if overall_score >= 80:
            return 'excellent'
        elif overall_score >= 65:
            return 'good'
        elif overall_score >= 45:
            return 'fair'
        else:
            return 'poor'
    
    def _create_poor_quality(self) -> FrameQuality:
        """Create a poor quality result for invalid frames"""
        
        return FrameQuality(
            sharpness_score=0.0,
            brightness_score=0.0,
            contrast_score=0.0,
            line_visibility_score=0.0,
            occlusion_score=0.0,
            overall_score=0.0,
            quality_grade='poor'
        )
