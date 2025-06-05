import numpy as np
from typing import Dict, Any, List
import logging
from dataclasses import dataclass

from .quality_analyzer import FrameQuality
from .court_detector import CourtFeatures

@dataclass
class FrameScore:
    """Data class for frame calibration suitability score"""
    frame_id: str
    quality_score: float
    feature_score: float
    geometric_score: float
    uniqueness_score: float
    total_score: float
    calibration_suitability: str  # 'excellent', 'good', 'fair', 'poor'
    recommended_for_calibration: bool

class FrameCalibrationScorer:
    """Score frames for their suitability in camera calibration"""[5]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights for different aspects
        self.weights = {
            'quality': 0.25,      # Frame quality (sharpness, brightness, etc.)
            'features': 0.35,     # Court feature detection quality
            'geometry': 0.25,     # Geometric distribution of features
            'uniqueness': 0.15    # Uniqueness compared to other frames
        }
        
        # Minimum thresholds for calibration recommendation
        self.min_total_score = 60.0
        self.min_feature_count = 5
    
    def score_frame_for_calibration(self, 
                                  frame_id: str,
                                  frame_quality: FrameQuality,
                                  court_features: CourtFeatures,
                                  frame_shape: tuple) -> FrameScore:
        """Score a single frame for calibration suitability"""[5]
        
        # Calculate individual score components
        quality_score = self._calculate_quality_score(frame_quality)
        feature_score = self._calculate_feature_score(court_features)
        geometric_score = self._calculate_geometric_score(court_features, frame_shape)
        
        # Uniqueness score will be calculated later when comparing with other frames
        uniqueness_score = 50.0  # Default neutral score
        
        # Calculate weighted total score
        total_score = (
            quality_score * self.weights['quality'] +
            feature_score * self.weights['features'] +
            geometric_score * self.weights['geometry'] +
            uniqueness_score * self.weights['uniqueness']
        )
        
        # Determine calibration suitability
        calibration_suitability = self._determine_suitability(total_score)
        
        # Recommendation based on score and feature count
        recommended = (
            total_score >= self.min_total_score and
            court_features.feature_count >= self.min_feature_count
        )
        
        return FrameScore(
            frame_id=frame_id,
            quality_score=quality_score,
            feature_score=feature_score,
            geometric_score=geometric_score,
            uniqueness_score=uniqueness_score,
            total_score=total_score,
            calibration_suitability=calibration_suitability,
            recommended_for_calibration=recommended
        )
    
    def _calculate_quality_score(self, frame_quality: FrameQuality) -> float:
        """Calculate score based on frame quality metrics"""[5]
        
        # Weighted combination of quality metrics
        quality_weights = {
            'sharpness': 0.3,
            'brightness': 0.2,
            'contrast': 0.2,
            'line_visibility': 0.2,
            'occlusion': 0.1
        }
        
        score = (
            frame_quality.sharpness_score * quality_weights['sharpness'] +
            frame_quality.brightness_score * quality_weights['brightness'] +
            frame_quality.contrast_score * quality_weights['contrast'] +
            frame_quality.line_visibility_score * quality_weights['line_visibility'] +
            frame_quality.occlusion_score * quality_weights['occlusion']
        )
        
        return float(score)
    
    def _calculate_feature_score(self, court_features: CourtFeatures) -> float:
        """Calculate score based on detected court features"""[3]
        
        # Base score from detection confidence
        base_score = court_features.detection_confidence * 60
        
        # Bonus points for specific features
        feature_bonuses = {
            'baselines': len(court_features.baselines) * 8,
            'service_lines': len(court_features.service_lines) * 6,
            'sidelines': len(court_features.sidelines) * 8,
            'center_line': 5 if court_features.center_line else 0,
            'net_line': 3 if court_features.net_line else 0,
            'intersections': min(20, len(court_features.intersections) * 2),
            'corners': min(15, len(court_features.court_corners) * 3)
        }
        
        bonus_score = sum(feature_bonuses.values())
        total_score = min(100.0, base_score + bonus_score)
        
        return float(total_score)
    
    def _calculate_geometric_score(self, court_features: CourtFeatures, frame_shape: tuple) -> float:
        """Calculate score based on geometric distribution of features"""[2]
        
        if not court_features.intersections:
            return 20.0  # Poor geometric distribution
        
        h, w = frame_shape[:2]
        
        # Analyze distribution of intersection points
        points = np.array(court_features.intersections)
        
        # Calculate coverage of frame quadrants
        quadrant_coverage = self._calculate_quadrant_coverage(points, w, h)
        
        # Calculate spread of points
        spread_score = self._calculate_point_spread(points, w, h)
        
        # Calculate aspect ratio compliance
        aspect_ratio_score = self._calculate_aspect_ratio_score(court_features)
        
        # Combine geometric scores
        geometric_score = (
            quadrant_coverage * 0.4 +
            spread_score * 0.4 +
            aspect_ratio_score * 0.2
        )
        
        return float(geometric_score)
    
    def _calculate_quadrant_coverage(self, points: np.ndarray, width: int, height: int) -> float:
        """Calculate how well points cover all quadrants"""
        
        if len(points) == 0:
            return 0.0
        
        # Define quadrants
        mid_x, mid_y = width // 2, height // 2
        
        quadrants = [
            (points[:, 0] < mid_x) & (points[:, 1] < mid_y),    # Top-left
            (points[:, 0] >= mid_x) & (points[:, 1] < mid_y),   # Top-right
            (points[:, 0] < mid_x) & (points[:, 1] >= mid_y),   # Bottom-left
            (points[:, 0] >= mid_x) & (points[:, 1] >= mid_y)   # Bottom-right
        ]
        
        # Count covered quadrants
        covered_quadrants = sum(1 for q in quadrants if np.any(q))
        
        # Score based on quadrant coverage
        coverage_score = (covered_quadrants / 4) * 100
        
        return coverage_score
    
    def _calculate_point_spread(self, points: np.ndarray, width: int, height: int) -> float:
        """Calculate how well points are spread across the frame"""
        
        if len(points) < 2:
            return 0.0
        
        # Calculate standard deviation of point positions
        x_std = np.std(points[:, 0]) / width
        y_std = np.std(points[:, 1]) / height
        
        # Good spread means higher standard deviation (up to a point)
        optimal_std = 0.3  # 30% of frame dimension
        
        x_score = min(1.0, x_std / optimal_std) * 50
        y_score = min(1.0, y_std / optimal_std) * 50
        
        spread_score = x_score + y_score
        
        return spread_score
    
    def _calculate_aspect_ratio_score(self, court_features: CourtFeatures) -> float:
        """Calculate score based on tennis court aspect ratio compliance"""
        
        if len(court_features.court_corners) < 4:
            return 50.0  # Neutral score if insufficient corners
        
        # Calculate bounding rectangle of court corners
        corners = np.array(court_features.court_corners)
        
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if height == 0:
            return 0.0
        
        detected_ratio = width / height
        
        # Expected tennis court aspect ratio (length/width = 23.77/10.97 ≈ 2.17)
        expected_ratio = 2.17
        
        # Score based on how close the detected ratio is to expected
        ratio_error = abs(detected_ratio - expected_ratio) / expected_ratio
        aspect_score = max(0, (1 - ratio_error) * 100)
        
        return float(aspect_score)
    
    def calculate_uniqueness_scores(self, frame_scores: List[FrameScore], 
                                  features_list: List[CourtFeatures]) -> List[FrameScore]:
        """Calculate uniqueness scores by comparing frames"""
        
        if len(frame_scores) < 2:
            return frame_scores
        
        # Calculate pairwise similarities
        similarities = self._calculate_pairwise_similarities(features_list)
        
        # Update uniqueness scores
        updated_scores = []
        for i, score in enumerate(frame_scores):
            # Calculate uniqueness as inverse of average similarity to other frames
            avg_similarity = np.mean([similarities[i][j] for j in range(len(similarities)) if i != j])
            uniqueness_score = (1 - avg_similarity) * 100
            
            # Recalculate total score with new uniqueness score
            new_total_score = (
                score.quality_score * self.weights['quality'] +
                score.feature_score * self.weights['features'] +
                score.geometric_score * self.weights['geometry'] +
                uniqueness_score * self.weights['uniqueness']
            )
            
            # Create updated score
            updated_score = FrameScore(
                frame_id=score.frame_id,
                quality_score=score.quality_score,
                feature_score=score.feature_score,
                geometric_score=score.geometric_score,
                uniqueness_score=uniqueness_score,
                total_score=new_total_score,
                calibration_suitability=self._determine_suitability(new_total_score),
                recommended_for_calibration=(
                    new_total_score >= self.min_total_score and
                    len(features_list[i].lines) >= self.min_feature_count
                )
            )
            
            updated_scores.append(updated_score)
        
        return updated_scores
    
    def _calculate_pairwise_similarities(self, features_list: List[CourtFeatures]) -> np.ndarray:
        """Calculate similarity matrix between all frames"""
        
        n_frames = len(features_list)
        similarities = np.zeros((n_frames, n_frames))
        
        for i in range(n_frames):
            for j in range(n_frames):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    similarity = self._calculate_frame_similarity(features_list[i], features_list[j])
                    similarities[i][j] = similarity
        
        return similarities
    
    def _calculate_frame_similarity(self, features1: CourtFeatures, features2: CourtFeatures) -> float:
        """Calculate similarity between two frames based on court features"""
        
        # Compare feature counts
        feature_count_similarity = 1 - abs(features1.feature_count - features2.feature_count) / max(features1.feature_count, features2.feature_count, 1)
        
        # Compare intersection counts
        intersection_similarity = 1 - abs(len(features1.intersections) - len(features2.intersections)) / max(len(features1.intersections), len(features2.intersections), 1)
        
        # Compare detection confidence
        confidence_similarity = 1 - abs(features1.detection_confidence - features2.detection_confidence)
        
        # Weighted average
        similarity = (
            feature_count_similarity * 0.4 +
            intersection_similarity * 0.3 +
            confidence_similarity * 0.3
        )
        
        return similarity
    
    def _determine_suitability(self, total_score: float) -> str:
        """Determine calibration suitability based on total score"""
        
        if total_score >= 80:
            return 'excellent'
        elif total_score >= 65:
            return 'good'
        elif total_score >= 45:
            return 'fair'
        else:
            return 'poor'
