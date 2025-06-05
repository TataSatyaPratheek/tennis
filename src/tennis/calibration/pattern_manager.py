import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
import logging
from pathlib import Path
import json

from .pattern_design import TennisCourtCalibrationPattern, CourtKeypoint, TennisCourtPattern
from .pattern_variants import TennisCourtPatternVariants

class CalibrationPatternManager:
    """Manage different calibration patterns and provide utilities for calibration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_pattern = TennisCourtCalibrationPattern()
        self.pattern_variants = TennisCourtPatternVariants()
        
        # Available pattern types
        self.available_patterns = {
            'standard': self.base_pattern.calibration_pattern,
            'minimal': self.pattern_variants.get_minimal_pattern(),
            'singles': self.pattern_variants.get_singles_pattern(),
            'robust': self.pattern_variants.get_robust_pattern(),
            'broadcast': self.pattern_variants.get_broadcast_pattern()
        }
    
    def get_pattern(self, pattern_type: str = 'standard') -> TennisCourtPattern:
        """Get calibration pattern by type"""
        
        if pattern_type not in self.available_patterns:
            self.logger.warning(f"Pattern type '{pattern_type}' not available. Using 'standard'.")
            pattern_type = 'standard'
        
        return self.available_patterns[pattern_type]
    
    def prepare_calibration_data(self, pattern_type: str = 'standard') -> Tuple[np.ndarray, Dict[str, any]]:
        """Prepare calibration data for OpenCV calibration functions"""
        
        pattern = self.get_pattern(pattern_type)
        
        # Extract 3D world coordinates
        world_points = np.array([
            [kp.world_coords[0], kp.world_coords[1], kp.world_coords[2]] 
            for kp in pattern.keypoints
        ], dtype=np.float32)
        
        # Prepare metadata
        metadata = {
            'pattern_type': pattern_type,
            'num_keypoints': len(pattern.keypoints),
            'coordinate_system': pattern.coordinate_system,
            'reference_frame': pattern.reference_frame,
            'court_dimensions': pattern.court_dimensions,
            'keypoint_names': [kp.name for kp in pattern.keypoints],
            'keypoint_importance': [kp.importance for kp in pattern.keypoints]
        }
        
        return world_points, metadata
    
    def validate_image_points(self, image_points: np.ndarray, pattern_type: str = 'standard') -> Dict[str, any]:
        """Validate that image points match the expected pattern"""
        
        pattern = self.get_pattern(pattern_type)
        expected_count = len(pattern.keypoints)
        
        validation_result = {
            'valid': False,
            'expected_points': expected_count,
            'provided_points': len(image_points) if image_points is not None else 0,
            'errors': []
        }
        
        if image_points is None:
            validation_result['errors'].append("No image points provided")
            return validation_result
        
        if len(image_points) != expected_count:
            validation_result['errors'].append(
                f"Point count mismatch: expected {expected_count}, got {len(image_points)}"
            )
            return validation_result
        
        # Check point format
        if image_points.shape[1] != 2:
            validation_result['errors'].append(
                f"Invalid point format: expected (N,2), got {image_points.shape}"
            )
            return validation_result
        
        # Check for reasonable coordinate ranges (assuming image coordinates)
        if np.any(image_points < 0) or np.any(image_points > 10000):
            validation_result['errors'].append("Image points contain unreasonable coordinates")
            return validation_result
        
        validation_result['valid'] = True
        return validation_result
    
    def create_point_correspondence_mapping(self, pattern_type: str = 'standard') -> Dict[str, Dict]:
        """Create mapping between keypoint names and their properties"""
        
        pattern = self.get_pattern(pattern_type)
        
        mapping = {}
        for i, keypoint in enumerate(pattern.keypoints):
            mapping[keypoint.name] = {
                'index': i,
                'id': keypoint.id,
                'world_coords': keypoint.world_coords,
                'description': keypoint.description,
                'importance': keypoint.importance
            }
        
        return mapping
    
    def get_critical_points_subset(self, image_points: np.ndarray, pattern_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """Extract critical points subset for robust calibration"""
        
        pattern = self.get_pattern(pattern_type)
        
        critical_indices = []
        for i, keypoint in enumerate(pattern.keypoints):
            if keypoint.importance == 'critical':
                critical_indices.append(i)
        
        if len(critical_indices) < 6:
            self.logger.warning(f"Only {len(critical_indices)} critical points available, need at least 6")
        
        # Extract critical world points
        critical_world_points = np.array([
            [pattern.keypoints[i].world_coords[0], 
             pattern.keypoints[i].world_coords[1], 
             pattern.keypoints[i].world_coords[2]]
            for i in critical_indices
        ], dtype=np.float32)
        
        # Extract corresponding image points
        critical_image_points = image_points[critical_indices] if len(image_points) > max(critical_indices) else None
        
        return critical_world_points, critical_image_points
    
    def export_pattern_for_detection(self, pattern_type: str = 'standard', output_path: Path = None) -> Dict:
        """Export pattern data for court detection algorithms"""
        
        pattern = self.get_pattern(pattern_type)
        
        export_data = {
            'pattern_info': {
                'type': pattern_type,
                'coordinate_system': pattern.coordinate_system,
                'reference_frame': pattern.reference_frame
            },
            'court_dimensions': pattern.court_dimensions,
            'keypoints': [],
            'expected_relationships': self._generate_expected_relationships(pattern)
        }
        
        for keypoint in pattern.keypoints:
            export_data['keypoints'].append({
                'id': keypoint.id,
                'name': keypoint.name,
                'world_coords': keypoint.world_coords,
                'description': keypoint.description,
                'importance': keypoint.importance
            })
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Pattern detection data exported to: {output_path}")
        
        return export_data
    
    def _generate_expected_relationships(self, pattern: TennisCourtPattern) -> Dict:
        """Generate expected geometric relationships between keypoints"""
        
        relationships = {
            'parallel_lines': [
                ['bottom_left_corner', 'bottom_right_corner', 'top_left_corner', 'top_right_corner'],  # Baselines
                ['bottom_left_singles', 'bottom_right_singles', 'top_left_singles', 'top_right_singles'],  # Singles lines
            ],
            'perpendicular_lines': [
                [['bottom_left_corner', 'top_left_corner'], ['bottom_left_corner', 'bottom_right_corner']],  # Court sides vs baselines
            ],
            'distance_constraints': {
                'court_length': ['bottom_left_corner', 'top_left_corner', 23.77],
                'court_width': ['bottom_left_corner', 'bottom_right_corner', 10.97],
                'singles_width': ['bottom_left_singles', 'bottom_right_singles', 8.23],
                'service_length': ['bottom_left_corner', 'service_line_left', 6.40]
            },
            'height_constraints': {
                'net_posts': ['net_post_left', 'net_post_right', 1.07]
            }
        }
        
        return relationships
    
    def visualize_all_patterns(self, output_dir: Path) -> None:
        """Generate visualizations for all available patterns"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern_name in self.available_patterns.keys():
            self.logger.info(f"Generating visualization for {pattern_name} pattern")
            
            # Get pattern and create visualization
            if pattern_name == 'standard':
                canvas = self.base_pattern.visualize_pattern()
            else:
                # Create a temporary pattern manager for visualization
                pattern = self.get_pattern(pattern_name)
                temp_manager = TennisCourtCalibrationPattern()
                temp_manager.calibration_pattern = pattern
                canvas = temp_manager.visualize_pattern()
            
            # Save visualization
            output_path = output_dir / f"tennis_court_pattern_{pattern_name}.png"
            cv2.imwrite(str(output_path), canvas)
            
            # Also export specifications
            spec_path = output_dir / f"tennis_court_pattern_{pattern_name}_specs.json"
            self.export_pattern_for_detection(pattern_name, spec_path)
        
        self.logger.info(f"All pattern visualizations saved to: {output_dir}")
