import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .pattern_design import TennisCourtCalibrationPattern, CourtKeypoint, TennisCourtPattern

class TennisCourtPatternVariants:
    """Alternative calibration patterns for different scenarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_pattern = TennisCourtCalibrationPattern()
    
    def get_minimal_pattern(self) -> TennisCourtPattern:
        """6-point minimal pattern for basic calibration"""[6]
        
        minimal_keypoints = [
            # Four corners of the court (ground plane)
            self.base_pattern.get_keypoint_by_name("bottom_left_corner"),
            self.base_pattern.get_keypoint_by_name("bottom_right_corner"), 
            self.base_pattern.get_keypoint_by_name("top_left_corner"),
            self.base_pattern.get_keypoint_by_name("top_right_corner"),
            
            # Two net posts (height information)
            self.base_pattern.get_keypoint_by_name("net_post_left"),
            self.base_pattern.get_keypoint_by_name("net_post_right")
        ]
        
        return TennisCourtPattern(
            keypoints=[kp for kp in minimal_keypoints if kp is not None],
            court_dimensions=self.base_pattern.court_dimensions,
            coordinate_system="ITF_standard_minimal",
            reference_frame="bottom_left_origin"
        )
    
    def get_singles_pattern(self) -> TennisCourtPattern:
        """Pattern optimized for singles court detection"""
        
        singles_keypoints = [
            # Singles court corners
            self.base_pattern.get_keypoint_by_name("bottom_left_singles"),
            self.base_pattern.get_keypoint_by_name("bottom_right_singles"),
            self.base_pattern.get_keypoint_by_name("top_left_singles"), 
            self.base_pattern.get_keypoint_by_name("top_right_singles"),
            
            # Service line intersections
            self.base_pattern.get_keypoint_by_name("service_line_left"),
            self.base_pattern.get_keypoint_by_name("service_line_right"),
            self.base_pattern.get_keypoint_by_name("service_line_left_far"),
            self.base_pattern.get_keypoint_by_name("service_line_right_far"),
            
            # Net posts for height
            self.base_pattern.get_keypoint_by_name("net_post_left"),
            self.base_pattern.get_keypoint_by_name("net_post_right")
        ]
        
        return TennisCourtPattern(
            keypoints=[kp for kp in singles_keypoints if kp is not None],
            court_dimensions=self.base_pattern.court_dimensions,
            coordinate_system="ITF_standard_singles",
            reference_frame="bottom_left_origin"
        )
    
    def get_robust_pattern(self) -> TennisCourtPattern:
        """Enhanced pattern with additional redundant points"""
        
        # Start with all base keypoints
        robust_keypoints = list(self.base_pattern.calibration_pattern.keypoints)
        
        # Add center court points for additional constraints
        center_keypoints = [
            CourtKeypoint(
                id=15, name="center_baseline_bottom",
                world_coords=(0.0, self.base_pattern.court_dimensions['width_doubles']/2, 0.0),
                description="Center of bottom baseline",
                importance="optional"
            ),
            CourtKeypoint(
                id=16, name="center_baseline_top",
                world_coords=(self.base_pattern.court_dimensions['length'], self.base_pattern.court_dimensions['width_doubles']/2, 0.0),
                description="Center of top baseline", 
                importance="optional"
            ),
            CourtKeypoint(
                id=17, name="net_center",
                world_coords=(self.base_pattern.court_dimensions['length']/2, self.base_pattern.court_dimensions['width_doubles']/2, self.base_pattern.court_dimensions['net_height_center']),
                description="Center of net at lowest point",
                importance="optional"
            )
        ]
        
        robust_keypoints.extend(center_keypoints)
        
        return TennisCourtPattern(
            keypoints=robust_keypoints,
            court_dimensions=self.base_pattern.court_dimensions,
            coordinate_system="ITF_standard_robust",
            reference_frame="bottom_left_origin"
        )
    
    def get_broadcast_pattern(self) -> TennisCourtPattern:
        """Pattern optimized for broadcast video (typically shows one half)"""
        
        # Focus on one half of the court that's typically visible in broadcast
        broadcast_keypoints = [
            # Near court half (bottom half in our coordinate system)
            self.base_pattern.get_keypoint_by_name("bottom_left_corner"),
            self.base_pattern.get_keypoint_by_name("bottom_right_corner"),
            self.base_pattern.get_keypoint_by_name("bottom_left_singles"),
            self.base_pattern.get_keypoint_by_name("bottom_right_singles"),
            
            # Service area
            self.base_pattern.get_keypoint_by_name("service_line_left"),
            self.base_pattern.get_keypoint_by_name("service_line_right"),
            
            # Net for perspective reference
            self.base_pattern.get_keypoint_by_name("net_post_left"),
            self.base_pattern.get_keypoint_by_name("net_post_right"),
            
            # Add center service line intersection points
            CourtKeypoint(
                id=18, name="center_service_bottom",
                world_coords=(self.base_pattern.court_dimensions['service_box_length'], self.base_pattern.court_dimensions['width_doubles']/2, 0.0),
                description="Center service line at near service line",
                importance="important"
            )
        ]
        
        return TennisCourtPattern(
            keypoints=[kp for kp in broadcast_keypoints if kp is not None],
            court_dimensions=self.base_pattern.court_dimensions,
            coordinate_system="ITF_standard_broadcast",
            reference_frame="bottom_left_origin"
        )
