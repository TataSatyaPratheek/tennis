import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
import json

from .metadata_extractor import VideoMetadataExtractor, CameraMetadata
from .motion_analyzer import CameraMotionAnalyzer, CameraMotion
from .lighting_analyzer import LightingConditionAnalyzer, LightingConditions
from .position_estimator import CameraPositionEstimator, CameraPosition

class ComprehensiveCameraAnalyzer:
    """Complete camera analysis pipeline for tennis video"""
    
    def __init__(self):
        self.metadata_extractor = VideoMetadataExtractor()
        self.motion_analyzer = CameraMotionAnalyzer()
        self.lighting_analyzer = LightingConditionAnalyzer()
        self.position_estimator = CameraPositionEstimator()
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_video_camera_characteristics(self, 
                                           video_path: Path,
                                           sample_frames: List[np.ndarray]) -> Dict[str, Any]:
        """Complete camera analysis for tennis video"""
        
        self.logger.info("Starting comprehensive camera analysis")
        
        # Extract basic metadata
        metadata = self.metadata_extractor.extract_basic_metadata(video_path)
        dynamic_params = self.metadata_extractor.analyze_dynamic_parameters(video_path)
        
        # Analyze camera motion
        motion_data = self.motion_analyzer.analyze_camera_motion(sample_frames)
        
        # Analyze lighting conditions
        lighting_conditions = self.lighting_analyzer.analyze_lighting_conditions(sample_frames)
        
        # Estimate camera position (using first frame as reference)
        if sample_frames:
            camera_position = self.position_estimator.estimate_camera_position(sample_frames[0])
        else:
            camera_position = None
        
        # Compile comprehensive analysis
        analysis_result = {
            'metadata': {
                'resolution': metadata.resolution,
                'fps': metadata.fps,
                'total_frames': metadata.total_frames,
                'duration': metadata.duration,
                'codec': metadata.codec,
                'aspect_ratio': metadata.aspect_ratio
            },
            'dynamic_parameters': dynamic_params,
            'motion_analysis': {
                'average_motion_magnitude': float(np.mean([m.motion_magnitude for m in motion_data])) if motion_data else 0.0,
                'stability_score': float(np.mean([m.stability_score for m in motion_data])) if motion_data else 1.0,
                'dominant_motion_type': self._get_dominant_motion_type(motion_data),
                'motion_frames': len(motion_data)
            },
            'lighting_conditions': {
                'overall_brightness': lighting_conditions.overall_brightness,
                'brightness_uniformity': lighting_conditions.brightness_uniformity,
                'contrast_ratio': lighting_conditions.contrast_ratio,
                'lighting_quality': lighting_conditions.lighting_quality,
                'shadow_areas': lighting_conditions.shadow_areas,
                'overexposed_areas': lighting_conditions.overexposed_areas
            },
            'camera_position': {
                'height_estimate': camera_position.height_estimate if camera_position else 6.0,
                'distance_estimate': camera_position.distance_estimate if camera_position else 20.0,
                'angle_horizontal': camera_position.angle_horizontal if camera_position else 45.0,
                'angle_vertical': camera_position.angle_vertical if camera_position else 0.0,
                'court_coverage': camera_position.court_coverage if camera_position else 50.0,
                'position_confidence': camera_position.position_confidence if camera_position else 0.5
            }
        }
        
        # Add analysis summary
        analysis_result['analysis_summary'] = self._generate_analysis_summary(analysis_result)
        
        self.logger.info("Camera analysis complete")
        return analysis_result
    
    def _get_dominant_motion_type(self, motion_data: List[CameraMotion]) -> str:
        """Determine dominant motion type across all frames"""
        
        if not motion_data:
            return 'unknown'
        
        motion_types = [m.motion_type for m in motion_data]
        
        # Count occurrences
        type_counts = {}
        for motion_type in motion_types:
            type_counts[motion_type] = type_counts.get(motion_type, 0) + 1
        
        # Return most common type
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable analysis summary"""
        
        summary = {}
        
        # Video quality assessment
        lighting_quality = analysis['lighting_conditions']['lighting_quality']
        stability_score = analysis['motion_analysis']['stability_score']
        
        if lighting_quality in ['excellent', 'good'] and stability_score > 0.7:
            summary['overall_quality'] = 'excellent'
        elif lighting_quality in ['good', 'fair'] and stability_score > 0.5:
            summary['overall_quality'] = 'good'
        elif lighting_quality == 'fair' or stability_score > 0.3:
            summary['overall_quality'] = 'fair'
        else:
            summary['overall_quality'] = 'poor'
        
        # Camera setup assessment
        height = analysis['camera_position']['height_estimate']
        distance = analysis['camera_position']['distance_estimate']
        
        if 4 <= height <= 8 and 10 <= distance <= 30:
            summary['camera_setup'] = 'optimal'
        elif 2 <= height <= 12 and 5 <= distance <= 50:
            summary['camera_setup'] = 'acceptable'
        else:
            summary['camera_setup'] = 'suboptimal'
        
        # Motion characteristics
        motion_type = analysis['motion_analysis']['dominant_motion_type']
        if motion_type == 'static':
            summary['camera_stability'] = 'excellent'
        elif motion_type in ['pan', 'tilt']:
            summary['camera_stability'] = 'good'
        else:
            summary['camera_stability'] = 'poor'
        
        return summary
    
    def save_analysis_report(self, analysis: Dict[str, Any], output_path: Path) -> None:
        """Save analysis report to JSON file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report saved to: {output_path}")
