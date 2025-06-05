import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

from .workflow_orchestrator import WorkflowConfiguration

class WorkflowConfigurationManager:
    """Manage workflow configurations and presets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def create_default_config(
        frames_dir: Path,
        output_dir: Path,
        video_path: Optional[Path] = None
    ) -> WorkflowConfiguration:
        """Create default workflow configuration"""
        
        return WorkflowConfiguration(
            video_path=video_path,
            frames_dir=frames_dir,
            output_dir=output_dir,
            min_calibration_frames=8,
            max_calibration_frames=15,
            max_frames_to_analyze=50,
            calibration_patterns=['minimal', 'standard'],
            use_indoor_detector=True,
            quality_threshold=2.0,
            perform_camera_analysis=True,
            generate_reports=True,
            save_intermediate_results=True,
            create_visualizations=True
        )
    
    @staticmethod
    def create_indoor_court_config(
        frames_dir: Path,
        output_dir: Path,
        video_path: Optional[Path] = None
    ) -> WorkflowConfiguration:
        """Create configuration optimized for indoor courts"""
        
        return WorkflowConfiguration(
            video_path=video_path,
            frames_dir=frames_dir,
            output_dir=output_dir,
            min_calibration_frames=6,
            max_calibration_frames=12,
            max_frames_to_analyze=30,
            calibration_patterns=['minimal', 'broadcast'],
            use_indoor_detector=True,
            quality_threshold=1.0,
            perform_camera_analysis=True,
            generate_reports=True,
            save_intermediate_results=True,
            create_visualizations=True
        )
    
    def validate_configuration(self, config: WorkflowConfiguration) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        # Check required paths
        if not config.frames_dir.exists():
            issues.append(f"Frames directory does not exist: {config.frames_dir}")
        
        if config.video_path and not config.video_path.exists():
            issues.append(f"Video file does not exist: {config.video_path}")
        
        # Check frame count requirements
        if config.min_calibration_frames > config.max_calibration_frames:
            issues.append("min_calibration_frames cannot be greater than max_calibration_frames")
        
        if config.min_calibration_frames < 3:
            issues.append("min_calibration_frames should be at least 3")
        
        # Check calibration patterns
        valid_patterns = ['minimal', 'standard', 'broadcast', 'singles', 'robust']
        for pattern in config.calibration_patterns:
            if pattern not in valid_patterns:
                issues.append(f"Invalid calibration pattern: {pattern}")
        
        # Check quality threshold
        if config.quality_threshold <= 0:
            issues.append("quality_threshold must be positive")
        
        return issues
