import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
import json
import time
from enum import Enum

from .pattern_manager import CalibrationPatternManager
from .calibration_engine import TennisCourtCalibrationEngine, CalibrationResult, CalibrationValidation
from ..frame_selection.selector import CalibrationFrameSelector
from ..camera_analysis.analyzer import ComprehensiveCameraAnalyzer

class WorkflowStage(Enum):
    """Calibration workflow stages"""
    INITIALIZATION = "initialization"
    CAMERA_ANALYSIS = "camera_analysis"
    FRAME_SELECTION = "frame_selection"
    KEYPOINT_DETECTION = "keypoint_detection"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    FINALIZATION = "finalization"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class WorkflowStageResult:
    """Result of a workflow stage execution"""
    stage: WorkflowStage
    status: WorkflowStatus
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str]
    warnings: List[str]

@dataclass
class WorkflowConfiguration:
    """Configuration for calibration workflow"""
    # Data paths
    video_path: Optional[Path]
    frames_dir: Path
    output_dir: Path
    
    # Frame selection parameters
    min_calibration_frames: int = 8
    max_calibration_frames: int = 20
    max_frames_to_analyze: int = 50
    
    # Calibration parameters
    calibration_patterns: List[str] = None
    use_indoor_detector: bool = True
    quality_threshold: float = 2.0
    
    # Workflow options
    perform_camera_analysis: bool = True
    generate_reports: bool = True
    save_intermediate_results: bool = True
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.calibration_patterns is None:
            self.calibration_patterns = ['minimal', 'standard']

@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    configuration: WorkflowConfiguration
    start_time: float
    end_time: Optional[float]
    total_duration: Optional[float]
    overall_status: WorkflowStatus
    stage_results: Dict[WorkflowStage, WorkflowStageResult]
    best_calibration: Optional[CalibrationResult]
    final_report: Dict[str, Any]

class DataclassJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for dataclasses and numpy types"""
    
    def default(self, obj):
        if is_dataclass(obj):
            return self._serialize_dataclass(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (WorkflowStage, WorkflowStatus)):
            return obj.value
        elif hasattr(obj, '__dict__'):
            # For non-dataclass objects with __dict__
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        return super().default(obj)
    
    def _serialize_dataclass(self, obj):
        """Serialize a dataclass instance"""
        result = {}
        for field_name, field_value in obj.__dict__.items():
            try:
                result[field_name] = self.default(field_value)
            except TypeError:
                # If serialization fails, convert to string as fallback
                result[field_name] = str(field_value)
        return result

class TennisCalibrationWorkflow:
    """Complete tennis court camera calibration workflow orchestrator"""
    
    def __init__(self, configuration: WorkflowConfiguration):
        self.config = configuration
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pattern_manager = CalibrationPatternManager()
        self.frame_selector = CalibrationFrameSelector(
            min_calibration_frames=self.config.min_calibration_frames,
            max_calibration_frames=self.config.max_calibration_frames
        )
        self.calibration_engine = TennisCourtCalibrationEngine(
            self.pattern_manager,
            use_indoor_detector=self.config.use_indoor_detector
        )
        
        if self.config.perform_camera_analysis:
            self.camera_analyzer = ComprehensiveCameraAnalyzer()
        
        # Workflow state
        self.workflow_id = f"calibration_{int(time.time())}"
        self.stage_results: Dict[WorkflowStage, WorkflowStageResult] = {}
        self.current_stage: Optional[WorkflowStage] = None
        self.workflow_data: Dict[str, Any] = {}
        
        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.config.output_dir / "reports"
        self.visualizations_dir = self.config.output_dir / "visualizations"
        self.intermediate_dir = self.config.output_dir / "intermediate"
        
        if self.config.save_intermediate_results:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        if self.config.generate_reports:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        if self.config.create_visualizations:
            self.visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    def execute_complete_workflow(self) -> WorkflowResult:
        """Execute the complete calibration workflow"""
        
        self.logger.info(f"Starting tennis court calibration workflow: {self.workflow_id}")
        start_time = time.time()
        overall_status = WorkflowStatus.IN_PROGRESS
        
        try:
            # Stage 1: Initialization
            self._execute_stage(WorkflowStage.INITIALIZATION, self._stage_initialization)
            
            # Stage 2: Camera Analysis (optional)
            if self.config.perform_camera_analysis:
                self._execute_stage(WorkflowStage.CAMERA_ANALYSIS, self._stage_camera_analysis)
            
            # Stage 3: Frame Selection
            self._execute_stage(WorkflowStage.FRAME_SELECTION, self._stage_frame_selection)
            
            # Stage 4: Keypoint Detection (validation step)
            self._execute_stage(WorkflowStage.KEYPOINT_DETECTION, self._stage_keypoint_detection)
            
            # Stage 5: Calibration
            self._execute_stage(WorkflowStage.CALIBRATION, self._stage_calibration)
            
            # Stage 6: Validation
            self._execute_stage(WorkflowStage.VALIDATION, self._stage_validation)
            
            # Stage 7: Finalization
            self._execute_stage(WorkflowStage.FINALIZATION, self._stage_finalization)
            
            overall_status = WorkflowStatus.COMPLETED
            self.logger.info("Workflow completed successfully")
            
        except Exception as e:
            overall_status = WorkflowStatus.FAILED
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            
            # Create error stage result if current stage failed
            if self.current_stage and self.current_stage not in self.stage_results:
                self.stage_results[self.current_stage] = WorkflowStageResult(
                    stage=self.current_stage,
                    status=WorkflowStatus.FAILED,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0.0,
                    success=False,
                    data={},
                    error_message=str(e),
                    warnings=[]
                )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate final report
        final_report = self._generate_final_report()
        
        # Determine best calibration
        best_calibration = self._select_best_calibration()
        
        workflow_result = WorkflowResult(
            workflow_id=self.workflow_id,
            configuration=self.config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            overall_status=overall_status,
            stage_results=self.stage_results,
            best_calibration=best_calibration,
            final_report=final_report
        )
        
        # Save workflow result
        self._save_workflow_result(workflow_result)
        
        return workflow_result
    
    def _execute_stage(self, stage: WorkflowStage, stage_function):
        """Execute a workflow stage with error handling and timing"""
        
        self.current_stage = stage
        self.logger.info(f"Executing stage: {stage.value}")
        
        start_time = time.time()
        warnings = []
        error_message = None
        success = False
        stage_data = {}
        
        try:
            stage_data = stage_function()
            success = True
            self.workflow_data[stage.value] = stage_data
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Stage {stage.value} failed: {e}", exc_info=True)
            raise
        
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self.stage_results[stage] = WorkflowStageResult(
                stage=stage,
                status=WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                data=stage_data,
                error_message=error_message,
                warnings=warnings
            )
            
            self.logger.info(f"Stage {stage.value} completed in {duration:.2f}s")
    
    def _stage_initialization(self) -> Dict[str, Any]:
        """Initialize workflow and validate inputs"""
        
        self.logger.info("Initializing calibration workflow")
        
        # Validate configuration
        if not self.config.frames_dir.exists():
            raise ValueError(f"Frames directory does not exist: {self.config.frames_dir}")
        
        # Count available frames
        frame_files = list(self.config.frames_dir.glob("*.jpg"))
        if len(frame_files) < self.config.min_calibration_frames:
            raise ValueError(f"Insufficient frames: found {len(frame_files)}, "
                           f"need at least {self.config.min_calibration_frames}")
        
        # Initialize calibration patterns
        available_patterns = {}
        for pattern_type in self.config.calibration_patterns:
            try:
                pattern = self.pattern_manager.get_pattern(pattern_type)
                available_patterns[pattern_type] = len(pattern.keypoints)
            except Exception as e:
                self.logger.warning(f"Pattern {pattern_type} not available: {e}")
        
        if not available_patterns:
            raise ValueError("No valid calibration patterns available")
        
        initialization_data = {
            'workflow_id': self.workflow_id,
            'available_frames': len(frame_files),
            'available_patterns': available_patterns,
            'configuration_summary': {
                'frames_dir': str(self.config.frames_dir),
                'output_dir': str(self.config.output_dir),
                'calibration_patterns': self.config.calibration_patterns,
                'use_indoor_detector': self.config.use_indoor_detector
            },
            'system_info': {
                'opencv_version': cv2.__version__,
                'numpy_version': np.__version__
            }
        }
        
        self.logger.info(f"Initialization complete: {len(frame_files)} frames, "
                        f"{len(available_patterns)} patterns")
        
        return initialization_data
    
    def _stage_camera_analysis(self) -> Dict[str, Any]:
        """Analyze camera characteristics"""
        
        self.logger.info("Performing camera analysis")
        
        # Load sample frames for analysis
        frame_files = sorted(self.config.frames_dir.glob("*.jpg"))
        sample_size = min(20, len(frame_files))
        sample_indices = np.linspace(0, len(frame_files)-1, sample_size, dtype=int)
        sample_frames = []
        
        for idx in sample_indices:
            frame = cv2.imread(str(frame_files[idx]))
            if frame is not None:
                sample_frames.append(frame)
        
        if not sample_frames:
            raise ValueError("No valid frames found for camera analysis")
        
        # Perform analysis
        if self.config.video_path and self.config.video_path.exists():
            analysis_result = self.camera_analyzer.analyze_video_camera_characteristics(
                self.config.video_path, sample_frames
            )
        else:
            # Create dummy video metadata for frame-based analysis
            dummy_metadata = {
                'resolution': sample_frames[0].shape[:2][::-1],
                'fps': 30.0,
                'total_frames': len(frame_files),
                'duration': len(frame_files) / 30.0,
                'codec': 'unknown'
            }
            analysis_result = self.camera_analyzer.analyze_frames_camera_characteristics(
                sample_frames, dummy_metadata
            )
        
        # Save camera analysis results
        if self.config.save_intermediate_results:
            analysis_path = self.intermediate_dir / "camera_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis_result, f, indent=2, cls=DataclassJSONEncoder)
        
        camera_analysis_data = {
            'analysis_result': analysis_result,
            'sample_frames_count': len(sample_frames),
            'analysis_quality': analysis_result.get('analysis_summary', {}).get('overall_quality', 'unknown')
        }
        
        self.logger.info("Camera analysis completed")
        return camera_analysis_data
    
    def _stage_frame_selection(self) -> Dict[str, Any]:
        """Select optimal frames for calibration"""
        
        self.logger.info("Selecting calibration frames")
        
        # Perform frame selection
        selection_results = self.frame_selector.select_calibration_frames(
            frames_dir=self.config.frames_dir,
            max_analyze=self.config.max_frames_to_analyze
        )
        
        # Save frame selection results
        if self.config.save_intermediate_results:
            selection_path = self.intermediate_dir / "frame_selection_results.json"
            self.frame_selector.save_selection_results(selection_results, selection_path)
        
        selected_frame_ids = selection_results['selection_report']['selected_frame_ids']
        
        frame_selection_data = {
            'selection_summary': {
                'total_analyzed': len(selection_results.get('all_analyses', [])),
                'selected_count': len(selected_frame_ids),
                'selection_quality': selection_results['selection_report']['recommendation']
            },
            'selected_frame_ids': selected_frame_ids
        }
        
        self.logger.info(f"Frame selection completed: {len(selected_frame_ids)} frames selected")
        return frame_selection_data
    
    def _stage_keypoint_detection(self) -> Dict[str, Any]:
        """Validate keypoint detection on selected frames"""
        
        self.logger.info("Validating keypoint detection")
        
        selected_frame_ids = self.workflow_data['frame_selection']['selected_frame_ids']
        
        # Find frame files
        frame_paths = []
        for frame_id in selected_frame_ids:
            possible_files = [
                self.config.frames_dir / f"{frame_id}.jpg",
                self.config.frames_dir / f"frame_{frame_id}.jpg",
                self.config.frames_dir / f"processed_frame_{frame_id}.jpg"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    frame_paths.append(file_path)
                    break
        
        if len(frame_paths) < self.config.min_calibration_frames:
            raise ValueError(f"Insufficient frame files found: {len(frame_paths)}")
        
        # Test keypoint detection on each pattern
        detection_results = {}
        
        for pattern_type in self.config.calibration_patterns:
            pattern_results = []
            
            for frame_path in frame_paths[:5]:  # Test on first 5 frames
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    detection_result = self.calibration_engine.keypoint_detector.detect_keypoints_in_frame(
                        frame, frame_path.stem, pattern_type
                    )
                    pattern_results.append({
                        'frame_id': frame_path.stem,
                        'detection_count': detection_result.detection_count,
                        'avg_confidence': float(detection_result.avg_confidence),
                        'quality': detection_result.detection_quality
                    })
            
            detection_results[pattern_type] = {
                'results': pattern_results,
                'avg_detections': float(np.mean([r['detection_count'] for r in pattern_results])),
                'avg_confidence': float(np.mean([r['avg_confidence'] for r in pattern_results]))
            }
        
        keypoint_detection_data = {
            'detection_results': detection_results,
            'frame_paths': [str(p) for p in frame_paths],
            'validation_summary': self._summarize_detection_validation(detection_results)
        }
        
        self.logger.info("Keypoint detection validation completed")
        return keypoint_detection_data
    
    def _stage_calibration(self) -> Dict[str, Any]:
        """Perform camera calibration with multiple patterns"""
        
        self.logger.info("Performing camera calibration")
        
        frame_paths = [Path(p) for p in self.workflow_data['keypoint_detection']['frame_paths']]
        calibration_results = {}
        
        # Calibrate with each pattern
        for pattern_type in self.config.calibration_patterns:
            self.logger.info(f"Calibrating with {pattern_type} pattern")
            
            try:
                calibration_result = self.calibration_engine.calibrate_camera_from_frames(
                    frame_paths, pattern_type=pattern_type
                )
                
                calibration_results[pattern_type] = {
                    'calibration_successful': True,
                    'reprojection_error': float(calibration_result.reprojection_error),
                    'quality': calibration_result.calibration_quality,
                    'method': calibration_result.calibration_method,
                    'keypoint_count': calibration_result.keypoint_count,
                    'used_frames': len(calibration_result.used_frames),
                    'meets_quality_threshold': calibration_result.reprojection_error <= self.config.quality_threshold,
                    'calibration_object': calibration_result  # Store the actual object for later use
                }
                
                # Save individual calibration results
                if self.config.save_intermediate_results:
                    calib_path = self.intermediate_dir / f"calibration_{pattern_type}.json"
                    validation = CalibrationValidation(
                        reprojection_error=calibration_result.reprojection_error,
                        error_std=0.0,
                        max_error=calibration_result.reprojection_error,
                        valid_frame_percentage=100.0,
                        keypoint_distribution_score=75.0,
                        overall_quality_score=80.0
                    )
                    self.calibration_engine.save_calibration_result(
                        calibration_result, validation, calib_path
                    )
                
                self.logger.info(f"Calibration with {pattern_type}: "
                                f"error={calibration_result.reprojection_error:.3f}px, "
                                f"quality={calibration_result.calibration_quality}")
                
            except Exception as e:
                self.logger.error(f"Calibration failed for pattern {pattern_type}: {e}")
                calibration_results[pattern_type] = {
                    'calibration_successful': False,
                    'error': str(e),
                    'meets_quality_threshold': False
                }
        
        # Check if any calibration meets quality threshold
        successful_calibrations = [
            (pattern, result) for pattern, result in calibration_results.items()
            if result.get('meets_quality_threshold', False)
        ]
        
        if not successful_calibrations:
            self.logger.warning("No calibrations meet the quality threshold")
        
        calibration_data = {
            'calibration_results': calibration_results,
            'successful_calibrations': len(successful_calibrations),
            'best_pattern': self._find_best_calibration_pattern(calibration_results)
        }
        
        self.logger.info(f"Calibration completed: {len(successful_calibrations)} successful")
        return calibration_data
    
    def _stage_validation(self) -> Dict[str, Any]:
        """Comprehensive validation of calibration results"""
        
        self.logger.info("Validating calibration results")
        
        calibration_results = self.workflow_data['calibration']['calibration_results']
        
        # Validation metrics
        validation_summary = {
            'total_patterns_tested': len(calibration_results),
            'successful_calibrations': 0,
            'failed_calibrations': 0,
            'quality_distribution': {},
            'reprojection_errors': {},
            'recommendations': []
        }
        
        for pattern_type, result in calibration_results.items():
            if result.get('calibration_successful'):
                validation_summary['successful_calibrations'] += 1
                validation_summary['reprojection_errors'][pattern_type] = result['reprojection_error']
                
                quality = result['quality']
                validation_summary['quality_distribution'][quality] = \
                    validation_summary['quality_distribution'].get(quality, 0) + 1
            else:
                validation_summary['failed_calibrations'] += 1
        
        # Generate recommendations
        validation_summary['recommendations'] = self._generate_calibration_recommendations(
            calibration_results
        )
        
        validation_data = {
            'validation_summary': validation_summary,
            'overall_success': validation_summary['successful_calibrations'] > 0,
            'best_calibration_found': validation_summary['successful_calibrations'] > 0
        }
        
        self.logger.info("Validation completed")
        return validation_data
    
    def _stage_finalization(self) -> Dict[str, Any]:
        """Finalize workflow and generate reports"""
        
        self.logger.info("Finalizing calibration workflow")
        
        # Copy best calibration to output
        best_calibration = self._select_best_calibration()
        if best_calibration:
            self._save_final_calibration(best_calibration)
        
        finalization_data = {
            'final_calibration_saved': best_calibration is not None,
            'reports_generated': self.config.generate_reports,
            'visualizations_created': self.config.create_visualizations,
            'workflow_success': best_calibration is not None
        }
        
        self.logger.info("Workflow finalization completed")
        return finalization_data
    
    def _summarize_detection_validation(self, detection_results: Dict) -> Dict[str, Any]:
        """Summarize keypoint detection validation results"""
        
        summary = {
            'patterns_tested': len(detection_results),
            'avg_detections_per_pattern': {},
            'overall_detection_quality': 'poor'
        }
        
        total_avg_detections = 0
        for pattern_type, results in detection_results.items():
            avg_detections = results['avg_detections']
            summary['avg_detections_per_pattern'][pattern_type] = float(avg_detections)
            total_avg_detections += avg_detections
        
        if summary['patterns_tested'] > 0:
            overall_avg = total_avg_detections / summary['patterns_tested']
            if overall_avg >= 6:
                summary['overall_detection_quality'] = 'excellent'
            elif overall_avg >= 4:
                summary['overall_detection_quality'] = 'good'
            elif overall_avg >= 3:
                summary['overall_detection_quality'] = 'fair'
        
        return summary
    
    def _find_best_calibration_pattern(self, calibration_results: Dict) -> Optional[str]:
        """Find the calibration pattern with the best results"""
        
        best_pattern = None
        best_error = float('inf')
        
        for pattern_type, result in calibration_results.items():
            if (result.get('calibration_successful') and 
                result.get('meets_quality_threshold') and 
                result.get('reprojection_error', float('inf')) < best_error):
                best_error = result['reprojection_error']
                best_pattern = pattern_type
        
        return best_pattern
    
    def _select_best_calibration(self) -> Optional[CalibrationResult]:
        """Select the best calibration result from all patterns"""
        
        if 'calibration' not in self.workflow_data:
            return None
        
        calibration_results = self.workflow_data['calibration']['calibration_results']
        best_pattern = self._find_best_calibration_pattern(calibration_results)
        
        if (best_pattern and 
            calibration_results[best_pattern].get('calibration_successful') and
            'calibration_object' in calibration_results[best_pattern]):
            return calibration_results[best_pattern]['calibration_object']
        
        return None
    
    def _generate_calibration_recommendations(self, calibration_results: Dict) -> List[str]:
        """Generate recommendations based on calibration results"""
        
        recommendations = []
        
        successful_count = sum(1 for result in calibration_results.values() 
                             if result.get('meets_quality_threshold', False))
        
        if successful_count == 0:
            recommendations.append("No calibrations met the quality threshold. Consider:")
            recommendations.append("- Capturing frames with better court line visibility")
            recommendations.append("- Using frames from different camera angles")
            recommendations.append("- Improving lighting conditions")
            recommendations.append("- Checking for camera shake or motion blur")
        elif successful_count == 1:
            recommendations.append("One successful calibration found. Consider capturing additional footage for validation.")
        else:
            recommendations.append("Multiple successful calibrations found. System is well-calibrated.")
        
        # Check reprojection errors
        errors = [result.get('reprojection_error', float('inf'))
                 for result in calibration_results.values() 
                 if result.get('calibration_successful')]
        
        if errors:
            min_error = min(errors)
            if min_error < 0.5:
                recommendations.append("Excellent calibration accuracy achieved (< 0.5 pixels).")
            elif min_error < 1.0:
                recommendations.append("Good calibration accuracy achieved (< 1.0 pixels).")
            elif min_error < 2.0:
                recommendations.append("Acceptable calibration accuracy achieved (< 2.0 pixels).")
        
        return recommendations
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final workflow report"""
        
        total_duration = sum(
            result.duration for result in self.stage_results.values() 
            if result.duration is not None
        )
        
        successful_stages = sum(1 for result in self.stage_results.values() if result.success)
        best_calibration = self._select_best_calibration()
        
        final_report = {
            'workflow_summary': {
                'workflow_id': self.workflow_id,
                'total_duration': float(total_duration),
                'successful_stages': successful_stages,
                'total_stages': len(self.stage_results),
                'overall_success': best_calibration is not None
            },
            'stage_performance': {
                stage.value: {
                    'duration': float(result.duration) if result.duration else 0.0,
                    'success': result.success,
                    'warnings_count': len(result.warnings)
                }
                for stage, result in self.stage_results.items()
            },
            'calibration_summary': self._generate_calibration_summary(),
            'recommendations': self._generate_final_recommendations(),
            'quality_assessment': self._assess_overall_quality()
        }
        
        return final_report
    
    def _generate_calibration_summary(self) -> Dict[str, Any]:
        """Generate calibration summary for final report"""
        
        if 'calibration' not in self.workflow_data:
            return {'status': 'not_performed'}
        
        calibration_results = self.workflow_data['calibration']['calibration_results']
        best_calibration = self._select_best_calibration()
        
        summary = {
            'patterns_tested': len(calibration_results),
            'successful_calibrations': len([r for r in calibration_results.values() 
                                          if r.get('meets_quality_threshold', False)]),
            'best_calibration': None
        }
        
        if best_calibration:
            summary['best_calibration'] = {
                'reprojection_error': float(best_calibration.reprojection_error),
                'quality': best_calibration.calibration_quality,
                'method': best_calibration.calibration_method,
                'keypoint_count': best_calibration.keypoint_count,
                'used_frames': len(best_calibration.used_frames)
            }
        
        return summary
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations for the entire workflow"""
        
        recommendations = []
        best_calibration = self._select_best_calibration()
        
        if best_calibration:
            error = best_calibration.reprojection_error
            if error < 0.5:
                recommendations.append("Outstanding calibration quality achieved - suitable for professional applications.")
            elif error < 1.0:
                recommendations.append("Excellent calibration quality - suitable for accurate sports analysis.")
            elif error < 2.0:
                recommendations.append("Good calibration quality - suitable for general video analysis.")
            else:
                recommendations.append("Calibration completed but quality could be improved.")
                
            best_pattern = self._find_best_calibration_pattern(
                self.workflow_data.get('calibration', {}).get('calibration_results', {})
            )
            if best_pattern:
                recommendations.append(f"Recommended calibration pattern: {best_pattern}")
        else:
            recommendations.append("Calibration was not successful. Please review the detailed error messages and try again with improved input data.")
        
        return recommendations
    
    def _assess_overall_quality(self) -> str:
        """Assess overall workflow quality"""
        
        best_calibration = self._select_best_calibration()
        
        if not best_calibration:
            return 'failed'
        
        error = best_calibration.reprojection_error
        
        if error < 0.5:
            return 'outstanding'
        elif error < 1.0:
            return 'excellent'
        elif error < 2.0:
            return 'good'
        else:
            return 'acceptable'
    
    def _save_workflow_result(self, workflow_result: WorkflowResult):
        """Save complete workflow result with robust serialization"""
        
        self.logger.info("Saving workflow result...")
        
        output_path = self.config.output_dir / f"workflow_result_{self.workflow_id}.json"
        
        try:
            # Use the enhanced JSON encoder for robust serialization
            with open(output_path, 'w') as f:
                json.dump(workflow_result, f, indent=2, cls=DataclassJSONEncoder)
            
            self.logger.info(f"Workflow result saved to: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Enhanced serialization failed: {e}. Trying simplified approach.")
            
            # Fallback: Create a simplified version of the result
            try:
                simplified_result = {
                    'workflow_id': workflow_result.workflow_id,
                    'overall_status': workflow_result.overall_status.value,
                    'total_duration': float(workflow_result.total_duration) if workflow_result.total_duration else 0.0,
                    'stage_results': {
                        stage.value: {
                            'success': result.success,
                            'duration': float(result.duration) if result.duration else 0.0,
                            'error_message': result.error_message
                        }
                        for stage, result in workflow_result.stage_results.items()
                    },
                    'best_calibration': {
                        'reprojection_error': float(workflow_result.best_calibration.reprojection_error),
                        'quality': workflow_result.best_calibration.calibration_quality,
                        'method': workflow_result.best_calibration.calibration_method
                    } if workflow_result.best_calibration else None,
                    'final_report': workflow_result.final_report
                }
                
                with open(output_path, 'w') as f:
                    json.dump(simplified_result, f, indent=2)
                
                self.logger.info(f"Simplified workflow result saved to: {output_path}")
                
            except Exception as final_error:
                self.logger.error(f"All serialization attempts failed: {final_error}")
                # At least save basic info
                basic_result = {
                    'workflow_id': workflow_result.workflow_id,
                    'overall_status': workflow_result.overall_status.value,
                    'error': 'Serialization failed - check logs for details'
                }
                
                with open(output_path, 'w') as f:
                    json.dump(basic_result, f, indent=2)
    
    def _save_final_calibration(self, calibration_result: CalibrationResult):
        """Save the final calibration result"""
        
        output_path = self.config.output_dir / "final_calibration.npz"
        np.savez(
            str(output_path),
            camera_matrix=calibration_result.camera_matrix,
            dist_coeffs=calibration_result.distortion_coefficients,
            rvecs=calibration_result.rotation_vectors,
            tvecs=calibration_result.translation_vectors
        )
        
        self.logger.info(f"Final calibration saved to: {output_path}")
