#!/usr/bin/env python3

from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.calibration.workflow_orchestrator import TennisCalibrationWorkflow
from tennis.calibration.workflow_config import WorkflowConfigurationManager

def main():
    """Example of complete tennis court calibration workflow"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Complete Tennis Court Calibration Workflow")
    print("="*50)
    
    # Define paths
    frames_dir = Path("data/processed_frames")
    output_dir = Path("data/calibration_workflow_results")
    video_path = Path("data/raw_videos/7 YEAR OLD TENNIS PRODIGY- M3.mp4")
    
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        logger.error("Please run frame extraction first")
        return
    
    # Create configuration manager
    config_manager = WorkflowConfigurationManager()
    
    # Create indoor court configuration (based on your video)
    config = config_manager.create_indoor_court_config(
        frames_dir=frames_dir,
        output_dir=output_dir,
        video_path=video_path if video_path.exists() else None
    )
    
    # Customize configuration for your specific needs
    config.max_frames_to_analyze = 30  # Analyze fewer frames for speed
    config.quality_threshold = 1.0     # Stricter quality requirement
    config.calibration_patterns = ['minimal', 'standard', 'broadcast']
    
    print(f"Configuration:")
    print(f"  Frames directory: {config.frames_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Video path: {config.video_path}")
    print(f"  Calibration patterns: {config.calibration_patterns}")
    print(f"  Quality threshold: {config.quality_threshold} pixels")
    
    # Validate configuration
    issues = config_manager.validate_configuration(config)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return
    
    # Create and execute workflow
    workflow = TennisCalibrationWorkflow(config)
    
    try:
        print(f"\nExecuting complete calibration workflow...")
        result = workflow.execute_complete_workflow()
        
        # Display results
        print(f"\n{'='*60}")
        print("WORKFLOW EXECUTION RESULTS")
        print(f"{'='*60}")
        
        print(f"Workflow ID: {result.workflow_id}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Total Duration: {result.total_duration:.1f} seconds")
        print(f"Successful Stages: {sum(1 for r in result.stage_results.values() if r.success)}/{len(result.stage_results)}")
        
        # Stage-by-stage results
        print(f"\nStage Execution Details:")
        for stage, stage_result in result.stage_results.items():
            status = "✓ PASSED" if stage_result.success else "✗ FAILED"
            duration = f"{stage_result.duration:.1f}s" if stage_result.duration else "N/A"
            print(f"  {status} | {stage.value.ljust(20)} | {duration}")
            
            if stage_result.error_message:
                print(f"    Error: {stage_result.error_message}")
            
            if stage_result.warnings:
                for warning in stage_result.warnings:
                    print(f"    Warning: {warning}")
        
        # Best calibration results
        if result.best_calibration:
            print(f"\n{'='*40}")
            print("BEST CALIBRATION ACHIEVED")
            print(f"{'='*40}")
            
            calib = result.best_calibration
            print(f"Reprojection Error: {calib.reprojection_error:.3f} pixels")
            print(f"Quality Rating: {calib.calibration_quality}")
            print(f"Calibration Method: {calib.calibration_method}")
            print(f"Frames Used: {len(calib.used_frames)}")
            print(f"Keypoints Used: {calib.keypoint_count}")
            
            print(f"\nCamera Matrix:")
            for i, row in enumerate(calib.camera_matrix):
                print(f"  [{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]")
            
            print(f"\nDistortion Coefficients:")
            print(f"  {calib.distortion_coefficients}")
            
        else:
            print(f"\n{'='*40}")
            print("NO SUCCESSFUL CALIBRATION")
            print(f"{'='*40}")
            print("Review the error messages above and try:")
            print("- Improving frame quality")
            print("- Using different camera angles")
            print("- Checking court line visibility")
        
        # Final recommendations
        if result.final_report.get('recommendations'):
            print(f"\nFinal Recommendations:")
            for i, rec in enumerate(result.final_report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Output locations
        print(f"\nOutput Files:")
        print(f"  Main results: {config.output_dir}")
        print(f"  Workflow log: calibration_workflow.log")
        
        if config.generate_reports:
            print(f"  Reports: {config.output_dir / 'reports'}")
        
        if config.create_visualizations:
            print(f"  Visualizations: {config.output_dir / 'visualizations'}")
        
        if config.save_intermediate_results:
            print(f"  Intermediate files: {config.output_dir / 'intermediate'}")
        
        print(f"\nWorkflow completed {'successfully' if result.overall_status.value == 'completed' else 'with errors'}!")
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
