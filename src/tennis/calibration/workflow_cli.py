import argparse
import sys
from pathlib import Path
import logging

from .workflow_orchestrator import TennisCalibrationWorkflow
from .workflow_config import WorkflowConfigurationManager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('calibration_workflow.log')
        ]
    )

def create_argument_parser():
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Tennis Court Camera Calibration Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calibration
  python -m tennis.calibration.workflow_cli --frames-dir data/frames --output-dir results
  
  # Indoor court with video
  python -m tennis.calibration.workflow_cli --frames-dir data/frames --output-dir results --video data/video.mp4 --preset indoor
  
  # Fast calibration
  python -m tennis.calibration.workflow_cli --frames-dir data/frames --output-dir results --preset fast --no-reports
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--frames-dir', 
        type=Path, 
        required=True,
        help='Directory containing extracted frames'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        required=True,
        help='Output directory for calibration results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--video', 
        type=Path,
        help='Original video file (optional, for camera analysis)'
    )
    
    parser.add_argument(
        '--config', 
        type=Path,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--preset', 
        choices=['default', 'indoor', 'outdoor', 'fast'],
        default='default',
        help='Configuration preset to use'
    )
    
    # Calibration parameters
    parser.add_argument(
        '--min-frames', 
        type=int, 
        default=8,
        help='Minimum number of calibration frames'
    )
    
    parser.add_argument(
        '--max-frames', 
        type=int, 
        default=15,
        help='Maximum number of calibration frames'
    )
    
    parser.add_argument(
        '--max-analyze', 
        type=int, 
        default=50,
        help='Maximum number of frames to analyze'
    )
    
    parser.add_argument(
        '--quality-threshold', 
        type=float, 
        default=2.0,
        help='Maximum acceptable reprojection error (pixels)'
    )
    
    parser.add_argument(
        '--patterns', 
        nargs='+',
        choices=['minimal', 'standard', 'broadcast', 'singles', 'robust'],
        default=['minimal', 'standard'],
        help='Calibration patterns to test'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-camera-analysis', 
        action='store_true',
        help='Skip camera analysis stage'
    )
    
    parser.add_argument(
        '--no-reports', 
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--no-visualizations', 
        action='store_true',
        help='Skip visualization creation'
    )
    
    parser.add_argument(
        '--no-intermediate', 
        action='store_true',
        help='Do not save intermediate results'
    )
    
    parser.add_argument(
        '--outdoor-detector', 
        action='store_true',
        help='Use outdoor court detector instead of indoor'
    )
    
    # Utility options
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Validate configuration without running workflow'
    )
    
    return parser

def main():
    """Main CLI entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Create configuration manager
    config_manager = WorkflowConfigurationManager()
    
    try:
        # Load or create configuration
        if args.config and args.config.exists():
            logger.info(f"Loading configuration from: {args.config}")
            config = config_manager.load_configuration(args.config)
        else:
            # Create configuration based on preset
            logger.info(f"Creating {args.preset} configuration")
            
            if args.preset == 'indoor':
                config = config_manager.create_indoor_court_config(
                    args.frames_dir, args.output_dir, args.video
                )
            elif args.preset == 'outdoor':
                config = config_manager.create_outdoor_court_config(
                    args.frames_dir, args.output_dir, args.video
                )
            elif args.preset == 'fast':
                config = config_manager.create_fast_config(
                    args.frames_dir, args.output_dir, args.video
                )
            else:  # default
                config = config_manager.create_default_config(
                    args.frames_dir, args.output_dir, args.video
                )
        
        # Override configuration with command line arguments
        config.min_calibration_frames = args.min_frames
        config.max_calibration_frames = args.max_frames
        config.max_frames_to_analyze = args.max_analyze
        config.quality_threshold = args.quality_threshold
        config.calibration_patterns = args.patterns
        config.perform_camera_analysis = not args.no_camera_analysis
        config.generate_reports = not args.no_reports
        config.create_visualizations = not args.no_visualizations
        config.save_intermediate_results = not args.no_intermediate
        config.use_indoor_detector = not args.outdoor_detector
        
        # Validate configuration
        issues = config_manager.validate_configuration(config)
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            sys.exit(1)
        
        # Save configuration for reference
        config_output_path = config.output_dir / "workflow_config.json"
        config_manager.save_configuration(config, config_output_path)
        
        if args.dry_run:
            logger.info("Dry run completed successfully - configuration is valid")
            return
        
        # Execute workflow
        logger.info("Starting tennis court calibration workflow")
        workflow = TennisCalibrationWorkflow(config)
        result = workflow.execute_complete_workflow()
        
        # Print summary
        print("\n" + "="*60)
        print("TENNIS COURT CALIBRATION WORKFLOW SUMMARY")
        print("="*60)
        
        print(f"Workflow ID: {result.workflow_id}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Total Duration: {result.total_duration:.2f} seconds")
        
        if result.best_calibration:
            print(f"\nBest Calibration Result:")
            print(f"  Reprojection Error: {result.best_calibration.reprojection_error:.3f} pixels")
            print(f"  Quality: {result.best_calibration.calibration_quality}")
            print(f"  Method: {result.best_calibration.calibration_method}")
            print(f"  Keypoints Used: {result.best_calibration.keypoint_count}")
        else:
            print("\nNo successful calibration achieved")
        
        print(f"\nStage Results:")
        for stage, stage_result in result.stage_results.items():
            status_symbol = "✓" if stage_result.success else "✗"
            print(f"  {status_symbol} {stage.value}: {stage_result.duration:.2f}s")
        
        print(f"\nResults saved to: {config.output_dir}")
        
        # Final recommendations
        if result.final_report.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in result.final_report['recommendations']:
                print(f"  - {rec}")
        
        # Exit with appropriate code
        sys.exit(0 if result.overall_status.value == 'completed' else 1)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
