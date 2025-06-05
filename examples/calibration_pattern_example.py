#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tennis.calibration.pattern_manager import CalibrationPatternManager

def main():
    """Example usage of tennis court calibration patterns"""
    
    print("Tennis Court Calibration Pattern Demo")
    print("="*50)
    
    # Initialize pattern manager
    pattern_manager = CalibrationPatternManager()
    
    # Display available patterns
    print("\nAvailable calibration patterns:")
    for pattern_name in pattern_manager.available_patterns.keys():
        pattern = pattern_manager.get_pattern(pattern_name)
        print(f"  - {pattern_name}: {len(pattern.keypoints)} keypoints")
    
    # Demo each pattern type
    for pattern_type in ['minimal', 'standard', 'broadcast', 'singles', 'robust']:
        print(f"\n--- {pattern_type.upper()} PATTERN ---")
        
        # Get pattern data
        world_points, metadata = pattern_manager.prepare_calibration_data(pattern_type)
        
        print(f"Keypoints: {metadata['num_keypoints']}")
        print(f"Coordinate system: {metadata['coordinate_system']}")
        
        # Show critical keypoints
        pattern = pattern_manager.get_pattern(pattern_type)
        critical_points = [kp for kp in pattern.keypoints if kp.importance == 'critical']
        print(f"Critical keypoints: {len(critical_points)}")
        
        for kp in critical_points:
            print(f"  {kp.name}: {kp.world_coords}")
    
    # Generate pattern visualizations and specifications
    output_dir = Path("data/calibration_patterns")
    print(f"\nGenerating pattern visualizations in: {output_dir}")
    pattern_manager.visualize_all_patterns(output_dir)
    
    # Demo point correspondence mapping
    print("\n--- POINT CORRESPONDENCE MAPPING ---")
    mapping = pattern_manager.create_point_correspondence_mapping('standard')
    
    print("Standard pattern keypoint mapping:")
    for name, info in list(mapping.items())[:5]:  # Show first 5
        print(f"  {name}: index {info['index']}, coords {info['world_coords']}")
    
    # Demo validation
    print("\n--- VALIDATION DEMO ---")
    
    # Simulate some image points (normally these would come from court detection)
    pattern = pattern_manager.get_pattern('minimal')
    num_points = len(pattern.keypoints)
    
    # Valid image points
    valid_image_points = np.random.rand(num_points, 2) * 1000  # Random points in [0, 1000]
    validation_result = pattern_manager.validate_image_points(valid_image_points, 'minimal')
    print(f"Valid points validation: {validation_result['valid']}")
    
    # Invalid image points (wrong count)
    invalid_image_points = np.random.rand(3, 2) * 1000
    validation_result = pattern_manager.validate_image_points(invalid_image_points, 'minimal')
    print(f"Invalid points validation: {validation_result['valid']}")
    print(f"Errors: {validation_result['errors']}")
    
    # Demo critical points extraction
    print("\n--- CRITICAL POINTS EXTRACTION ---")
    world_points, metadata = pattern_manager.prepare_calibration_data('robust')
    
    # Simulate image points for robust pattern
    robust_pattern = pattern_manager.get_pattern('robust')
    simulated_image_points = np.random.rand(len(robust_pattern.keypoints), 2) * 1000
    
    critical_world, critical_image = pattern_manager.get_critical_points_subset(
        simulated_image_points, 'robust'
    )
    
    print(f"Total points in robust pattern: {len(robust_pattern.keypoints)}")
    print(f"Critical points extracted: {len(critical_world)}")
    print(f"Critical world points shape: {critical_world.shape}")
    print(f"Critical image points shape: {critical_image.shape}")
    
    print("\nPattern generation complete!")
    print(f"Check output directory: {output_dir}")

if __name__ == "__main__":
    main()
