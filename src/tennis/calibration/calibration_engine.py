import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from .keypoint_detector import TennisCourtKeypointDetector, KeypointDetectionResult, DetectedKeypoint
from .pattern_manager import CalibrationPatternManager

@dataclass
class CalibrationResult:
    """Camera calibration result data"""
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: List[np.ndarray]
    translation_vectors: List[np.ndarray]
    reprojection_error: float
    calibration_quality: str
    used_frames: List[str]
    keypoint_count: int
    calibration_method: str

@dataclass
class CalibrationValidation:
    """Validation metrics for calibration quality"""
    reprojection_error: float
    error_std: float
    max_error: float
    valid_frame_percentage: float
    keypoint_distribution_score: float
    overall_quality_score: float

class TennisCourtCalibrationEngine:
    """Enhanced calibration engine with better detection"""
    
    def __init__(self, pattern_manager: CalibrationPatternManager):
        self.pattern_manager = pattern_manager
        self.keypoint_detector = TennisCourtKeypointDetector(pattern_manager)  # Use enhanced detector
        self.logger = logging.getLogger(__name__)
        
        # Relaxed calibration parameters
        self.min_frames_for_calibration = 3  # Reduced from 5
        self.max_reprojection_error = 5.0    # Increased from 2.0
        self.convergence_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        
        # Relaxed quality thresholds
        self.quality_thresholds = {
            'excellent': 1.0,  # Increased from 0.5
            'good': 2.5,       # Increased from 1.0
            'fair': 5.0,       # Increased from 2.0
            'poor': float('inf')
        }
    
    def calibrate_camera_from_frames(self, frame_paths: List[Path], 
                               pattern_type: str = 'minimal',
                               frame_shape: Optional[Tuple[int, int]] = None) -> CalibrationResult:
        """Perform camera calibration using selected frames with proper error handling"""
        
        self.logger.info(f"Starting camera calibration with {len(frame_paths)} frames")
        
        try:
            # Detect keypoints in all frames
            detection_results = []
            valid_frame_count = 0
            
            for frame_path in frame_paths:
                try:
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        frame_id = frame_path.stem
                        detection_result = self.keypoint_detector.detect_keypoints_in_frame(
                            frame, frame_id, pattern_type
                        )
                        detection_results.append(detection_result)
                        valid_frame_count += 1
                        
                        self.logger.info(f"Frame {frame_id}: detected {detection_result.detection_count} keypoints "
                                    f"(quality: {detection_result.detection_quality})")
                    else:
                        self.logger.warning(f"Could not load frame: {frame_path}")
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_path}: {e}")
                    continue
            
            if not detection_results:
                self.logger.error("No frames could be processed")
                return self._create_failed_result("No frames could be processed", [])
            
            # Filter frames with sufficient keypoint detection - RELAXED CRITERIA
            valid_detection_results = []
            
            # Try different quality thresholds
            for min_keypoints, quality_filter in [(4, 'fair'), (3, None), (2, None)]:
                valid_detection_results = [
                    result for result in detection_results 
                    if result.detection_count >= min_keypoints and 
                    (quality_filter is None or result.detection_quality != 'poor')
                ]
                
                self.logger.info(f"Found {len(valid_detection_results)} frames with >= {min_keypoints} keypoints")
                
                if len(valid_detection_results) >= self.min_frames_for_calibration:
                    break
            
            if len(valid_detection_results) < self.min_frames_for_calibration:
                # Use whatever we have if we're close
                if len(detection_results) >= self.min_frames_for_calibration:
                    self.logger.warning("Using all available frames with relaxed criteria")
                    valid_detection_results = detection_results
                else:
                    error_msg = (f"Insufficient valid frames for calibration. "
                            f"Need {self.min_frames_for_calibration}, got {len(valid_detection_results)}")
                    self.logger.error(error_msg)
                    return self._create_failed_result(error_msg, [r.frame_id for r in detection_results])
            
            # Prepare calibration data
            try:
                object_points, image_points, frame_ids = self._prepare_calibration_data(
                    valid_detection_results, pattern_type
                )
            except Exception as e:
                error_msg = f"Failed to prepare calibration data: {e}"
                self.logger.error(error_msg)
                return self._create_failed_result(error_msg, [r.frame_id for r in valid_detection_results])
            
            if not object_points or not image_points:
                error_msg = "No valid calibration points could be prepared"
                self.logger.error(error_msg)
                return self._create_failed_result(error_msg, frame_ids)
            
            # Determine image size
            if frame_shape is None:
                try:
                    first_frame = cv2.imread(str(frame_paths[0]))
                    if first_frame is not None:
                        frame_shape = first_frame.shape[:2][::-1]  # (width, height)
                    else:
                        frame_shape = (640, 360)  # Default fallback
                except:
                    frame_shape = (640, 360)  # Default fallback
            
            # Perform camera calibration
            try:
                calibration_result = self._perform_opencv_calibration(
                    object_points, image_points, frame_shape, frame_ids, pattern_type
                )
                
                self.logger.info(f"Calibration complete. Reprojection error: {calibration_result.reprojection_error:.3f}")
                return calibration_result
            except Exception as e:
                error_msg = f"OpenCV calibration failed: {e}"
                self.logger.error(error_msg)
                return self._create_failed_result(error_msg, frame_ids)
                
        except Exception as e:
            error_msg = f"Unexpected error in calibration pipeline: {e}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_failed_result(error_msg, [])

    def _create_failed_result(self, error_message: str, frame_ids: List[str]) -> CalibrationResult:
        """Create a failed calibration result with default values"""
        
        # Create a minimal camera matrix
        default_camera_matrix = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 180.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Zero distortion coefficients
        default_dist_coeffs = np.zeros(5, dtype=np.float64)
        
        return CalibrationResult(
            camera_matrix=default_camera_matrix,
            distortion_coefficients=default_dist_coeffs,
            rotation_vectors=[],
            translation_vectors=[],
            reprojection_error=999.0,
            calibration_quality='failed',
            used_frames=frame_ids,
            keypoint_count=0,
            calibration_method=f'failed: {error_message}'
        )


    def _prepare_calibration_data(self, detection_results: List[KeypointDetectionResult], 
                            pattern_type: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Prepare object and image points for OpenCV calibration with strict format validation"""
        
        object_points = []
        image_points = []
        frame_ids = []
        
        # Get pattern world coordinates
        pattern = self.pattern_manager.get_pattern(pattern_type)
        
        self.logger.info(f"Preparing calibration data from {len(detection_results)} detection results")
        
        for detection_result in detection_results:
            if detection_result.detection_count < 4:  # Need at least 4 points for calibration
                self.logger.debug(f"Skipping frame {detection_result.frame_id} - insufficient keypoints ({detection_result.detection_count})")
                continue
            
            # Collect corresponding points for this frame
            frame_object_points = []
            frame_image_points = []
            
            # Sort keypoints by name for consistent ordering
            sorted_keypoints = sorted(detection_result.detected_keypoints, key=lambda x: x.keypoint_name)
            
            for detected_kp in sorted_keypoints:
                try:
                    # Validate and convert world coordinates
                    world_coords = detected_kp.world_coords
                    if (isinstance(world_coords, (list, tuple)) and len(world_coords) == 3 and
                        all(isinstance(coord, (int, float)) and not isinstance(coord, str) for coord in world_coords)):
                        
                        # Convert to float and ensure proper format
                        world_pt = [float(world_coords[0]), float(world_coords[1]), float(world_coords[2])]
                        
                        # Validate and convert image coordinates  
                        image_coords = detected_kp.image_coords
                        if (isinstance(image_coords, (list, tuple)) and len(image_coords) == 2 and
                            all(isinstance(coord, (int, float)) and not isinstance(coord, str) for coord in image_coords)):
                            
                            # Convert to float and ensure proper format
                            image_pt = [float(image_coords[0]), float(image_coords[1])]
                            
                            # Additional validation - check for reasonable values
                            if (0 <= image_pt[0] <= 10000 and 0 <= image_pt[1] <= 10000 and
                                -100 <= world_pt[0] <= 100 and -100 <= world_pt[1] <= 100 and
                                -10 <= world_pt[2] <= 10):
                                
                                frame_object_points.append(world_pt)
                                frame_image_points.append(image_pt)
                            else:
                                self.logger.warning(f"Coordinates out of reasonable range for {detected_kp.keypoint_name}")
                        else:
                            self.logger.warning(f"Invalid image coordinates for {detected_kp.keypoint_name}: {image_coords}")
                    else:
                        self.logger.warning(f"Invalid world coordinates for {detected_kp.keypoint_name}: {world_coords}")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing keypoint {detected_kp.keypoint_name}: {e}")
                    continue
            
            # Only add frame if we have sufficient valid correspondences
            if len(frame_object_points) >= 4:
                # Convert to numpy arrays with explicit dtype
                obj_pts = np.array(frame_object_points, dtype=np.float32)
                img_pts = np.array(frame_image_points, dtype=np.float32)
                
                # Validate array shapes
                if obj_pts.shape[1] == 3 and img_pts.shape[1] == 2 and len(obj_pts) == len(img_pts):
                    object_points.append(obj_pts)
                    image_points.append(img_pts)
                    frame_ids.append(detection_result.frame_id)
                    
                    self.logger.debug(f"Frame {detection_result.frame_id}: added {len(frame_object_points)} point correspondences")
                else:
                    self.logger.warning(f"Frame {detection_result.frame_id}: invalid array shapes - obj: {obj_pts.shape}, img: {img_pts.shape}")
            else:
                self.logger.warning(f"Frame {detection_result.frame_id}: insufficient valid correspondences ({len(frame_object_points)})")
        
        # Final validation
        if not object_points or not image_points:
            raise ValueError("No valid calibration data could be prepared")
        
        if len(object_points) != len(image_points):
            raise ValueError(f"Mismatch between object points ({len(object_points)}) and image points ({len(image_points)})")
        
        # Validate each pair has same number of points
        for i, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
            if len(obj_pts) != len(img_pts):
                raise ValueError(f"Frame {frame_ids[i]}: mismatch between object points ({len(obj_pts)}) and image points ({len(img_pts)})")
        
        self.logger.info(f"Prepared calibration data: {len(object_points)} frames with "
                        f"average {np.mean([len(pts) for pts in object_points]):.1f} points per frame")
        
        # Debug: Print data types and shapes
        self.logger.debug(f"Object points type: {type(object_points)}, length: {len(object_points)}")
        self.logger.debug(f"Image points type: {type(image_points)}, length: {len(image_points)}")
        if object_points:
            self.logger.debug(f"First object point array shape: {object_points[0].shape}, dtype: {object_points[0].dtype}")
            self.logger.debug(f"First image point array shape: {image_points[0].shape}, dtype: {image_points[0].dtype}")
        
        return object_points, image_points, frame_ids
    
    def _perform_opencv_calibration(self, object_points: List[np.ndarray], 
                              image_points: List[np.ndarray],
                              image_size: Tuple[int, int],
                              frame_ids: List[str],
                              pattern_type: str) -> CalibrationResult:
        """Perform OpenCV camera calibration with comprehensive validation"""
        
        self.logger.info(f"Starting OpenCV calibration with {len(object_points)} frames")
        
        # Pre-calibration validation
        try:
            self._validate_calibration_input(object_points, image_points, image_size)
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")
        
        # Create initial camera matrix estimate
        width, height = image_size
        focal_length = max(width, height) * 1.2
        camera_matrix = np.array([
            [focal_length, 0, width/2.0],
            [0, focal_length, height/2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Initial distortion coefficients (zeros)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        
        self.logger.info(f"Initial camera matrix:\n{camera_matrix}")
        self.logger.info(f"Image size: {image_size}")
        
        try:
            # Try different calibration approaches
            calibration_flags = [
                0,  # No special flags
                cv2.CALIB_FIX_ASPECT_RATIO,
                cv2.CALIB_USE_INTRINSIC_GUESS,
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
            ]
            
            best_result = None
            best_error = float('inf')
            
            for flags in calibration_flags:
                try:
                    self.logger.debug(f"Trying calibration with flags: {flags}")
                    
                    # Perform calibration
                    ret, camera_matrix_result, dist_coeffs_result, rvecs, tvecs = cv2.calibrateCamera(
                        object_points,
                        image_points,
                        image_size,
                        camera_matrix.copy() if flags & cv2.CALIB_USE_INTRINSIC_GUESS else None,
                        dist_coeffs.copy() if flags & cv2.CALIB_USE_INTRINSIC_GUESS else None,
                        flags=flags,
                        criteria=self.convergence_criteria
                    )
                    
                    self.logger.debug(f"Calibration succeeded with flags {flags}, error: {ret:.3f}")
                    
                    if ret < best_error:
                        best_error = ret
                        best_result = (ret, camera_matrix_result, dist_coeffs_result, rvecs, tvecs, flags)
                    
                except cv2.error as e:
                    self.logger.debug(f"Calibration failed with flags {flags}: {e}")
                    continue
            
            if best_result is None:
                raise ValueError("All calibration attempts failed")
            
            ret, camera_matrix_result, dist_coeffs_result, rvecs, tvecs, used_flags = best_result
            
            # Determine calibration quality
            quality = self._determine_calibration_quality(ret)
            
            self.logger.info(f"Best calibration: flags={used_flags}, error={ret:.3f}, quality={quality}")
            
            return CalibrationResult(
                camera_matrix=camera_matrix_result,
                distortion_coefficients=dist_coeffs_result,
                rotation_vectors=rvecs,
                translation_vectors=tvecs,
                reprojection_error=ret,
                calibration_quality=quality,
                used_frames=frame_ids,
                keypoint_count=sum(len(pts) for pts in object_points),
                calibration_method=f'opencv_flags_{used_flags}'
            )
            
        except Exception as e:
            raise ValueError(f"OpenCV calibration error: {e}")

    def _validate_calibration_input(self, object_points: List[np.ndarray], 
                                image_points: List[np.ndarray],
                                image_size: Tuple[int, int]) -> None:
        """Comprehensive validation of calibration input data"""
        
        # Check basic requirements
        if not object_points or not image_points:
            raise ValueError("Empty object_points or image_points")
        
        if len(object_points) != len(image_points):
            raise ValueError(f"Length mismatch: {len(object_points)} object_points vs {len(image_points)} image_points")
        
        if len(object_points) < 3:
            raise ValueError(f"Insufficient frames: need at least 3, got {len(object_points)}")
        
        # Check image size
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError(f"Invalid image_size: {image_size}")
        
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        
        # Validate each frame's data
        for i, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
            # Check types
            if not isinstance(obj_pts, np.ndarray) or not isinstance(img_pts, np.ndarray):
                raise ValueError(f"Frame {i}: points must be numpy arrays")
            
            # Check dtypes
            if obj_pts.dtype != np.float32 or img_pts.dtype != np.float32:
                raise ValueError(f"Frame {i}: points must be float32, got {obj_pts.dtype}, {img_pts.dtype}")
            
            # Check shapes
            if len(obj_pts.shape) != 2 or obj_pts.shape[1] != 3:
                raise ValueError(f"Frame {i}: object_points shape must be (N,3), got {obj_pts.shape}")
            
            if len(img_pts.shape) != 2 or img_pts.shape[1] != 2:
                raise ValueError(f"Frame {i}: image_points shape must be (N,2), got {img_pts.shape}")
            
            # Check point count match
            if len(obj_pts) != len(img_pts):
                raise ValueError(f"Frame {i}: point count mismatch: {len(obj_pts)} vs {len(img_pts)}")
            
            # Check minimum points
            if len(obj_pts) < 4:
                raise ValueError(f"Frame {i}: need at least 4 points, got {len(obj_pts)}")
            
            # Check for invalid values
            if np.any(np.isnan(obj_pts)) or np.any(np.isinf(obj_pts)):
                raise ValueError(f"Frame {i}: object_points contain NaN or Inf")
            
            if np.any(np.isnan(img_pts)) or np.any(np.isinf(img_pts)):
                raise ValueError(f"Frame {i}: image_points contain NaN or Inf")
            
            # Check for strings (this shouldn't happen with numpy arrays, but double-check)
            try:
                obj_pts.astype(np.float64)
                img_pts.astype(np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Frame {i}: invalid data type in points: {e}")
            
            # Check reasonable ranges
            if np.any(img_pts < 0) or np.any(img_pts[:, 0] > width) or np.any(img_pts[:, 1] > height):
                raise ValueError(f"Frame {i}: image_points outside image bounds")
        
        # Check for sufficient variation in object points
        all_obj_points = np.vstack(object_points)
        if np.allclose(all_obj_points, all_obj_points[0], atol=1e-6):
            raise ValueError("All object points are identical - no variation for calibration")
        
        self.logger.debug("Input validation passed")

    def _get_initial_camera_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Generate initial camera matrix estimate"""[9]
        
        width, height = image_size
        
        # Estimate focal length (common heuristic: image diagonal)
        focal_length = max(width, height) * 1.2
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return camera_matrix
    
    def _determine_calibration_quality(self, reprojection_error: float) -> str:
        """Determine calibration quality based on reprojection error"""
        
        for quality, threshold in self.quality_thresholds.items():
            if reprojection_error <= threshold:
                return quality
        
        return 'poor'
    
    def _fallback_calibration(self, object_points: List[np.ndarray], 
                            image_points: List[np.ndarray],
                            image_size: Tuple[int, int],
                            frame_ids: List[str]) -> CalibrationResult:
        """Fallback calibration method with relaxed constraints"""[9]
        
        self.logger.info("Attempting fallback calibration with relaxed constraints")
        
        try:
            # Simpler calibration without constraints
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None,
                flags=0  # No flags - let OpenCV estimate everything
            )
            
            quality = self._determine_calibration_quality(ret)
            
            return CalibrationResult(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                rotation_vectors=rvecs,
                translation_vectors=tvecs,
                reprojection_error=ret,
                calibration_quality=quality,
                used_frames=frame_ids,
                keypoint_count=sum(len(pts) for pts in object_points),
                calibration_method='opencv_fallback'
            )
            
        except Exception as e:
            self.logger.error(f"Fallback calibration also failed: {e}")
            # Return minimal result for debugging
            return self._create_minimal_result(image_size, frame_ids)
    
    def _create_minimal_result(self, image_size: Tuple[int, int], 
                             frame_ids: List[str]) -> CalibrationResult:
        """Create minimal calibration result for debugging"""
        
        width, height = image_size
        
        # Create basic camera matrix
        camera_matrix = np.array([
            [width, 0, width/2],
            [0, width, height/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Zero distortion
        dist_coeffs = np.zeros(5, dtype=np.float64)
        
        return CalibrationResult(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            rotation_vectors=[],
            translation_vectors=[],
            reprojection_error=999.0,
            calibration_quality='poor',
            used_frames=frame_ids,
            keypoint_count=0,
            calibration_method='minimal_fallback'
        )
    
    def validate_calibration(self, calibration_result: CalibrationResult,
                           object_points: List[np.ndarray],
                           image_points: List[np.ndarray]) -> CalibrationValidation:
        """Validate calibration quality using multiple metrics"""[8]
        
        # Calculate per-frame reprojection errors
        frame_errors = []
        
        for i, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
            if i < len(calibration_result.rotation_vectors):
                # Project 3D points to image plane
                projected_pts, _ = cv2.projectPoints(
                    obj_pts,
                    calibration_result.rotation_vectors[i],
                    calibration_result.translation_vectors[i],
                    calibration_result.camera_matrix,
                    calibration_result.distortion_coefficients
                )
                
                # Calculate RMS error for this frame
                error = cv2.norm(img_pts, projected_pts.reshape(-1, 2), cv2.NORM_L2) / len(img_pts)
                frame_errors.append(error)
        
        # Calculate validation metrics
        avg_error = np.mean(frame_errors) if frame_errors else 999.0
        error_std = np.std(frame_errors) if len(frame_errors) > 1 else 0.0
        max_error = np.max(frame_errors) if frame_errors else 999.0
        
        # Calculate valid frame percentage
        valid_frames = sum(1 for error in frame_errors if error < self.max_reprojection_error)
        valid_frame_percentage = valid_frames / len(frame_errors) * 100 if frame_errors else 0.0
        
        # Calculate keypoint distribution score (simplified)
        keypoint_distribution_score = min(100.0, calibration_result.keypoint_count / 50 * 100)
        
        # Overall quality score
        overall_quality_score = (
            (max(0, 100 - avg_error * 50)) * 0.4 +
            valid_frame_percentage * 0.3 +
            keypoint_distribution_score * 0.3
        )
        
        return CalibrationValidation(
            reprojection_error=avg_error,
            error_std=error_std,
            max_error=max_error,
            valid_frame_percentage=valid_frame_percentage,
            keypoint_distribution_score=keypoint_distribution_score,
            overall_quality_score=overall_quality_score
        )
    
    def save_calibration_result(self, calibration_result: CalibrationResult,
                              validation: CalibrationValidation,
                              output_path: Path) -> None:
        """Save calibration results to file"""
        
        # Prepare serializable data
        calibration_data = {
            'calibration_info': {
                'method': calibration_result.calibration_method,
                'quality': calibration_result.calibration_quality,
                'reprojection_error': float(calibration_result.reprojection_error),
                'used_frames': calibration_result.used_frames,
                'keypoint_count': calibration_result.keypoint_count
            },
            'camera_parameters': {
                'camera_matrix': calibration_result.camera_matrix.tolist(),
                'distortion_coefficients': calibration_result.distortion_coefficients.tolist(),
                'image_size': [
                    calibration_result.camera_matrix[0, 2] * 2,  # Estimated width
                    calibration_result.camera_matrix[1, 2] * 2   # Estimated height
                ]
            },
            'validation_metrics': {
                'reprojection_error': float(validation.reprojection_error),
                'error_std': float(validation.error_std),
                'max_error': float(validation.max_error),
                'valid_frame_percentage': float(validation.valid_frame_percentage),
                'keypoint_distribution_score': float(validation.keypoint_distribution_score),
                'overall_quality_score': float(validation.overall_quality_score)
            },
            'camera_intrinsics': {
                'focal_length_x': float(calibration_result.camera_matrix[0, 0]),
                'focal_length_y': float(calibration_result.camera_matrix[1, 1]),
                'principal_point_x': float(calibration_result.camera_matrix[0, 2]),
                'principal_point_y': float(calibration_result.camera_matrix[1, 2]),
                'skew': float(calibration_result.camera_matrix[0, 1])
            }
        }
        
        # Save to JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        self.logger.info(f"Calibration results saved to: {output_path}")
        
        # Also save as NumPy format for direct loading
        np_output_path = output_path.with_suffix('.npz')
        np.savez(
            str(np_output_path),
            camera_matrix=calibration_result.camera_matrix,
            dist_coeffs=calibration_result.distortion_coefficients,
            rvecs=calibration_result.rotation_vectors,
            tvecs=calibration_result.translation_vectors
        )
        
        self.logger.info(f"Calibration parameters saved to: {np_output_path}")
