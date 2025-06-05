import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import math

from .quality_analyzer import FrameQualityAnalyzer, FrameQuality
from .court_detector import TennisCourtDetector, CourtFeatures
from .frame_scorer import FrameCalibrationScorer, FrameScore

class TennisAnalysisJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for tennis analysis data with comprehensive numpy support"""
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-compatible types"""
        
        # Handle numpy scalar types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            float_val = float(obj)
            if math.isinf(float_val):
                return 999999.0 if float_val > 0 else -999999.0
            elif math.isnan(float_val):
                return None
            return float_val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle numpy generic types
        elif isinstance(obj, np.generic):
            item_value = obj.item()
            if isinstance(item_value, float):
                if math.isinf(item_value):
                    return 999999.0 if item_value > 0 else -999999.0
                elif math.isnan(item_value):
                    return None
            return item_value
        
        # Handle Python infinity and NaN values
        elif isinstance(obj, float):
            if math.isinf(obj):
                return 999999.0 if obj > 0 else -999999.0
            elif math.isnan(obj):
                return None
            return obj
        
        # Handle dataclass objects
        elif hasattr(obj, '__dict__'):
            return {key: self.default(value) for key, value in obj.__dict__.items()}
        
        # Let the base class handle other types
        return super().default(obj)

class CalibrationFrameSelector:
    """Select optimal frames for tennis court camera calibration"""
    
    def __init__(self, min_calibration_frames: int = 8, max_calibration_frames: int = 20):
        self.quality_analyzer = FrameQualityAnalyzer()
        self.court_detector = TennisCourtDetector()
        self.frame_scorer = FrameCalibrationScorer()
        
        self.min_calibration_frames = min_calibration_frames
        self.max_calibration_frames = max_calibration_frames
        
        self.logger = logging.getLogger(__name__)
    
    def select_calibration_frames(self, frames_dir: Path, 
                                max_analyze: Optional[int] = None) -> Dict[str, Any]:
        """Select optimal frames for camera calibration from processed frames"""
        
        self.logger.info(f"Starting frame selection from: {frames_dir}")
        
        # Load available frames
        frame_files = self._load_frame_files(frames_dir, max_analyze)
        
        if len(frame_files) < self.min_calibration_frames:
            raise ValueError(f"Insufficient frames found. Need at least {self.min_calibration_frames}, found {len(frame_files)}")
        
        # Analyze all frames
        frame_analyses = []
        for i, frame_file in enumerate(frame_files):
            self.logger.info(f"Analyzing frame {i+1}/{len(frame_files)}: {frame_file.name}")
            
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.warning(f"Could not load frame: {frame_file}")
                continue
            
            # Analyze frame quality and features
            frame_analysis = self._analyze_single_frame(frame_file.stem, frame)
            frame_analyses.append(frame_analysis)
        
        if len(frame_analyses) < self.min_calibration_frames:
            raise ValueError(f"Too few valid frames after analysis. Need {self.min_calibration_frames}, got {len(frame_analyses)}")
        
        # Score all frames
        frame_scores = [analysis['score'] for analysis in frame_analyses]
        features_list = [analysis['features'] for analysis in frame_analyses]
        
        # Calculate uniqueness scores
        updated_scores = self.frame_scorer.calculate_uniqueness_scores(frame_scores, features_list)
        
        # Update analyses with new scores
        for i, updated_score in enumerate(updated_scores):
            frame_analyses[i]['score'] = updated_score
        
        # Select best frames for calibration
        selected_frames = self._select_best_frames(frame_analyses)
        
        # Generate selection report
        selection_report = self._generate_selection_report(frame_analyses, selected_frames)
        
        self.logger.info(f"Frame selection complete. Selected {len(selected_frames)} frames for calibration.")
        
        return {
            'selected_frames': selected_frames,
            'all_analyses': frame_analyses,
            'selection_report': selection_report
        }
    
    def _load_frame_files(self, frames_dir: Path, max_analyze: Optional[int]) -> List[Path]:
        """Load frame files from directory"""
        
        # Look for both original and processed frames
        frame_patterns = ['frame_*.jpg', 'processed_frame_*.jpg']
        
        all_frame_files = []
        for pattern in frame_patterns:
            frame_files = list(frames_dir.glob(pattern))
            all_frame_files.extend(frame_files)
        
        # Remove duplicates and sort
        unique_files = list(set(all_frame_files))
        unique_files.sort()
        
        # Limit number of frames to analyze if specified
        if max_analyze and len(unique_files) > max_analyze:
            # Sample evenly across available frames
            indices = np.linspace(0, len(unique_files)-1, max_analyze, dtype=int)
            unique_files = [unique_files[i] for i in indices]
        
        self.logger.info(f"Found {len(unique_files)} frames to analyze")
        return unique_files
    
    def _analyze_single_frame(self, frame_id: str, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame for calibration suitability"""
        
        # Analyze frame quality
        frame_quality = self.quality_analyzer.analyze_frame_quality(frame, frame_id)
        
        # Detect court features
        court_features = self.court_detector.detect_court_features(frame)
        
        # Score frame for calibration
        frame_score = self.frame_scorer.score_frame_for_calibration(
            frame_id, frame_quality, court_features, frame.shape[:2]
        )
        
        return {
            'frame_id': frame_id,
            'quality': frame_quality,
            'features': court_features,
            'score': frame_score,
            'frame_shape': frame.shape
        }
    
    def _select_best_frames(self, frame_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the best frames for calibration based on scores and diversity"""
        
        # Filter frames that meet minimum requirements
        recommended_frames = [
            analysis for analysis in frame_analyses
            if analysis['score'].recommended_for_calibration
        ]
        
        if len(recommended_frames) < self.min_calibration_frames:
            # If not enough recommended frames, take the best available
            self.logger.warning(f"Only {len(recommended_frames)} recommended frames found. "
                              f"Taking top {self.min_calibration_frames} frames by score.")
            
            sorted_frames = sorted(frame_analyses, key=lambda x: x['score'].total_score, reverse=True)
            selected_frames = sorted_frames[:max(self.min_calibration_frames, len(sorted_frames))]
        else:
            # Select from recommended frames, prioritizing diversity
            selected_frames = self._select_diverse_frames(recommended_frames)
        
        return selected_frames
    
    def _select_diverse_frames(self, candidate_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select diverse frames to maximize calibration effectiveness"""
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_frames, key=lambda x: x['score'].total_score, reverse=True)
        
        # Start with the highest scoring frame
        selected = [sorted_candidates[0]]
        remaining_candidates = sorted_candidates[1:]
        
        # Select additional frames based on diversity and score
        while len(selected) < self.max_calibration_frames and remaining_candidates:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining_candidates:
                # Calculate diversity score (lower similarity to selected frames)
                diversity_score = self._calculate_diversity_score(candidate, selected)
                combined_score = candidate['score'].total_score * 0.7 + diversity_score * 0.3
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break
        
        # Ensure minimum frame count
        while len(selected) < self.min_calibration_frames and remaining_candidates:
            selected.append(remaining_candidates.pop(0))
        
        return selected
    
    def _calculate_diversity_score(self, candidate: Dict[str, Any], selected_frames: List[Dict[str, Any]]) -> float:
        """Calculate how diverse a candidate frame is compared to already selected frames"""
        
        if not selected_frames:
            return 100.0
        
        # Calculate minimum distance to any selected frame
        min_similarity = float('inf')
        
        for selected_frame in selected_frames:
            similarity = self.frame_scorer._calculate_frame_similarity(
                candidate['features'], selected_frame['features']
            )
            min_similarity = min(min_similarity, similarity)
        
        # Diversity score is inverse of minimum similarity
        diversity_score = (1 - min_similarity) * 100
        
        return diversity_score
    
    def _generate_selection_report(self, all_analyses: List[Dict[str, Any]], 
                                 selected_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive selection report"""
        
        # Calculate statistics with safe conversion
        all_scores = [float(analysis['score'].total_score) for analysis in all_analyses]
        selected_scores = [float(frame['score'].total_score) for frame in selected_frames]
        
        # Count frames by quality grade
        quality_distribution = {}
        for analysis in all_analyses:
            grade = analysis['score'].calibration_suitability
            quality_distribution[grade] = quality_distribution.get(grade, 0) + 1
        
        # Calculate feature statistics with safe conversion
        feature_stats = {
            'avg_features_detected': float(np.mean([a['features'].feature_count for a in all_analyses])),
            'avg_intersections': float(np.mean([len(a['features'].intersections) for a in all_analyses])),
            'avg_detection_confidence': float(np.mean([a['features'].detection_confidence for a in all_analyses]))
        }
        
        report = {
            'total_frames_analyzed': len(all_analyses),
            'frames_selected': len(selected_frames),
            'selection_rate': float(len(selected_frames) / len(all_analyses) * 100),
            'score_statistics': {
                'all_frames': {
                    'mean': float(np.mean(all_scores)),
                    'std': float(np.std(all_scores)),
                    'min': float(np.min(all_scores)),
                    'max': float(np.max(all_scores))
                },
                'selected_frames': {
                    'mean': float(np.mean(selected_scores)),
                    'std': float(np.std(selected_scores)),
                    'min': float(np.min(selected_scores)),
                    'max': float(np.max(selected_scores))
                }
            },
            'quality_distribution': quality_distribution,
            'feature_statistics': feature_stats,
            'selected_frame_ids': [frame['frame_id'] for frame in selected_frames],
            'recommendation': self._generate_recommendation(selected_frames)
        }
        
        return report
    
    def _generate_recommendation(self, selected_frames: List[Dict[str, Any]]) -> str:
        """Generate a recommendation based on selected frames"""
        
        if len(selected_frames) < self.min_calibration_frames:
            return "WARNING: Insufficient frames selected. Calibration quality may be poor."
        
        excellent_count = sum(1 for f in selected_frames if f['score'].calibration_suitability == 'excellent')
        good_count = sum(1 for f in selected_frames if f['score'].calibration_suitability == 'good')
        
        if excellent_count >= len(selected_frames) * 0.6:
            return "EXCELLENT: Selected frames are high quality and should provide accurate calibration."
        elif excellent_count + good_count >= len(selected_frames) * 0.8:
            return "GOOD: Selected frames should provide reliable calibration results."
        else:
            return "FAIR: Calibration may be acceptable but consider capturing additional footage for better results."
    
    def save_selection_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save frame selection results to file with enhanced JSON serialization"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use enhanced JSON encoder for robust serialization
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, cls=TennisAnalysisJSONEncoder)
            
            self.logger.info(f"Selection results saved to: {output_path}")
            
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Enhanced encoder failed: {e}. Trying fallback method.")
            
            # Fallback to manual serialization if needed
            try:
                serializable_results = self._make_json_serializable_fallback(results)
                
                with open(output_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                
                self.logger.info(f"Selection results saved to: {output_path} (using fallback method)")
                
            except Exception as fallback_error:
                self.logger.error(f"Both serialization methods failed: {fallback_error}")
                # Save as pickle as last resort
                import pickle
                pickle_path = output_path.with_suffix('.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(results, f)
                self.logger.info(f"Results saved as pickle to: {pickle_path}")
    
    def _make_json_serializable_fallback(self, obj: Any) -> Any:
        """Fallback method for JSON serialization using recursive approach"""
        
        # Handle numpy scalar types using .item()
        if isinstance(obj, np.generic):
            item_value = obj.item()
            if isinstance(item_value, float):
                if math.isinf(item_value):
                    return 999999.0 if item_value > 0 else -999999.0
                elif math.isnan(item_value):
                    return None
            return item_value
        
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            array_list = obj.tolist()
            return self._make_json_serializable_fallback(array_list)
        
        # Handle Python infinity and NaN values
        elif isinstance(obj, float):
            if math.isinf(obj):
                return 999999.0 if obj > 0 else -999999.0
            elif math.isnan(obj):
                return None
            return obj
        
        # Handle numpy integer types explicitly
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # Handle numpy floating types explicitly  
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            float_val = float(obj)
            if math.isinf(float_val):
                return 999999.0 if float_val > 0 else -999999.0
            elif math.isnan(float_val):
                return None
            return float_val
        
        # Handle numpy boolean types
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle dataclass objects
        elif hasattr(obj, '__dict__'):
            return {key: self._make_json_serializable_fallback(value) for key, value in obj.__dict__.items()}
        
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable_fallback(value) for key, value in obj.items()}
        
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable_fallback(item) for item in obj]
        
        # Handle other types that might cause issues
        elif hasattr(obj, 'tolist'):  # Any object with tolist method (additional numpy types)
            return self._make_json_serializable_fallback(obj.tolist())
        
        # Return as-is for basic Python types
        else:
            return obj

    def export_selected_frames_list(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export just the selected frame IDs for easy use in calibration"""
        
        selected_frame_ids = results['selection_report']['selected_frame_ids']
        
        frame_list = {
            'selected_frames': selected_frame_ids,
            'total_selected': len(selected_frame_ids),
            'selection_timestamp': str(Path().cwd()),
            'recommendation': results['selection_report']['recommendation']
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(frame_list, f, indent=2)
        
        self.logger.info(f"Selected frames list exported to: {output_path}")

    def validate_json_serialization(self, data: Any) -> bool:
        """Validate that data can be JSON serialized"""
        
        try:
            json.dumps(data, cls=TennisAnalysisJSONEncoder)
            return True
        except (TypeError, ValueError):
            return False

    def get_serialization_info(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about serialization compatibility"""
        
        info = {
            'total_objects': 0,
            'numpy_objects': 0,
            'serializable': False,
            'problematic_keys': []
        }
        
        def count_objects(obj, path=""):
            info['total_objects'] += 1
            
            if isinstance(obj, (np.ndarray, np.generic, np.integer, np.floating)):
                info['numpy_objects'] += 1
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    count_objects(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    count_objects(item, f"{path}[{i}]")
        
        count_objects(results)
        info['serializable'] = self.validate_json_serialization(results)
        
        return info
