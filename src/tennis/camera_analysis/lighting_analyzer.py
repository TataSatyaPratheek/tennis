import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass

@dataclass
class LightingConditions:
    """Data class for lighting analysis results"""
    overall_brightness: float
    brightness_uniformity: float
    contrast_ratio: float
    lighting_quality: str  # 'excellent', 'good', 'fair', 'poor'
    shadow_areas: float  # percentage of frame in shadow
    overexposed_areas: float  # percentage of overexposed areas

class LightingConditionAnalyzer:
    """Analyze lighting conditions in tennis court footage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Lighting thresholds based on tennis court standards
        self.brightness_thresholds = {
            'excellent': (120, 180),  # 500-750lx equivalent
            'good': (80, 120),        # 300-500lx equivalent  
            'fair': (50, 80),         # 150-300lx equivalent
            'poor': (0, 50)           # <150lx equivalent
        }
    
    def analyze_lighting_conditions(self, frames: List[np.ndarray]) -> LightingConditions:
        """Analyze lighting conditions across multiple frames"""
        
        if not frames:
            raise ValueError("No frames provided for lighting analysis")
        
        brightness_values = []
        uniformity_values = []
        contrast_values = []
        shadow_percentages = []
        overexposed_percentages = []
        
        for frame in frames:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness metrics
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Calculate uniformity (coefficient of variation)
            brightness_std = np.std(gray)
            uniformity = 1.0 - (brightness_std / brightness) if brightness > 0 else 0
            uniformity_values.append(uniformity)
            
            # Calculate contrast ratio
            min_val, max_val = np.min(gray), np.max(gray)
            contrast_ratio = max_val / min_val if min_val > 0 else float('inf')
            contrast_values.append(contrast_ratio)
            
            # Detect shadow areas (very dark regions)
            shadow_mask = gray < 30  # Dark threshold
            shadow_percentage = np.sum(shadow_mask) / gray.size * 100
            shadow_percentages.append(shadow_percentage)
            
            # Detect overexposed areas
            overexposed_mask = gray > 240  # Bright threshold
            overexposed_percentage = np.sum(overexposed_mask) / gray.size * 100
            overexposed_percentages.append(overexposed_percentage)
        
        # Calculate average metrics
        overall_brightness = float(np.mean(brightness_values))
        brightness_uniformity = float(np.mean(uniformity_values))
        contrast_ratio = float(np.mean(contrast_values))
        shadow_areas = float(np.mean(shadow_percentages))
        overexposed_areas = float(np.mean(overexposed_percentages))
        
        # Determine lighting quality
        lighting_quality = self._determine_lighting_quality(
            overall_brightness, brightness_uniformity, shadow_areas, overexposed_areas
        )
        
        return LightingConditions(
            overall_brightness=overall_brightness,
            brightness_uniformity=brightness_uniformity,
            contrast_ratio=contrast_ratio,
            lighting_quality=lighting_quality,
            shadow_areas=shadow_areas,
            overexposed_areas=overexposed_areas
        )
    
    def _determine_lighting_quality(self, brightness: float, uniformity: float, 
                                  shadows: float, overexposed: float) -> str:
        """Determine overall lighting quality based on multiple factors"""
        
        # Check brightness range
        brightness_score = 0
        for quality, (min_b, max_b) in self.brightness_thresholds.items():
            if min_b <= brightness <= max_b:
                brightness_score = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}[quality]
                break
        
        # Check uniformity (higher is better)
        uniformity_score = 4 if uniformity > 0.8 else 3 if uniformity > 0.6 else 2 if uniformity > 0.4 else 1
        
        # Check shadow percentage (lower is better)
        shadow_score = 4 if shadows < 5 else 3 if shadows < 15 else 2 if shadows < 30 else 1
        
        # Check overexposure (lower is better)  
        overexposure_score = 4 if overexposed < 2 else 3 if overexposed < 5 else 2 if overexposed < 10 else 1
        
        # Calculate weighted average
        total_score = (brightness_score * 0.4 + uniformity_score * 0.3 + 
                      shadow_score * 0.2 + overexposure_score * 0.1)
        
        if total_score >= 3.5:
            return 'excellent'
        elif total_score >= 2.5:
            return 'good'
        elif total_score >= 1.5:
            return 'fair'
        else:
            return 'poor'
    
    def analyze_court_illumination_pattern(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze illumination patterns specific to tennis courts"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Divide frame into court regions
        regions = {
            'service_box_left': gray[h//3:2*h//3, :w//3],
            'service_box_right': gray[h//3:2*h//3, 2*w//3:],
            'baseline_area': gray[:h//4, :],
            'net_area': gray[2*h//5:3*h//5, w//3:2*w//3],
            'court_center': gray[h//3:2*h//3, w//3:2*w//3]
        }
        
        region_analysis = {}
        for region_name, region in regions.items():
            if region.size > 0:
                region_analysis[region_name] = {
                    'brightness': float(np.mean(region)),
                    'uniformity': float(1.0 - np.std(region) / np.mean(region)) if np.mean(region) > 0 else 0,
                    'contrast': float(np.max(region) - np.min(region))
                }
        
        return region_analysis
