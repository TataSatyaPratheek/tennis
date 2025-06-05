import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CourtKeypoint:
    """Data class for tennis court calibration keypoints"""
    id: int
    name: str
    world_coords: Tuple[float, float, float]  # 3D world coordinates (x, y, z)
    description: str
    importance: str  # 'critical', 'important', 'optional'

@dataclass
class TennisCourtPattern:
    """Tennis court calibration pattern with 14 keypoints"""
    keypoints: List[CourtKeypoint]
    court_dimensions: Dict[str, float]
    coordinate_system: str
    reference_frame: str

class TennisCourtCalibrationPattern:
    """Design and manage tennis court calibration patterns based on ITF standards"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ITF standard tennis court dimensions (in meters)
        self.court_dimensions = {
            'length': 23.77,           # Court length
            'width_doubles': 10.97,    # Doubles court width
            'width_singles': 8.23,     # Singles court width
            'service_box_length': 6.40, # Distance from net to service line
            'service_box_width': 4.115, # Half of singles width
            'net_height_posts': 1.07,   # Net height at posts
            'net_height_center': 0.914, # Net height at center
            'baseline_width': 0.1016,   # Baseline width (up to 10.16 cm)
            'line_width': 0.0508       # Standard line width (5.08 cm)
        }
        
        # Create the 14-keypoint calibration pattern
        self.calibration_pattern = self._create_tennis_court_pattern()
    
    def _create_tennis_court_pattern(self) -> TennisCourtPattern:
        """Create the standard 14-keypoint tennis court calibration pattern"""
        
        # Define coordinate system: origin at bottom-left corner of doubles court
        # X-axis: along the length (baseline to baseline)
        # Y-axis: along the width (sideline to sideline)  
        # Z-axis: height above court surface
        
        keypoints = [
            # Corner points (critical for calibration)
            CourtKeypoint(
                id=1, name="bottom_left_corner",
                world_coords=(0.0, 0.0, 0.0),
                description="Bottom-left corner of doubles court",
                importance="critical"
            ),
            CourtKeypoint(
                id=2, name="bottom_right_corner", 
                world_coords=(0.0, self.court_dimensions['width_doubles'], 0.0),
                description="Bottom-right corner of doubles court",
                importance="critical"
            ),
            CourtKeypoint(
                id=3, name="top_left_corner",
                world_coords=(self.court_dimensions['length'], 0.0, 0.0),
                description="Top-left corner of doubles court", 
                importance="critical"
            ),
            CourtKeypoint(
                id=4, name="top_right_corner",
                world_coords=(self.court_dimensions['length'], self.court_dimensions['width_doubles'], 0.0),
                description="Top-right corner of doubles court",
                importance="critical"
            ),
            
            # Singles court corners (important for singles matches)
            CourtKeypoint(
                id=5, name="bottom_left_singles",
                world_coords=(0.0, (self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2, 0.0),
                description="Bottom-left corner of singles court",
                importance="important"
            ),
            CourtKeypoint(
                id=6, name="bottom_right_singles",
                world_coords=(0.0, (self.court_dimensions['width_doubles'] + self.court_dimensions['width_singles'])/2, 0.0),
                description="Bottom-right corner of singles court", 
                importance="important"
            ),
            CourtKeypoint(
                id=7, name="top_left_singles",
                world_coords=(self.court_dimensions['length'], (self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2, 0.0),
                description="Top-left corner of singles court",
                importance="important"
            ),
            CourtKeypoint(
                id=8, name="top_right_singles", 
                world_coords=(self.court_dimensions['length'], (self.court_dimensions['width_doubles'] + self.court_dimensions['width_singles'])/2, 0.0),
                description="Top-right corner of singles court",
                importance="important"
            ),
            
            # Service line intersections (important for court geometry)
            CourtKeypoint(
                id=9, name="service_line_left",
                world_coords=(self.court_dimensions['service_box_length'], (self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2, 0.0),
                description="Left intersection of service line and singles sideline",
                importance="important"
            ),
            CourtKeypoint(
                id=10, name="service_line_right",
                world_coords=(self.court_dimensions['service_box_length'], (self.court_dimensions['width_doubles'] + self.court_dimensions['width_singles'])/2, 0.0),
                description="Right intersection of service line and singles sideline", 
                importance="important"
            ),
            CourtKeypoint(
                id=11, name="service_line_left_far",
                world_coords=(self.court_dimensions['length'] - self.court_dimensions['service_box_length'], (self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2, 0.0),
                description="Far left intersection of service line and singles sideline",
                importance="important"
            ),
            CourtKeypoint(
                id=12, name="service_line_right_far",
                world_coords=(self.court_dimensions['length'] - self.court_dimensions['service_box_length'], (self.court_dimensions['width_doubles'] + self.court_dimensions['width_singles'])/2, 0.0),
                description="Far right intersection of service line and singles sideline",
                importance="important"
            ),
            
            # Net endpoints (critical for 3D calibration)
            CourtKeypoint(
                id=13, name="net_post_left",
                world_coords=(self.court_dimensions['length']/2, (self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2 - 0.914, self.court_dimensions['net_height_posts']),
                description="Left net post top",
                importance="critical"
            ),
            CourtKeypoint(
                id=14, name="net_post_right", 
                world_coords=(self.court_dimensions['length']/2, (self.court_dimensions['width_doubles'] + self.court_dimensions['width_singles'])/2 + 0.914, self.court_dimensions['net_height_posts']),
                description="Right net post top",
                importance="critical"
            )
        ]
        
        return TennisCourtPattern(
            keypoints=keypoints,
            court_dimensions=self.court_dimensions,
            coordinate_system="ITF_standard",
            reference_frame="bottom_left_origin"
        )
    
    def get_world_coordinates_array(self, importance_filter: Optional[str] = None) -> np.ndarray:
        """Get world coordinates as numpy array for OpenCV calibration"""
        
        filtered_keypoints = self.calibration_pattern.keypoints
        
        if importance_filter:
            filtered_keypoints = [kp for kp in filtered_keypoints if kp.importance == importance_filter]
        
        world_coords = np.array([kp.world_coords for kp in filtered_keypoints], dtype=np.float32)
        
        return world_coords
    
    def get_critical_keypoints(self) -> List[CourtKeypoint]:
        """Get the 6 critical keypoints needed for basic calibration"""
        
        critical_keypoints = [kp for kp in self.calibration_pattern.keypoints if kp.importance == "critical"]
        
        self.logger.info(f"Retrieved {len(critical_keypoints)} critical keypoints for calibration")
        return critical_keypoints
    
    def get_keypoint_by_id(self, keypoint_id: int) -> Optional[CourtKeypoint]:
        """Get specific keypoint by ID"""
        
        for keypoint in self.calibration_pattern.keypoints:
            if keypoint.id == keypoint_id:
                return keypoint
        
        return None
    
    def get_keypoint_by_name(self, name: str) -> Optional[CourtKeypoint]:
        """Get specific keypoint by name"""
        
        for keypoint in self.calibration_pattern.keypoints:
            if keypoint.name == name:
                return keypoint
        
        return None
    
    def visualize_pattern(self, output_path: Optional[Path] = None) -> np.ndarray:
        """Create a top-down visualization of the calibration pattern"""
        
        # Create visualization canvas (scale: 1 meter = 50 pixels)
        scale = 50
        canvas_width = int(self.court_dimensions['width_doubles'] * scale) + 100
        canvas_height = int(self.court_dimensions['length'] * scale) + 100
        
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Draw court outline
        court_corners = [
            (50, 50),  # bottom-left
            (50 + int(self.court_dimensions['width_doubles'] * scale), 50),  # bottom-right  
            (50 + int(self.court_dimensions['width_doubles'] * scale), 50 + int(self.court_dimensions['length'] * scale)),  # top-right
            (50, 50 + int(self.court_dimensions['length'] * scale))  # top-left
        ]
        
        cv2.polylines(canvas, [np.array(court_corners)], True, (0, 0, 0), 2)
        
        # Draw singles court
        singles_offset = int((self.court_dimensions['width_doubles'] - self.court_dimensions['width_singles'])/2 * scale)
        singles_corners = [
            (50 + singles_offset, 50),
            (50 + singles_offset + int(self.court_dimensions['width_singles'] * scale), 50),
            (50 + singles_offset + int(self.court_dimensions['width_singles'] * scale), 50 + int(self.court_dimensions['length'] * scale)),
            (50 + singles_offset, 50 + int(self.court_dimensions['length'] * scale))
        ]
        
        cv2.polylines(canvas, [np.array(singles_corners)], True, (128, 128, 128), 1)
        
        # Draw service lines
        service_y1 = 50 + int(self.court_dimensions['service_box_length'] * scale)
        service_y2 = 50 + int((self.court_dimensions['length'] - self.court_dimensions['service_box_length']) * scale)
        
        cv2.line(canvas, (50 + singles_offset, service_y1), 
                (50 + singles_offset + int(self.court_dimensions['width_singles'] * scale), service_y1), (128, 128, 128), 1)
        cv2.line(canvas, (50 + singles_offset, service_y2),
                (50 + singles_offset + int(self.court_dimensions['width_singles'] * scale), service_y2), (128, 128, 128), 1)
        
        # Draw center service line
        center_x = 50 + int(self.court_dimensions['width_doubles']/2 * scale)
        cv2.line(canvas, (center_x, service_y1), (center_x, service_y2), (128, 128, 128), 1)
        
        # Draw net line
        net_y = 50 + int(self.court_dimensions['length']/2 * scale)
        cv2.line(canvas, (50, net_y), (50 + int(self.court_dimensions['width_doubles'] * scale), net_y), (0, 0, 255), 2)
        
        # Draw keypoints
        colors = {
            'critical': (0, 0, 255),    # Red
            'important': (0, 165, 255), # Orange  
            'optional': (0, 255, 0)     # Green
        }
        
        for keypoint in self.calibration_pattern.keypoints:
            x = int(50 + keypoint.world_coords[1] * scale)  # Y-coordinate becomes X in image
            y = int(50 + keypoint.world_coords[0] * scale)  # X-coordinate becomes Y in image
            
            color = colors.get(keypoint.importance, (128, 128, 128))
            
            # Draw point
            cv2.circle(canvas, (x, y), 5, color, -1)
            
            # Add label
            cv2.putText(canvas, str(keypoint.id), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add legend
        legend_y = canvas_height - 80
        cv2.putText(canvas, "Keypoint Importance:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(canvas, (20, legend_y + 20), 5, colors['critical'], -1)
        cv2.putText(canvas, "Critical", (35, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(canvas, (120, legend_y + 20), 5, colors['important'], -1)
        cv2.putText(canvas, "Important", (135, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save visualization if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), canvas)
            self.logger.info(f"Pattern visualization saved to: {output_path}")
        
        return canvas
    
    def export_pattern_specifications(self, output_path: Path) -> None:
        """Export pattern specifications to JSON file"""
        
        import json
        
        pattern_data = {
            'pattern_name': 'ITF_Standard_Tennis_Court_14_Keypoints',
            'coordinate_system': self.calibration_pattern.coordinate_system,
            'reference_frame': self.calibration_pattern.reference_frame,
            'court_dimensions': self.court_dimensions,
            'keypoints': [
                {
                    'id': kp.id,
                    'name': kp.name,
                    'world_coordinates': {
                        'x': kp.world_coords[0],
                        'y': kp.world_coords[1], 
                        'z': kp.world_coords[2]
                    },
                    'description': kp.description,
                    'importance': kp.importance
                }
                for kp in self.calibration_pattern.keypoints
            ],
            'usage_notes': [
                "Minimum 6 keypoints required for basic calibration",
                "Critical keypoints provide the most robust calibration",
                "Net posts provide essential Z-axis (height) information",
                "Court corners define the primary perspective geometry"
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        self.logger.info(f"Pattern specifications exported to: {output_path}")
