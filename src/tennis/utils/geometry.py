"""Geometric operations using Shapely - no custom geometry code"""
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union
import numpy as np
from typing import List, Tuple

def find_line_intersections(lines: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float]]:
    """Use Shapely's proven intersection algorithms"""
    intersections = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            l1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
            l2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
            
            intersection = l1.intersection(l2)
            if isinstance(intersection, Point):
                intersections.append((intersection.x, intersection.y))
    
    return intersections

def create_court_polygon(corners: np.ndarray) -> Polygon:
    """Create court polygon using Shapely"""
    return Polygon([(p[0], p[1]) for p in corners])

def filter_points_in_court(points: np.ndarray, court_polygon: Polygon) -> np.ndarray:
    """Filter points using Shapely's contains method"""
    valid_points = []
    for point in points:
        if court_polygon.contains(Point(point[0], point[1])):
            valid_points.append(point)
    return np.array(valid_points)

def calculate_court_metrics(court_polygon: Polygon) -> dict:
    """Use Shapely for all geometric calculations"""
    return {
        'area': court_polygon.area,
        'perimeter': court_polygon.length,
        'bounds': court_polygon.bounds,
        'centroid': (court_polygon.centroid.x, court_polygon.centroid.y),
        'is_valid': court_polygon.is_valid
    }
