"""Geometric operations using Shapely"""
from shapely.geometry import LineString, Point
import numpy as np
from typing import List, Tuple

def find_line_intersections(lines: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float]]:
    """Find line intersections using Shapely"""
    intersections = []
    
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            # Use Shapely's proven intersection algorithms
            l1 = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
            l2 = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
            
            intersection = l1.intersection(l2)
            if isinstance(intersection, Point):
                intersections.append((intersection.x, intersection.y))
    
    return intersections

def filter_court_points(points: np.ndarray, court_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """Filter points within court boundaries using Shapely"""
    from shapely.geometry import Polygon
    
    x_min, y_min, x_max, y_max = court_bounds
    court_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    valid_points = []
    for point in points:
        if court_polygon.contains(Point(point[0], point[1])):
            valid_points.append(point)
    
    return np.array(valid_points)
