"""Court detection using YOLO + MediaPipe - no custom algorithms"""
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CourtKeypoints:
    points: np.ndarray
    confidence: np.ndarray
    method: str

class CourtDetector:
    """Professional court detection using proven ML models"""
    
    def __init__(self):
        # Use pre-trained YOLO for sports field detection
        self.yolo = YOLO('yolov8n.pt')
        
        # MediaPipe for robust feature detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_objectron = mp.solutions.objectron
        
    def detect_court_keypoints(self, frame: np.ndarray) -> CourtKeypoints:
        """Use YOLO + MediaPipe instead of custom line detection"""
        
        # Method 1: YOLO for initial court region detection
        results = self.yolo(frame)
        
        # Extract court region using YOLO results
        court_region = self._extract_court_region_yolo(frame, results)
        
        if court_region is not None:
            # Use MediaPipe within court region for precise keypoints
            keypoints = self._detect_keypoints_mediapipe(court_region)
            if len(keypoints) >= 4:
                return CourtKeypoints(keypoints, np.full(len(keypoints), 0.9), 'yolo_mediapipe')
        
        # Fallback: OpenCV contours (battle-tested)
        return self._fallback_opencv_contours(frame)
    
    def _extract_court_region_yolo(self, frame: np.ndarray, results) -> np.ndarray:
        """Extract court region using YOLO detection"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Use the largest detected region as court
                largest_box = max(boxes.xyxy, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                x1, y1, x2, y2 = map(int, largest_box)
                return frame[y1:y2, x1:x2]
        return None
    
    def _detect_keypoints_mediapipe(self, court_region: np.ndarray) -> np.ndarray:
        """Use MediaPipe for precise keypoint detection"""
        with self.mp_objectron.Objectron(
            static_image_mode=True,
            max_num_objects=1,
            min_detection_confidence=0.5) as objectron:
            
            results = objectron.process(cv2.cvtColor(court_region, cv2.COLOR_BGR2RGB))
            
            if results.detected_objects:
                # Extract keypoints from MediaPipe results
                return self._extract_mediapipe_keypoints(results.detected_objects[0])
        
        return np.array([])
    
    def _fallback_opencv_contours(self, frame: np.ndarray) -> CourtKeypoints:
        """OpenCV contour detection as reliable fallback"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Use OpenCV's built-in rectangular approximation
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                corners = approx.reshape(-1, 2)
                return CourtKeypoints(corners, np.full(4, 0.7), 'opencv_contour')
        
        # Emergency fallback
        h, w = frame.shape[:2]
        estimated = np.array([[w*0.1, h*0.2], [w*0.9, h*0.2], 
                             [w*0.9, h*0.8], [w*0.1, h*0.8]])
        return CourtKeypoints(estimated, np.full(4, 0.3), 'estimated')
