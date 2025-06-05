# src/tennis/analysis/gait_analyzer.py
"""Professional gait analysis using MediaPipe + biomechanical analysis"""
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class GaitMetrics:
    step_length: float
    step_time: float
    gait_speed: float
    joint_angles: Dict[str, float]
    movement_efficiency: float
    court_positioning: tuple

class TennisGaitAnalyzer:
    def __init__(self):
        # MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Court-specific movement analysis
        self.movement_patterns = {
            'serve': [], 'return': [], 'rally': [], 'approach': []
        }
        
    def analyze_player_movement(self, frame, player_id):
        """Comprehensive movement analysis"""
        
        # Extract pose landmarks
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Extract key biomechanical points
            landmarks = self._extract_key_landmarks(results.pose_landmarks)
            
            # Calculate joint angles (hip, knee, ankle)
            joint_angles = self._calculate_joint_angles(landmarks)
            
            # Analyze gait pattern
            gait_metrics = self._analyze_gait_pattern(landmarks, player_id)
            
            # Court positioning analysis
            court_position = self._analyze_court_positioning(landmarks)
            
            # Movement efficiency scoring
            efficiency = self._calculate_movement_efficiency(landmarks, gait_metrics)
            
            return {
                'pose_landmarks': landmarks,
                'joint_angles': joint_angles,
                'gait_metrics': gait_metrics,
                'court_position': court_position,
                'movement_efficiency': efficiency,
                'biomechanical_analysis': self._biomechanical_analysis(landmarks)
            }
            
        return None
    
    def _calculate_joint_angles(self, landmarks):
        """Calculate sagittal plane joint angles"""
        angles = {}
        
        # Hip angle (as mentioned in research with 4.0° accuracy)
        hip_angle = self._calculate_angle(
            landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee']
        )
        
        # Knee angle (5.6° accuracy from research)
        knee_angle = self._calculate_angle(
            landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle']
        )
        
        # Ankle angle (7.4° accuracy from research)
        ankle_angle = self._calculate_angle(
            landmarks['left_knee'], landmarks['left_ankle'], landmarks['left_foot']
        )
        
        return {
            'hip': hip_angle, 'knee': knee_angle, 'ankle': ankle_angle
        }
