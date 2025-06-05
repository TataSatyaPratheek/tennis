# src/tennis/scoring/broadcast_scorer.py
"""Professional tennis scoring system with broadcast integration"""
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class TennisScore:
    player1_sets: int = 0
    player2_sets: int = 0
    player1_games: int = 0
    player2_games: int = 0
    player1_points: int = 0
    player2_points: int = 0
    serving_player: int = 1
    is_tiebreak: bool = False
    is_break_point: bool = False

class BroadcastScoreManager:
    def __init__(self):
        self.score = TennisScore()
        self.match_history = []
        self.point_analysis = []
        
    def analyze_point_from_tracking(self, ball_trajectory, player_positions):
        """Automatically determine point outcome from tracking data"""
        
        # Analyze ball landing position
        landing_position = self._get_ball_landing_position(ball_trajectory)
        
        # Check if ball is in/out using court calibration
        is_in = self._is_ball_in_court(landing_position)
        
        # Determine winner based on tracking data
        point_winner = self._determine_point_winner(
            ball_trajectory, player_positions, is_in
        )
        
        # Update score
        if point_winner:
            self.update_score(point_winner)
            
        return {
            'winner': point_winner,
            'ball_landing': landing_position,
            'is_in': is_in,
            'point_type': self._classify_point_type(ball_trajectory)
        }
    
    def get_broadcast_data(self):
        """Format data for broadcast overlay"""
        return {
            'score': self.score,
            'match_stats': self._calculate_match_stats(),
            'serve_info': self._get_serve_information(),
            'break_point': self.score.is_break_point,
            'set_point': self._is_set_point(),
            'match_point': self._is_match_point()
        }
