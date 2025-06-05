# src/tennis/__init__.py
"""
Tennis Analysis and Broadcast Package

This package provides a comprehensive suite of tools for analyzing tennis matches,
including player gait analysis, ball tracking, court calibration, scoring,
and generating professional broadcast overlays.
"""

# Analysis
from .analysis import GaitMetrics, TennisGaitAnalyzer

# Broadcast
from .broadcast import TennisBroadcastPipeline, ProfessionalBroadcastOverlay

# Calibration
from .calibration import (
    BroadcastCameraCalibrator,
    CalibrationResult,
    TennisCalibrator,
)

# Detection
from .detection import CourtKeypoints, CourtDetector

# Scoring
from .scoring import TennisScore, BroadcastScoreManager

# Tracking
from .tracking import ProfessionalBallTracker

# Utils
from .utils import (
    find_line_intersections,
    create_court_polygon,
    filter_points_in_court,
    calculate_court_metrics,
)

# Video
from .video import VideoOverlay, VideoProcessor

# Video Acquisition
from .video_acquisition import TennisVideoAcquisitionPipeline

__all__ = [
    # Analysis
    "GaitMetrics", "TennisGaitAnalyzer",
    # Broadcast
    "TennisBroadcastPipeline", "ProfessionalBroadcastOverlay",
    # Calibration
    "BroadcastCameraCalibrator", "CalibrationResult", "TennisCalibrator",
    # Detection
    "CourtKeypoints", "CourtDetector",
    # Scoring
    "TennisScore", "BroadcastScoreManager",
    # Tracking
    "ProfessionalBallTracker",
    # Utils
    "find_line_intersections", "create_court_polygon",
    "filter_points_in_court", "calculate_court_metrics",
    # Video
    "VideoOverlay", "VideoProcessor",
    # Video Acquisition
    "TennisVideoAcquisitionPipeline", # Exporting the aliased name
]