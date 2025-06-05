# src/tennis/scoring/__init__.py
"""Scoring module for tennis matches."""

from .broadcast_scorer import TennisScore, BroadcastScoreManager

__all__ = ["TennisScore", "BroadcastScoreManager"]