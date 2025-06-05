# src/tennis/video_acquisition/__init__.py
"""Video acquisition module."""
from .pipeline import SimpleVideoAcquisitionPipeline

# Alias SimpleVideoAcquisitionPipeline to TennisVideoAcquisitionPipeline
# for a consistent public API.
TennisVideoAcquisitionPipeline = SimpleVideoAcquisitionPipeline

__all__ = [
    "TennisVideoAcquisitionPipeline",
]
