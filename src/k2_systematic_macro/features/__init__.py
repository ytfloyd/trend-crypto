"""Feature generation for K2 systematic macro research."""
from __future__ import annotations

from .core import FeatureConfig, build_feature_frame
from .targets import TargetConfig, build_target_frame

__all__ = ["FeatureConfig", "TargetConfig", "build_feature_frame", "build_target_frame"]
