# SignalScope Model Package
"""
SignalScope Media Forensics & Authenticity Engine
Includes core classifier and bonus modules A through G.
"""

from .predict import predict_image, predict_batch

__all__ = ["predict_image", "predict_batch"]
