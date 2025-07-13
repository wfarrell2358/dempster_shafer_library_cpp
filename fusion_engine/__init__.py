"""
Multi-sensor multi-target fusion engine using neural networks.

This package implements a fusion engine that combines:
- LSTM neural networks for prediction
- Transformer-based associations for data association
- Set-based neural networks for track management
"""

from .fusion_engine import FusionEngine
from .lstm_predictor import LSTMPredictor
from .transformer_association import TransformerAssociation
from .set_based_track_manager import SetBasedTrackManager

__version__ = "0.1.0"
__all__ = ["FusionEngine", "LSTMPredictor", "TransformerAssociation", "SetBasedTrackManager"]