#!/usr/bin/env python3
"""
Basic tests for the fusion engine components.
"""

import numpy as np
import sys
import os
import logging

# Add the parent directory to path to import fusion_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_engine import FusionEngine, LSTMPredictor, TransformerAssociation, SetBasedTrackManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lstm_predictor():
    """Test LSTM predictor functionality."""
    logger.info("Testing LSTM Predictor...")
    
    predictor = LSTMPredictor(input_dim=4, hidden_dim=32, output_dim=4, sequence_length=3)
    
    # Test prediction
    sequence = np.array([
        [1.0, 2.0, 0.5, 0.3],
        [1.5, 2.3, 0.5, 0.3],
        [2.0, 2.6, 0.5, 0.3]
    ])
    
    prediction = predictor.predict(sequence)
    assert prediction.shape == (4,), f"Expected shape (4,), got {prediction.shape}"
    
    # Test training with dummy data
    sequences = [sequence, sequence + 1.0, sequence + 2.0]
    targets = [np.array([2.5, 2.9, 0.5, 0.3]), np.array([3.5, 3.9, 0.5, 0.3]), np.array([4.5, 4.9, 0.5, 0.3])]
    
    predictor.train(sequences, targets, epochs=10, verbose=False)
    
    logger.info("LSTM Predictor test passed!")


def test_transformer_association():
    """Test transformer association functionality."""
    logger.info("Testing Transformer Association...")
    
    associator = TransformerAssociation(feature_dim=4, model_dim=32, num_heads=2, num_layers=1)
    
    # Test association
    tracks = [
        np.array([1.0, 2.0, 0.5, 0.3]),
        np.array([3.0, 4.0, -0.2, 0.1])
    ]
    
    measurements = [
        np.array([1.1, 2.1, 0.0, 0.0]),
        np.array([2.9, 4.1, 0.0, 0.0]),
        np.array([5.0, 6.0, 0.0, 0.0])
    ]
    
    associations = associator.associate(tracks, measurements)
    assert isinstance(associations, list), "Expected list of associations"
    
    logger.info("Transformer Association test passed!")


def test_track_manager():
    """Test set-based track manager functionality."""
    logger.info("Testing Track Manager...")
    
    manager = SetBasedTrackManager(feature_dim=4, hidden_dim=32, max_tracks=10)
    
    # Test track initialization
    measurement = np.array([1.0, 2.0, 0.5, 0.3])
    track_id = manager.initialize_track(measurement, 0.0)
    assert track_id is not None, "Track should be initialized"
    
    # Test track update
    new_measurement = np.array([1.5, 2.3, 0.5, 0.3])
    success = manager.update_track(track_id, new_measurement, 1.0)
    assert success, "Track update should succeed"
    
    # Test track management
    measurements = [np.array([2.0, 2.6, 0.5, 0.3]), np.array([10.0, 10.0, 0.0, 0.0])]
    associations = [(track_id, 0)]
    
    mgmt_info = manager.manage_tracks(measurements, associations, 2.0)
    assert 'num_tracks' in mgmt_info, "Management info should contain num_tracks"
    
    logger.info("Track Manager test passed!")


def test_fusion_engine():
    """Test the complete fusion engine."""
    logger.info("Testing Fusion Engine...")
    
    engine = FusionEngine(feature_dim=4, max_tracks=20, max_sensors=5)
    
    # Add sensors
    from fusion_engine.fusion_engine import Sensor
    sensor1 = Sensor(sensor_id=0, sensor_type='radar', position=np.array([0.0, 0.0]))
    sensor2 = Sensor(sensor_id=1, sensor_type='lidar', position=np.array([10.0, 0.0]))
    
    assert engine.add_sensor(sensor1), "Should add sensor successfully"
    assert engine.add_sensor(sensor2), "Should add sensor successfully"
    
    # Test measurement processing
    measurements = [
        {'sensor_id': 0, 'measurement': [1.0, 2.0], 'timestamp': 0.0},
        {'sensor_id': 1, 'measurement': [3.0, 4.0], 'timestamp': 0.0}
    ]
    
    result = engine.process_measurements(measurements, 0.0)
    assert 'track_states' in result, "Result should contain track states"
    assert 'num_measurements' in result, "Result should contain num_measurements"
    
    logger.info("Fusion Engine test passed!")


def test_simulation():
    """Test scenario simulation."""
    logger.info("Testing Simulation...")
    
    engine = FusionEngine(feature_dim=4, max_tracks=10, max_sensors=3)
    
    # Run short simulation
    results = engine.simulate_scenario(num_targets=3, num_sensors=2, duration=2.0, dt=0.5)
    
    assert len(results) > 0, "Simulation should produce results"
    assert all('timestamp' in r for r in results), "All results should have timestamp"
    
    stats = engine.get_statistics()
    assert 'total_measurements' in stats, "Stats should contain total_measurements"
    
    logger.info("Simulation test passed!")


def run_all_tests():
    """Run all tests."""
    logger.info("Running all tests...")
    
    try:
        test_lstm_predictor()
        test_transformer_association()
        test_track_manager()
        test_fusion_engine()
        test_simulation()
        
        logger.info("All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)