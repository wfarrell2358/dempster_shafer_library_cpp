"""
Multi-sensor multi-target fusion engine.

This module implements the main fusion engine that coordinates:
- LSTM neural networks for prediction
- Transformer-based associations for data association
- Set-based neural networks for track management
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import time

from .lstm_predictor import LSTMPredictor
from .transformer_association import TransformerAssociation
from .set_based_track_manager import SetBasedTrackManager

logger = logging.getLogger(__name__)


class Sensor:
    """Represents a sensor in the multi-sensor system."""
    
    def __init__(self, sensor_id: int, sensor_type: str, position: np.ndarray, 
                 noise_covariance: Optional[np.ndarray] = None):
        """
        Initialize a sensor.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor (e.g., 'radar', 'lidar', 'camera')
            position: Sensor position
            noise_covariance: Measurement noise covariance matrix
        """
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.position = position
        self.noise_covariance = noise_covariance if noise_covariance is not None else np.eye(2) * 0.1
        self.last_measurement_time = 0.0
        
    def generate_measurement(self, target_state: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Generate a measurement from a target state (for simulation).
        
        Args:
            target_state: Target state vector
            timestamp: Measurement timestamp
            
        Returns:
            measurement: Noisy measurement vector
        """
        # Extract position from state
        position = target_state[:2]
        
        # Add noise
        noise = np.random.multivariate_normal(np.zeros(2), self.noise_covariance)
        measurement = position + noise
        
        self.last_measurement_time = timestamp
        return measurement


class FusionEngine:
    """
    Multi-sensor multi-target fusion engine.
    
    This class coordinates the LSTM predictor, transformer association,
    and set-based track manager to perform multi-target tracking.
    """
    
    def __init__(self, feature_dim: int = 4, max_tracks: int = 50, max_sensors: int = 10):
        """
        Initialize the fusion engine.
        
        Args:
            feature_dim: Dimension of target state vectors
            max_tracks: Maximum number of tracks to maintain
            max_sensors: Maximum number of sensors
        """
        self.feature_dim = feature_dim
        self.max_tracks = max_tracks
        self.max_sensors = max_sensors
        
        # Initialize component networks
        self.predictor = LSTMPredictor(
            input_dim=feature_dim,
            hidden_dim=64,
            output_dim=feature_dim,
            sequence_length=5
        )
        
        self.associator = TransformerAssociation(
            feature_dim=feature_dim,
            model_dim=64,
            num_heads=4,
            num_layers=2,
            max_tracks=max_tracks,
            max_measurements=20
        )
        
        self.track_manager = SetBasedTrackManager(
            feature_dim=feature_dim,
            hidden_dim=64,
            max_tracks=max_tracks,
            track_lifetime=10
        )
        
        # Sensor management
        self.sensors: Dict[int, Sensor] = {}
        
        # Fusion parameters
        self.fusion_window = 1.0  # Time window for measurement fusion
        self.current_time = 0.0
        
        # Statistics
        self.fusion_stats = {
            'total_measurements': 0,
            'total_tracks': 0,
            'successful_associations': 0,
            'new_tracks': 0,
            'terminated_tracks': 0,
            'processing_times': []
        }
        
    def add_sensor(self, sensor: Sensor) -> bool:
        """
        Add a sensor to the fusion engine.
        
        Args:
            sensor: Sensor to add
            
        Returns:
            success: Whether sensor was successfully added
        """
        if len(self.sensors) >= self.max_sensors:
            logger.warning("Maximum number of sensors reached")
            return False
        
        if sensor.sensor_id in self.sensors:
            logger.warning(f"Sensor {sensor.sensor_id} already exists")
            return False
        
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Added sensor {sensor.sensor_id} of type {sensor.sensor_type}")
        return True
    
    def remove_sensor(self, sensor_id: int) -> bool:
        """
        Remove a sensor from the fusion engine.
        
        Args:
            sensor_id: ID of sensor to remove
            
        Returns:
            success: Whether sensor was successfully removed
        """
        if sensor_id not in self.sensors:
            logger.warning(f"Sensor {sensor_id} not found")
            return False
        
        del self.sensors[sensor_id]
        logger.info(f"Removed sensor {sensor_id}")
        return True
    
    def process_measurements(self, measurements: List[Dict[str, Any]], 
                           timestamp: float) -> Dict[str, Any]:
        """
        Process a batch of measurements from multiple sensors.
        
        Args:
            measurements: List of measurement dictionaries with keys:
                         'sensor_id', 'measurement', 'timestamp'
            timestamp: Current processing timestamp
            
        Returns:
            fusion_result: Dictionary with fusion results
        """
        start_time = time.time()
        self.current_time = timestamp
        
        # Extract measurement vectors
        measurement_vectors = []
        sensor_ids = []
        
        for meas_data in measurements:
            sensor_id = meas_data.get('sensor_id')
            measurement = meas_data.get('measurement')
            meas_time = meas_data.get('timestamp', timestamp)
            
            if sensor_id is not None and measurement is not None:
                # Convert to numpy array and ensure correct dimension
                meas_vector = np.array(measurement)
                if len(meas_vector) < self.feature_dim:
                    # Pad with zeros or velocity estimates
                    if len(meas_vector) == 2:  # Position only
                        # Add zero velocity
                        meas_vector = np.concatenate([meas_vector, np.zeros(2)])
                    else:
                        # Pad with zeros
                        padding = np.zeros(self.feature_dim - len(meas_vector))
                        meas_vector = np.concatenate([meas_vector, padding])
                elif len(meas_vector) > self.feature_dim:
                    # Truncate
                    meas_vector = meas_vector[:self.feature_dim]
                
                measurement_vectors.append(meas_vector)
                sensor_ids.append(sensor_id)
        
        # Update statistics
        self.fusion_stats['total_measurements'] += len(measurement_vectors)
        
        # Step 1: Predict existing tracks
        track_predictions = self.track_manager.predict_tracks(timestamp)
        
        # Step 2: Data association
        track_states = list(track_predictions.values())
        track_ids = list(track_predictions.keys())
        
        associations = []
        if len(track_states) > 0 and len(measurement_vectors) > 0:
            # Use transformer for association
            associations = self.associator.associate(track_states, measurement_vectors)
            
            # Convert to track_id based associations
            track_associations = []
            for track_idx, meas_idx in associations:
                if track_idx < len(track_ids):
                    track_associations.append((track_ids[track_idx], meas_idx))
            associations = track_associations
        
        # Step 3: Track management
        management_info = self.track_manager.manage_tracks(
            measurement_vectors, associations, timestamp
        )
        
        # Step 4: Update predictions using LSTM
        self._update_predictions(timestamp)
        
        # Update statistics
        self.fusion_stats['successful_associations'] += len(associations)
        self.fusion_stats['new_tracks'] += management_info['num_new']
        self.fusion_stats['terminated_tracks'] += management_info['num_terminated']
        self.fusion_stats['total_tracks'] = len(self.track_manager.tracks)
        
        processing_time = time.time() - start_time
        self.fusion_stats['processing_times'].append(processing_time)
        
        # Prepare result
        result = {
            'timestamp': timestamp,
            'num_measurements': len(measurement_vectors),
            'num_tracks': len(self.track_manager.tracks),
            'associations': associations,
            'track_states': self.track_manager.get_track_states(),
            'track_info': self.track_manager.get_track_info(),
            'management_info': management_info,
            'sensor_ids': sensor_ids,
            'processing_time': processing_time
        }
        
        logger.debug(f"Processed {len(measurement_vectors)} measurements, "
                    f"{len(associations)} associations, "
                    f"{management_info['num_new']} new tracks, "
                    f"{management_info['num_terminated']} terminated tracks")
        
        return result
    
    def _update_predictions(self, timestamp: float) -> None:
        """
        Update track predictions using LSTM.
        
        Args:
            timestamp: Current timestamp
        """
        for track_id, track in self.track_manager.tracks.items():
            if len(track.state_history) >= self.predictor.sequence_length:
                # Use recent history for prediction
                recent_history = track.state_history[-self.predictor.sequence_length:]
                sequence = np.array(recent_history)
                
                # Predict next state
                try:
                    predicted_state = self.predictor.predict(sequence)
                    
                    # Update track with prediction if no recent measurements
                    time_since_update = timestamp - track.last_update_time
                    if time_since_update > 0.5:  # 0.5 second threshold
                        track.state = predicted_state
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for track {track_id}: {e}")
    
    def train_components(self, training_data: Dict[str, Any], epochs: int = 100) -> None:
        """
        Train the fusion engine components.
        
        Args:
            training_data: Training data with keys:
                          'lstm_sequences', 'lstm_targets',
                          'association_data', 'track_management_data'
            epochs: Number of training epochs
        """
        logger.info("Training fusion engine components")
        
        # Train LSTM predictor
        if 'lstm_sequences' in training_data and 'lstm_targets' in training_data:
            self.predictor.train(
                training_data['lstm_sequences'],
                training_data['lstm_targets'],
                epochs=epochs,
                verbose=True
            )
        
        # Train transformer association
        if 'association_data' in training_data:
            assoc_data = training_data['association_data']
            self.associator.train(
                assoc_data.get('track_sequences', []),
                assoc_data.get('measurement_sequences', []),
                assoc_data.get('ground_truth_associations', []),
                epochs=epochs
            )
        
        # Train track manager
        if 'track_management_data' in training_data:
            self.track_manager.train(
                training_data['track_management_data'],
                epochs=epochs
            )
    
    def simulate_scenario(self, num_targets: int = 5, num_sensors: int = 3, 
                         duration: float = 10.0, dt: float = 0.1) -> List[Dict[str, Any]]:
        """
        Simulate a multi-target tracking scenario.
        
        Args:
            num_targets: Number of targets to simulate
            num_sensors: Number of sensors to simulate
            duration: Simulation duration in seconds
            dt: Time step
            
        Returns:
            results: List of fusion results for each time step
        """
        logger.info(f"Simulating scenario: {num_targets} targets, "
                   f"{num_sensors} sensors, {duration}s duration")
        
        # Initialize sensors
        self.sensors.clear()
        for i in range(num_sensors):
            sensor = Sensor(
                sensor_id=i,
                sensor_type='radar',
                position=np.random.uniform(-100, 100, 2),
                noise_covariance=np.eye(2) * 0.1
            )
            self.add_sensor(sensor)
        
        # Initialize targets
        targets = []
        for i in range(num_targets):
            # Random initial state [x, y, vx, vy]
            state = np.array([
                np.random.uniform(-50, 50),  # x
                np.random.uniform(-50, 50),  # y
                np.random.uniform(-5, 5),    # vx
                np.random.uniform(-5, 5)     # vy
            ])
            targets.append(state)
        
        # Reset track manager
        self.track_manager.reset()
        
        results = []
        time_steps = np.arange(0, duration, dt)
        
        for t in time_steps:
            # Update target states
            for i, target in enumerate(targets):
                # Simple constant velocity motion
                target[0] += target[2] * dt  # x += vx * dt
                target[1] += target[3] * dt  # y += vy * dt
                
                # Add some noise to velocity
                target[2] += np.random.normal(0, 0.1)
                target[3] += np.random.normal(0, 0.1)
            
            # Generate measurements
            measurements = []
            for sensor_id, sensor in self.sensors.items():
                for i, target in enumerate(targets):
                    # Generate measurement with some probability
                    if np.random.random() < 0.8:  # 80% detection probability
                        measurement = sensor.generate_measurement(target, t)
                        measurements.append({
                            'sensor_id': sensor_id,
                            'measurement': measurement,
                            'timestamp': t
                        })
            
            # Process measurements
            result = self.process_measurements(measurements, t)
            results.append(result)
        
        logger.info(f"Simulation completed: {len(results)} time steps processed")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fusion engine statistics.
        
        Returns:
            stats: Dictionary with statistics
        """
        stats = self.fusion_stats.copy()
        
        if len(stats['processing_times']) > 0:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        
        stats['num_sensors'] = len(self.sensors)
        stats['current_tracks'] = len(self.track_manager.tracks)
        
        return stats
    
    def reset(self) -> None:
        """Reset the fusion engine."""
        self.track_manager.reset()
        self.current_time = 0.0
        self.fusion_stats = {
            'total_measurements': 0,
            'total_tracks': 0,
            'successful_associations': 0,
            'new_tracks': 0,
            'terminated_tracks': 0,
            'processing_times': []
        }
        logger.info("Fusion engine reset")