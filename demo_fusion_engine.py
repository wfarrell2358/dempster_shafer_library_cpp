#!/usr/bin/env python3
"""
Demonstration of the multi-sensor multi-target fusion engine.

This script shows how to use the fusion engine with LSTM prediction,
transformer-based association, and set-based track management.
"""

import numpy as np
import sys
import os
import logging
import time

# Add the parent directory to path to import fusion_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_engine import FusionEngine
from fusion_engine.fusion_engine import Sensor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_fusion_engine():
    """Demonstrate the fusion engine capabilities."""
    logger.info("=== Multi-Sensor Multi-Target Fusion Engine Demo ===")
    
    # Create fusion engine
    engine = FusionEngine(feature_dim=4, max_tracks=20, max_sensors=5)
    
    # Add sensors with different characteristics
    sensors = [
        Sensor(0, 'radar', np.array([0.0, 0.0]), np.eye(2) * 0.05),
        Sensor(1, 'lidar', np.array([50.0, 0.0]), np.eye(2) * 0.02),
        Sensor(2, 'camera', np.array([0.0, 50.0]), np.eye(2) * 0.1)
    ]
    
    for sensor in sensors:
        engine.add_sensor(sensor)
        logger.info(f"Added {sensor.sensor_type} sensor at position {sensor.position}")
    
    # Demonstrate real-time processing
    logger.info("\n=== Real-time Processing Demo ===")
    
    # Simulate some targets
    targets = [
        {'id': 0, 'state': np.array([10.0, 10.0, 2.0, 1.0])},
        {'id': 1, 'state': np.array([20.0, 30.0, -1.0, 0.5])},
        {'id': 2, 'state': np.array([40.0, 5.0, 0.0, 2.0])}
    ]
    
    for step in range(20):
        timestamp = step * 0.1
        
        # Update target states
        for target in targets:
            state = target['state']
            dt = 0.1
            
            # Update position
            state[0] += state[2] * dt
            state[1] += state[3] * dt
            
            # Add some noise
            state[2] += np.random.normal(0, 0.1)
            state[3] += np.random.normal(0, 0.1)
        
        # Generate measurements
        measurements = []
        for sensor in sensors:
            for target in targets:
                # Detection probability
                if np.random.random() < 0.85:
                    measurement = sensor.generate_measurement(target['state'], timestamp)
                    measurements.append({
                        'sensor_id': sensor.sensor_id,
                        'measurement': measurement.tolist(),
                        'timestamp': timestamp
                    })
        
        # Process measurements
        result = engine.process_measurements(measurements, timestamp)
        
        # Display results
        if step % 5 == 0:  # Display every 5 steps
            logger.info(f"Step {step}: {result['num_measurements']} measurements, "
                       f"{result['num_tracks']} tracks, "
                       f"{len(result['associations'])} associations")
            
            if result['track_states']:
                logger.info("Current tracks:")
                for track_id, state in result['track_states'].items():
                    logger.info(f"  Track {track_id}: pos=({state[0]:.2f}, {state[1]:.2f}), "
                               f"vel=({state[2]:.2f}, {state[3]:.2f})")
    
    # Show final statistics
    stats = engine.get_statistics()
    logger.info("\n=== Final Statistics ===")
    logger.info(f"Total measurements processed: {stats['total_measurements']}")
    logger.info(f"Total tracks created: {stats['new_tracks']}")
    logger.info(f"Total tracks terminated: {stats['terminated_tracks']}")
    logger.info(f"Current active tracks: {stats['current_tracks']}")
    logger.info(f"Successful associations: {stats['successful_associations']}")
    logger.info(f"Average processing time: {stats.get('avg_processing_time', 0):.4f}s")
    
    return engine


def demonstrate_training():
    """Demonstrate training capabilities (simplified)."""
    logger.info("\n=== Training Demo ===")
    
    # Create fusion engine
    engine = FusionEngine(feature_dim=4, max_tracks=10, max_sensors=3)
    
    # Generate simple training data
    logger.info("Generating training data...")
    
    # LSTM training data
    lstm_sequences = []
    lstm_targets = []
    
    for _ in range(20):
        # Generate a simple sequence
        sequence = np.random.uniform(-5, 5, (5, 4))  # sequence_length=5, feature_dim=4
        target = np.random.uniform(-5, 5, 4)
        
        lstm_sequences.append(sequence)
        lstm_targets.append(target)
    
    # Simple training data structure
    training_data = {
        'lstm_sequences': lstm_sequences,
        'lstm_targets': lstm_targets,
        'track_management_data': []
    }
    
    # Train only LSTM component to avoid transformer dimension issues
    logger.info("Training LSTM component...")
    engine.predictor.train(lstm_sequences, lstm_targets, epochs=20, verbose=False)
    
    logger.info("Training completed!")


def demonstrate_scenario_simulation():
    """Demonstrate scenario simulation."""
    logger.info("\n=== Scenario Simulation Demo ===")
    
    engine = FusionEngine(feature_dim=4, max_tracks=15, max_sensors=4)
    
    # Run simulation
    logger.info("Running simulation with 5 targets and 3 sensors for 5 seconds...")
    results = engine.simulate_scenario(
        num_targets=5,
        num_sensors=3,
        duration=5.0,
        dt=0.1
    )
    
    # Analyze results
    logger.info(f"Simulation completed with {len(results)} time steps")
    
    # Extract some statistics
    num_tracks_over_time = [r['num_tracks'] for r in results]
    num_measurements_over_time = [r['num_measurements'] for r in results]
    processing_times = [r['processing_time'] for r in results]
    
    logger.info(f"Average number of tracks: {np.mean(num_tracks_over_time):.2f}")
    logger.info(f"Average number of measurements: {np.mean(num_measurements_over_time):.2f}")
    logger.info(f"Average processing time: {np.mean(processing_times):.4f}s")
    logger.info(f"Max processing time: {np.max(processing_times):.4f}s")
    
    # Show final state
    final_result = results[-1]
    logger.info(f"Final state: {final_result['num_tracks']} tracks")
    
    if final_result['track_states']:
        logger.info("Final track positions:")
        for track_id, state in final_result['track_states'].items():
            logger.info(f"  Track {track_id}: ({state[0]:.2f}, {state[1]:.2f})")


def main():
    """Main demonstration function."""
    logger.info("Starting Multi-Sensor Multi-Target Fusion Engine Demonstration")
    
    try:
        # Demonstrate basic functionality
        engine = demonstrate_fusion_engine()
        
        # Demonstrate training
        demonstrate_training()
        
        # Demonstrate scenario simulation
        demonstrate_scenario_simulation()
        
        logger.info("\n=== Demo Complete ===")
        logger.info("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()