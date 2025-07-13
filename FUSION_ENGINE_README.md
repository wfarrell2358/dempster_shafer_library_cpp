# Multi-Sensor Multi-Target Fusion Engine

## Overview

This repository contains both a C++ Dempster-Shafer evidence theory library and a Python multi-sensor multi-target fusion engine implementation.

### C++ Dempster-Shafer Library

The original C++ implementation provides:
- Dempster-Shafer evidence theory for sensor fusion
- Learning classificator for feature classification
- CSV reader for data processing
- Emotion classification based on facial features

### Python Fusion Engine

The new Python implementation provides a comprehensive multi-sensor multi-target fusion engine with:

- **LSTM Neural Network for Prediction**: Predicts target states based on historical observations
- **Transformer-based Association Network**: Associates measurements with existing tracks using attention mechanisms
- **Set-based Neural Networks for Track Management**: Manages track initialization, maintenance, and termination

## Features

### Core Components

1. **LSTMPredictor**: Neural network for predicting target states
   - Handles sequences of observations
   - Provides uncertainty estimates
   - Supports online learning

2. **TransformerAssociation**: Transformer-based data association
   - Multi-head attention mechanism
   - Handles variable numbers of tracks and measurements
   - Solves the assignment problem

3. **SetBasedTrackManager**: Neural network-based track management
   - Automatic track initialization
   - Intelligent track termination
   - Handles multiple track states

4. **FusionEngine**: Main coordination engine
   - Integrates all components
   - Supports multiple sensors
   - Real-time processing capabilities

### Key Capabilities

- Multi-sensor fusion from different sensor types (radar, lidar, camera)
- Real-time multi-target tracking
- Automatic track lifecycle management
- Neural network-based prediction and association
- Scenario simulation for testing
- Training capabilities for all components

## Installation

### C++ Components

```bash
make all
```

### Python Components

```bash
pip install -r requirements.txt
```

## Usage

### C++ Usage

```bash
./lab_exercise test_data.csv
```

### Python Usage

#### Basic Usage

```python
from fusion_engine import FusionEngine
from fusion_engine.fusion_engine import Sensor
import numpy as np

# Create fusion engine
engine = FusionEngine(feature_dim=4, max_tracks=20, max_sensors=5)

# Add sensors
sensor = Sensor(0, 'radar', np.array([0.0, 0.0]))
engine.add_sensor(sensor)

# Process measurements
measurements = [
    {'sensor_id': 0, 'measurement': [1.0, 2.0], 'timestamp': 0.0}
]
result = engine.process_measurements(measurements, 0.0)
```

#### Running the Demo

```bash
python demo_fusion_engine.py
```

#### Running Tests

```bash
python test_fusion_engine.py
```

## Architecture

### Python Fusion Engine Architecture

```
FusionEngine
├── LSTMPredictor (Target State Prediction)
├── TransformerAssociation (Data Association)
├── SetBasedTrackManager (Track Management)
└── Sensor Management
```

### Processing Pipeline

1. **Sensor Data Input**: Multiple sensors provide measurements
2. **Track Prediction**: LSTM networks predict target states
3. **Data Association**: Transformer networks associate measurements to tracks
4. **Track Management**: Set-based networks manage track lifecycle
5. **State Fusion**: Final state estimates are produced

## Examples

### Scenario Simulation

```python
# Run a simulation with 5 targets and 3 sensors
results = engine.simulate_scenario(
    num_targets=5,
    num_sensors=3,
    duration=10.0,
    dt=0.1
)

# Analyze results
stats = engine.get_statistics()
print(f"Total measurements: {stats['total_measurements']}")
print(f"Active tracks: {stats['current_tracks']}")
```

### Training Components

```python
# Train the fusion engine components
training_data = {
    'lstm_sequences': sequences,
    'lstm_targets': targets,
    'association_data': association_data,
    'track_management_data': track_data
}

engine.train_components(training_data, epochs=100)
```

## Testing

The implementation includes comprehensive tests:

- Unit tests for individual components
- Integration tests for the complete fusion engine
- Scenario simulation tests
- Performance benchmarks

Run all tests:
```bash
python test_fusion_engine.py
```

## Performance

The fusion engine is designed for real-time operation:
- Typical processing time: <10ms per frame
- Scales to handle 50+ tracks and 10+ sensors
- Memory efficient implementation
- Configurable parameters for different scenarios

## Dependencies

- Python 3.7+
- NumPy 1.21+
- (Optional) Matplotlib for visualization

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.