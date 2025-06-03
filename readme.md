# ArUco Marker Tracking with Kalman Filter

A robust real-time ArUco marker tracking system that uses Kalman filtering to handle detection noise, temporary occlusions, and marker blinking. The system implements both Constant Velocity (CV) and Constant Acceleration (CA) motion models for smooth and accurate pose estimation.

## Features

- Real-time ArUco marker detection and tracking
- Dual Kalman filter implementation (CV and CA models)
- Handles temporary marker loss and occlusions
- Noise reduction and smooth trajectory estimation
- Live visualization with comparison views
- Data logging and post-processing analysis
- Automatic tracker management

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam or USB camera

### Required Dependencies

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

### Alternative Installation (using conda)

```bash
conda install opencv numpy matplotlib
pip install opencv-contrib-python
```

### Verify Installation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
```

## Instructions for Use

### 1. Camera Calibration (Recommended)

Before using the tracker, calibrate your camera for accurate pose estimation:

```python
# Use OpenCV camera calibration or update these values in the code:
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])
```

### 2. ArUco Marker Preparation

- Print ArUco markers from the DICT_4X4_1000 dictionary
- Measure the actual size of your printed markers
- Update `marker_length` in the code (in meters)

### 3. Parameter Tuning

Adjust these parameters based on your setup:

```python
dt = 1/30.0                    # Frame rate (30 FPS)
process_noise_std = 0.5        # Process noise (higher = more responsive)
meas_noise_std = 2.0           # Measurement noise (higher = more smoothing)
max_frames_without_detection = 10  # Prediction frames without detection
```

## How to Use

### Basic Usage

1. **Run the tracker:**
   ```bash
   python aruco_kalman_tracker.py
   ```

2. **Controls:**
   - ESC: Exit the application
   - The system automatically detects and tracks ArUco markers in view

3. **Display Windows:**
   - **Left Panel**: Acceleration KF results
   - **Middle Panel**: Raw detection results  
   - **Right Panel**: Velocity KF results

### Advanced Usage

#### Custom Marker Dictionary

```python
# Change ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
```

#### Video File Input

```python
# Replace webcam input with video file
cap = cv2.VideoCapture('your_video.mp4')
```

#### Save Tracking Data

```python
# Add after the main loop
import pickle
with open('tracking_data.pkl', 'wb') as f:
    pickle.dump(log_data, f)
```

## Mathematical Equations

### Kalman Filter Framework

The Kalman filter operates in two phases: **Prediction** and **Update**.

#### 1. Prediction Phase

**State Prediction:**
```
x̂(k|k-1) = F × x̂(k-1|k-1) + B × u(k)
```

**Covariance Prediction:**
```
P(k|k-1) = F × P(k-1|k-1) × F^T + Q
```

Where:
- `x̂(k|k-1)` is the predicted state at time k
- `F` is the state transition matrix
- `B` is the control input matrix
- `u(k)` is the control input vector
- `Q` is the process noise covariance

#### 2. Update Phase

**Innovation:**
```
y(k) = z(k) - H × x̂(k|k-1)
```

**Innovation Covariance:**
```
S(k) = H × P(k|k-1) × H^T + R
```

**Kalman Gain:**
```
K(k) = P(k|k-1) × H^T × S(k)^(-1)
```

**State Update:**
```
x̂(k|k) = x̂(k|k-1) + K(k) × y(k)
```

**Covariance Update:**
```
P(k|k) = (I - K(k) × H) × P(k|k-1)
```

### Control Input Model (B × u)

For ArUco marker tracking, the control input can model external forces or known accelerations:

**Control Input Matrix (CV Model):**
```
B = [dt²/2   0     0  ]
    [ dt     0     0  ]
    [ 0    dt²/2   0  ]
    [ 0     dt     0  ]
    [ 0     0    dt²/2]
    [ 0     0     dt  ]
```

**Control Input Vector:**
```
u = [ax, ay, az]^T
```

Where `ax`, `ay`, `az` are known accelerations (e.g., gravity compensation).

**Note:** In this implementation, control input is omitted (`B × u = 0`) since ArUco markers typically don't follow predictable external forces.

### Motion Models

#### Constant Velocity (CV) Model

**State Vector:**
```
x = [px, vx, py, vy, pz, vz]^T
```

**State Transition Matrix:**
```
F = [1  dt  0   0  0   0 ]
    [0   1  0   0  0   0 ]
    [0   0  1  dt  0   0 ]
    [0   0  0   1  0   0 ]
    [0   0  0   0  1  dt ]
    [0   0  0   0  0   1 ]
```

#### Constant Acceleration (CA) Model

**State Vector:**
```
x = [px, vx, ax, py, vy, ay, pz, vz, az]^T
```

**State Transition Matrix:**
```
F = [1  dt  dt²/2   0   0     0     0   0     0   ]
    [0   1    dt    0   0     0     0   0     0   ]
    [0   0     1    0   0     0     0   0     0   ]
    [0   0     0    1  dt  dt²/2    0   0     0   ]
    [0   0     0    0   1    dt     0   0     0   ]
    [0   0     0    0   0     1     0   0     0   ]
    [0   0     0    0   0     0     1  dt  dt²/2 ]
    [0   0     0    0   0     0     0   1    dt  ]
    [0   0     0    0   0     0     0   0     1  ]
```

### Observation Model

**Observation Matrix (Position-only measurements):**

For CV model:
```
H = [1  0  0  0  0  0]
    [0  0  1  0  0  0]
    [0  0  0  0  1  0]
```

For CA model:
```
H = [1  0  0  0  0  0  0  0  0]
    [0  0  0  1  0  0  0  0  0]
    [0  0  0  0  0  0  1  0  0]
```

### Noise Models

**Process Noise Covariance (CV Model):**
```
Q = σ²_process × [dt⁴/4  dt³/2    0      0      0      0   ]
                  [dt³/2   dt²     0      0      0      0   ]
                  [  0      0   dt⁴/4  dt³/2    0      0   ]
                  [  0      0   dt³/2   dt²     0      0   ]
                  [  0      0     0      0   dt⁴/4  dt³/2 ]
                  [  0      0     0      0   dt³/2   dt²  ]
```

**Measurement Noise Covariance:**
```
R = σ²_measurement × I₃
```

### Initial Conditions

**Initial State:**
```
x₀ = [0, 0, 0, 0, 0, 0]^T  (CV model)
x₀ = [0, 0, 0, 0, 0, 0, 0, 0, 0]^T  (CA model)
```

**Initial Covariance:**
```
P₀ = 1000 × I_n
```
where n = 6 for CV model, n = 9 for CA model.

The large initial uncertainty (1000) ensures rapid convergence to measurements during filter initialization.

## Performance Analysis

### Filter Comparison

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Constant Velocity** | Fast computation, stable | Can't handle acceleration | Smooth, predictable motion |
| **Constant Acceleration** | Handles complex motion | More computational cost | Rapid changes, curved paths |

### Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `process_noise_std` | 0.1 - 1.0 | Higher = more responsive to changes |
| `meas_noise_std` | 0.5 - 5.0 | Higher = more smoothing |
| `dt` | 1/30 - 1/60 | Should match actual frame rate |

## Author
Tadtawan Chaloempornmongkol