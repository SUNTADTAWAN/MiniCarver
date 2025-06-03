import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======= Kalman Filter Creation =======
def create_kf_cv(dt, process_noise_std, meas_noise_std):
    """Create Constant Velocity Kalman Filter"""
    # State: [x, vx, y, vy, z, vz]
    F = np.array([
        [1, dt, 0,  0,  0,  0],
        [0, 1,  0,  0,  0,  0],
        [0, 0,  1, dt,  0,  0],
        [0, 0,  0, 1,   0,  0],
        [0, 0,  0, 0,   1, dt],
        [0, 0,  0, 0,   0, 1]
    ], dtype=np.float32)

    # Observation matrix (we observe position only)
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ], dtype=np.float32)

    # Process noise covariance (higher for acceleration components)
    Q = np.array([
        [dt**4/4, dt**3/2, 0, 0, 0, 0],
        [dt**3/2, dt**2, 0, 0, 0, 0],
        [0, 0, dt**4/4, dt**3/2, 0, 0],
        [0, 0, dt**3/2, dt**2, 0, 0],
        [0, 0, 0, 0, dt**4/4, dt**3/2],
        [0, 0, 0, 0, dt**3/2, dt**2]
    ], dtype=np.float32) * process_noise_std**2

    # Measurement noise covariance
    R = np.eye(3, dtype=np.float32) * meas_noise_std**2
    
    # Initial state and covariance
    x = np.zeros((6,), dtype=np.float32)
    P = np.eye(6, dtype=np.float32) * 1000  # High initial uncertainty
    
    return {'F': F, 'H': H, 'Q': Q, 'R': R, 'x': x, 'P': P, 'initialized': False}

def create_kf_ca(dt, process_noise_std, meas_noise_std):
    """Create Constant Acceleration Kalman Filter"""
    # State: [x, vx, ax, y, vy, ay, z, vz, az]
    F = np.eye(9, dtype=np.float32)
    dt2 = 0.5 * dt * dt
    
    # Position updates
    F[0, 1], F[0, 2] = dt, dt2
    F[3, 4], F[3, 5] = dt, dt2
    F[6, 7], F[6, 8] = dt, dt2
    
    # Velocity updates
    F[1, 2] = dt
    F[4, 5] = dt
    F[7, 8] = dt

    # Observation matrix (we observe position only)
    H = np.zeros((3, 9), dtype=np.float32)
    H[0, 0], H[1, 3], H[2, 6] = 1, 1, 1

    # Process noise covariance
    Q = np.eye(9, dtype=np.float32) * process_noise_std**2
    
    # Measurement noise covariance
    R = np.eye(3, dtype=np.float32) * meas_noise_std**2
    
    # Initial state and covariance
    x = np.zeros((9,), dtype=np.float32)
    P = np.eye(9, dtype=np.float32) * 1000  # High initial uncertainty
    
    return {'F': F, 'H': H, 'Q': Q, 'R': R, 'x': x, 'P': P, 'initialized': False}

def kalman_predict(kf):
    """Prediction step"""
    kf['x'] = kf['F'] @ kf['x']
    kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']

def kalman_update(kf, measurement):
    """Update step"""
    if not kf['initialized']:
        # Initialize state with first measurement
        if len(kf['x']) == 6:  # CV model
            kf['x'][0], kf['x'][2], kf['x'][4] = measurement
        else:  # CA model
            kf['x'][0], kf['x'][3], kf['x'][6] = measurement
        kf['initialized'] = True
        return
    
    # Standard Kalman update
    y = measurement - kf['H'] @ kf['x']  # Innovation
    S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']  # Innovation covariance
    
    # Check if S is invertible
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
    
    K = kf['P'] @ kf['H'].T @ S_inv  # Kalman gain
    kf['x'] = kf['x'] + K @ y
    kf['P'] = (np.eye(len(kf['x'])) - K @ kf['H']) @ kf['P']

def get_position_estimate(kf):
    """Get position estimate from state"""
    if len(kf['x']) == 6:  # CV model
        return kf['x'][[0, 2, 4]]
    else:  # CA model
        return kf['x'][[0, 3, 6]]

# ======= Parameters =======
dt = 1/30.0
process_noise_std = 0.1  # Increased for better tracking    (Q)
meas_noise_std = 1.0     # Adjusted based on typical ArUco noise    (R)
max_frames_without_detection = 100  # Max frames to predict without measurement

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera parameters (you should calibrate these for your camera)
camera_matrix = np.array([[1063.7383, 0, 959.8903],
                          [0, 1077.995, 581.7375],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([0.2217, -0.0257, 0.0268, -0.0141, -0.3499], dtype=np.float32)
marker_length = 0.08

# Storage for filters and data
kf_vel = {}
kf_acc = {}
log_data = {}
frames_since_detection = {}

# ======= Main Loop =======
cap = cv2.VideoCapture(0)
frame_count = 0

print("Starting ArUco tracking with Kalman filtering...")
print("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # Create display frames
    raw_frame = frame.copy()
    vel_frame = frame.copy()
    acc_frame = frame.copy()

    detected_ids = set()
    
    if ids is not None:
        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                detected_ids.add(marker_id)
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]

                # Initialize filters if needed
                if marker_id not in kf_vel:
                    kf_vel[marker_id] = create_kf_cv(dt, process_noise_std, meas_noise_std)
                    kf_acc[marker_id] = create_kf_ca(dt, process_noise_std, meas_noise_std)
                    log_data[marker_id] = {"raw": [], "vel": [], "acc": []}
                    frames_since_detection[marker_id] = 0

                # Reset counter since we detected the marker
                frames_since_detection[marker_id] = 0

                # Kalman filter processing
                # Constant Velocity Filter
                kalman_predict(kf_vel[marker_id])
                kalman_update(kf_vel[marker_id], tvec)
                vel_pos = get_position_estimate(kf_vel[marker_id])

                # Constant Acceleration Filter  
                kalman_predict(kf_acc[marker_id])
                kalman_update(kf_acc[marker_id], tvec)
                acc_pos = get_position_estimate(kf_acc[marker_id])

                # Log data
                log_data[marker_id]["raw"].append(tvec.copy())
                log_data[marker_id]["vel"].append(vel_pos.copy())
                log_data[marker_id]["acc"].append(acc_pos.copy())

                # Draw markers and axes
                cv2.aruco.drawDetectedMarkers(raw_frame, corners)
                cv2.drawFrameAxes(raw_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                cv2.drawFrameAxes(vel_frame, camera_matrix, dist_coeffs, rvec, vel_pos, 0.05)
                cv2.drawFrameAxes(acc_frame, camera_matrix, dist_coeffs, rvec, acc_pos, 0.05)

                # Add text information
                y_offset = 30 + i * 80
                cv2.putText(raw_frame, f"ID {marker_id} Raw: {tvec}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(vel_frame, f"ID {marker_id} Vel KF: {vel_pos}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(acc_frame, f"ID {marker_id} Acc KF: {acc_pos}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        except Exception as e:
            print(f"Error processing ArUco markers: {e}")

    # Handle markers that weren't detected (prediction only)
    for marker_id in list(frames_since_detection.keys()):
        if marker_id not in detected_ids:
            frames_since_detection[marker_id] += 1
            
            # Continue prediction for a limited number of frames
            if frames_since_detection[marker_id] <= max_frames_without_detection:
                if marker_id in kf_vel and kf_vel[marker_id]['initialized']:
                    # Predict only
                    kalman_predict(kf_vel[marker_id])
                    kalman_predict(kf_acc[marker_id])
                    
                    vel_pos = get_position_estimate(kf_vel[marker_id])
                    acc_pos = get_position_estimate(kf_acc[marker_id])
                    
                    # Create dummy rvec for visualization
                    rvec = np.array([0.0, 0.0, 0.0])
                    
                    # Draw predicted positions (with different color to indicate prediction)
                    cv2.drawFrameAxes(vel_frame, camera_matrix, dist_coeffs, rvec, vel_pos, 0.05)
                    cv2.drawFrameAxes(acc_frame, camera_matrix, dist_coeffs, rvec, acc_pos, 0.05)
                    
                    # Add text to show it's predicted
                    cv2.putText(vel_frame, f"ID {marker_id} PREDICTED", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(acc_frame, f"ID {marker_id} PREDICTED", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # Remove old trackers
                if marker_id in kf_vel:
                    del kf_vel[marker_id]
                if marker_id in kf_acc:
                    del kf_acc[marker_id]
                del frames_since_detection[marker_id]

    # Display combined view
    if raw_frame.shape[0] > 0:
        # Resize frames for better display
        height = 1000
        width = int(raw_frame.shape[1] * height / raw_frame.shape[0])
        raw_resized = cv2.resize(raw_frame, (width, height))
        vel_resized = cv2.resize(vel_frame, (width, height))
        acc_resized = cv2.resize(acc_frame, (width, height))
        
        # Stack horizontally
        stacked = np.hstack((acc_resized, raw_resized, vel_resized))
        
        # Add labels
        cv2.putText(stacked, "Acceleration KF", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(stacked, "Raw Detection", (width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(stacked, "Velocity KF", (2*width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("ArUco Tracking: Acc KF | Raw | Vel KF", stacked)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count} frames")
print(f"Tracked {len(log_data)} unique markers")

# ======= Plotting Results =======
if log_data:
    print("Generating plots...")
    for marker_id, data in log_data.items():
        if len(data["raw"]) > 10: 
            raw_np = np.array(data["raw"])
            vel_np = np.array(data["vel"])
            acc_np = np.array(data["acc"])
            
            plt.figure(figsize=(15, 10))
            axes_labels = ["X", "Y", "Z"]
            colors = ['red', 'green', 'blue']
            
            for i, (label, color) in enumerate(zip(axes_labels, colors)):
                plt.subplot(3, 1, i+1)
                plt.plot(raw_np[:, i], '--', color=color, alpha=0.7, label='Raw', linewidth=1)
                plt.plot(vel_np[:, i], '-', color='green', label='Velocity KF', linewidth=2)
                plt.plot(acc_np[:, i], '-', color='blue', label='Acceleration KF', linewidth=2)
                plt.title(f"Marker {marker_id} - {label} Position Tracking")
                plt.ylabel(f"{label} Position (m)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
            plt.xlabel("Frame Number")
            plt.suptitle(f"ArUco Marker {marker_id} Tracking Comparison", fontsize=14)
            plt.tight_layout()
            plt.show()
else:
    print("No tracking data collected for plotting.")