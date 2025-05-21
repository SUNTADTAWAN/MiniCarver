import cv2
import numpy as np

# --- ArUco marker and camera setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()


# Camera calibration (replace with your actual calibration data)
camera_matrix = np.array([[600, 0, 320],
                          [0, 600, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_length = 0.1  # meter

# --- Kalman Filter Setup ---
dt = 1/30.0  # assuming 30 FPS

F = np.array([[1, 0, 0, dt, 0,  0],
              [0, 1, 0, 0,  dt, 0],
              [0, 0, 1, 0,  0,  dt],
              [0, 0, 0, 1,  0,  0],
              [0, 0, 0, 0,  1,  0],
              [0, 0, 0, 0,  0,  1]], dtype=np.float32)

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]], dtype=np.float32)

# Process noise Q
accel_std = 1.0
q11 = (dt**3)/3.0
q13 = (dt**2)/2.0
q33 = dt
Q = np.zeros((6, 6), dtype=np.float32)
for i in range(3):
    Q[i, i] = q11
    Q[i, i+3] = q13
    Q[i+3, i] = q13
    Q[i+3, i+3] = q33
Q *= accel_std**2

# Measurement noise R
R = np.eye(3, dtype=np.float32) * 0.25

# Initial state
x_est = np.zeros((6,), dtype=np.float32)
P = np.eye(6, dtype=np.float32)
P[3:, 3:] *= 1000.0
kalman_initialized = False

# --- Start Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        tvec = tvecs[0][0]  # use first marker

        if not kalman_initialized:
            x_est[:3] = tvec
            x_est[3:] = 0
            kalman_initialized = True

        # Prediction
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        # Update
        y = tvec - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred

        # Draw
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvec, 0.1)

        cv2.putText(frame, f"Raw: {np.round(tvec, 2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"KF : {np.round(x_est[:3], 2)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print("Measured:", np.round(tvec, 2), "Filtered:", np.round(x_est[:3], 2))
    else:
        print("No marker detected.")

    cv2.imshow("Kalman ArUco Tracker", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
