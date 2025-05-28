import cv2
import numpy as np
import time

# ======= Kalman Filter Creation =======
def create_kf_constant_velocity(dt, accel_std, meas_std):
    F = np.array([[1, 0, 0, dt, 0,  0],
                  [0, 1, 0, 0,  dt, 0],
                  [0, 0, 1, 0,  0,  dt],
                  [0, 0, 0, 1,  0,  0],
                  [0, 0, 0, 0,  1,  0],
                  [0, 0, 0, 0,  0,  1]], dtype=np.float32)

    H = np.eye(3, 6, dtype=np.float32)
    Q = np.eye(6, dtype=np.float32) * accel_std**2
    R = np.eye(3, dtype=np.float32) * meas_std**2
    x = np.zeros((6,), dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    P[3:, 3:] *= 1000.0
    return F, H, Q, R, x, P

def create_kf_constant_acceleration(dt, jerk_std, meas_std):
    F = np.eye(9, dtype=np.float32)
    for i in range(3):
        F[i, i+3] = dt
        F[i, i+6] = 0.5 * dt**2
        F[i+3, i+6] = dt
    H = np.eye(3, 9, dtype=np.float32)
    Q = np.eye(9, dtype=np.float32) * jerk_std**2
    R = np.eye(3, dtype=np.float32) * meas_std**2
    x = np.zeros((9,), dtype=np.float32)
    P = np.eye(9, dtype=np.float32)
    P[3:, 3:] *= 1000.0
    return F, H, Q, R, x, P

# ======= ArUco & Kalman Setup =======
dt = 1/30.0
accel_std = 0.1     # Q of Constant_Velo
meas_std = 1.0      # R
jerk_std = 0.1      # Q of Constant_Accel

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

camera_matrix = np.array([[1063.7383, 0, 959.8903],
                          [0, 1077.995, 581.7375],
                          [0, 0, 1]])
dist_coeffs = np.array([[0.2217, -0.0257, 0.0268, -0.0141, -0.3499]])
marker_length = 0.08

F_v, H_v, Q_v, R_v, x_v, P_v = create_kf_constant_velocity(dt, accel_std, meas_std)
F_a, H_a, Q_a, R_a, x_a, P_a = create_kf_constant_acceleration(dt, jerk_std, meas_std)

# ======= Main Loop =======
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        tvec = tvecs[0][0]  # only 1st marker for comparison

        # ==== Constant Velocity KF ====
        x_pred_v = F_v @ x_v
        P_pred_v = F_v @ P_v @ F_v.T + Q_v
        y_v = tvec - H_v @ x_pred_v
        S_v = H_v @ P_pred_v @ H_v.T + R_v
        K_v = P_pred_v @ H_v.T @ np.linalg.inv(S_v)
        x_v = x_pred_v + K_v @ y_v
        P_v = (np.eye(6) - K_v @ H_v) @ P_pred_v

        # ==== Constant Acceleration KF ====
        x_pred_a = F_a @ x_a
        P_pred_a = F_a @ P_a @ F_a.T + Q_a
        y_a = tvec - H_a @ x_pred_a
        S_a = H_a @ P_pred_a @ H_a.T + R_a
        K_a = P_pred_a @ H_a.T @ np.linalg.inv(S_a)
        x_a = x_pred_a + K_a @ y_a
        P_a = (np.eye(9) - K_a @ H_a) @ P_pred_a

        # ==== Draw results ====
        raw = frame.copy()
        kalman_v = frame.copy()
        kalman_a = frame.copy()

        cv2.drawFrameAxes(raw, camera_matrix, dist_coeffs, rvecs[0], tvec, 0.1)
        cv2.putText(raw, f"Raw: {np.round(tvec, 2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.drawFrameAxes(kalman_v, camera_matrix, dist_coeffs, rvecs[0], x_v[:3], 0.1)
        cv2.putText(kalman_v, f"Vel KF: {np.round(x_v[:3], 2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.drawFrameAxes(kalman_a, camera_matrix, dist_coeffs, rvecs[0], x_a[:3], 0.1)
        cv2.putText(kalman_a, f"Acc KF: {np.round(x_a[:3], 2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        combined = np.hstack((kalman_a, raw, kalman_v))
        cv2.imshow("Kalman Comparison: Acceleration | Raw | Velocity", combined)
    else:
        cv2.imshow("Kalman Comparison: Acceleration | Raw | Velocity", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
