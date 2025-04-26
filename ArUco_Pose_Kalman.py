import cv2
import cv2.aruco as aruco
import numpy as np
import time

# === Kalman Filter Tuning Parameters ===
PROCESS_NOISE = 1e-3
MEASUREMENT_NOISE = 1e-2
INITIAL_ERROR = 1
DELTA_TIME = 1.0  # for constant velocity model

# === Load camera intrinsics from YML ===
def load_camera_parameters(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {yml_path}")
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, dist_coeffs

# === Create Kalman Filter for 3D position ===
def create_kalman_filter():
    kf = cv2.KalmanFilter(6, 3)

    kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 0, DELTA_TIME, 0, 0],
        [0, 1, 0, 0, DELTA_TIME, 0],
        [0, 0, 1, 0, 0, DELTA_TIME],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(6, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(6, dtype=np.float32) * INITIAL_ERROR

    return kf

# === Main ===
def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.08  # meters
    aruco_dict_type = aruco.DICT_4X4_1000

    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)
    kalman = create_kalman_filter()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()

    last_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - last_time
        fps = 1.0 / dt
        last_time = current_time

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                position = tvec[i][0]

                # Kalman Filter update
                kalman.predict()
                kalman.correct(np.array([[position[0]], [position[1]], [position[2]]], dtype=np.float32))

                # Draw detected markers and axes
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.03)

                # Center of marker
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))

                # Get filtered position
                filtered_pos = kalman.statePost[:3].flatten()

                # Draw local background box (bigger because 4 lines now)
                box_w, box_h = 220, 120
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (center_x + 10, center_y - 10),
                    (center_x + 10 + box_w, center_y - 10 + box_h),
                    (255, 255, 255),
                    -1
                )
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Write marker info (separate lines for x, y, z)
                cv2.putText(
                    frame,
                    f"ID {ids[i][0]}",
                    (center_x + 20, center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"x = {filtered_pos[0]:.2f}",
                    (center_x + 20, center_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"y = {filtered_pos[1]:.2f}",
                    (center_x + 20, center_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"z = {filtered_pos[2]:.2f}",
                    (center_x + 20, center_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        else:
            # Only predict if no detection
            kalman.predict()

        # Show FPS separately
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display result
        cv2.imshow("Aruco Tracker + Kalman + Moving Info Box", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
