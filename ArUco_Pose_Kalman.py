import cv2
import cv2.aruco as aruco
import numpy as np

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
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(6, dtype=np.float32)
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


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Estimate pose of first detected marker
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            position = tvec[0][0]

            # Kalman Filter update
            kalman.predict()
            kalman.correct(np.array([[position[0]], [position[1]], [position[2]]], dtype=np.float32))

            # Draw markers and axes
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.03)
        else:
            # Only predict if no detection
            kalman.predict()

        # Get filtered pose
        filtered_pos = kalman.statePost[:3].flatten()

        # Display filtered position
        cv2.putText(
            frame,
            f"Filtered Position: x={filtered_pos[0]:.2f} y={filtered_pos[1]:.2f} z={filtered_pos[2]:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        cv2.imshow("Aruco Tracker + Kalman", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
