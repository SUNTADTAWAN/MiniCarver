import cv2
import cv2.aruco as aruco
import numpy as np
import itertools

# === Load camera intrinsics from YML ===
def load_camera_parameters(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {yml_path}")
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, dist_coeffs

# === Create Kalman filter for a marker ===
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

# === Get center of marker from corners ===
def get_marker_center(corner):
    c = corner.reshape((4, 2))
    center = np.mean(c, axis=0).astype(int)
    return tuple(center)

# === Main ===
def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.08  # meters
    aruco_dict_type = aruco.DICT_4X4_1000

    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)
    cap = cv2.VideoCapture(5)
    if not cap.isOpened():
        print("Camera not detected")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    kalman_filters = {}  # ID -> KalmanFilter

    print("Press 'ESC' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_positions = {}
        marker_centers = {}

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                t_raw = tvecs[i][0]
                corner = corners[i]

                # Create Kalman if new marker
                if marker_id not in kalman_filters:
                    kalman_filters[marker_id] = create_kalman_filter()

                kf = kalman_filters[marker_id]
                kf.predict()
                kf.correct(np.array([[t_raw[0]], [t_raw[1]], [t_raw[2]]], dtype=np.float32))
                t_filtered = kf.statePost[:3].flatten()

                marker_positions[marker_id] = t_filtered
                marker_centers[marker_id] = get_marker_center(corner)

                # Draw axis
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # Draw floating info box
                marker_x, marker_y = marker_centers[marker_id]
                text_lines = [
                    f"ID: {marker_id}",
                    f"X: {t_filtered[0]:.2f} m",
                    f"Y: {t_filtered[1]:.2f} m",
                    f"Z: {t_filtered[2]:.2f} m"
                ]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                line_height = 20

                text_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines])
                text_height = line_height * len(text_lines)

                padding = 10
                box_w = text_width + padding * 2
                box_h = text_height + padding
                box_x = marker_x + 10
                box_y = marker_y - box_h - 10
                if box_y < 0:
                    box_y = marker_y + 20

                overlay = frame.copy()
                cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                for j, line in enumerate(text_lines):
                    tx = box_x + padding
                    ty = box_y + padding + j * line_height
                    cv2.putText(frame, line, (tx, ty), font, font_scale, (0, 0, 0), thickness)

            # === Draw lines and distances between all marker pairs ===
            for (id1, id2) in itertools.combinations(marker_positions, 2):
                t1, t2 = marker_positions[id1], marker_positions[id2]
                c1, c2 = marker_centers[id1], marker_centers[id2]

                distance = np.linalg.norm(t1 - t2)
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                label = f"{distance:.2f} m"

                # Line and label
                cv2.line(frame, c1, c2, (0, 0, 255), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (midpoint[0] - 5, midpoint[1] - th - 5),
                              (midpoint[0] + tw + 5, midpoint[1] + 5), (255, 255, 255), -1)
                cv2.putText(frame, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow("ArUco Detection with Kalman + Distance", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
