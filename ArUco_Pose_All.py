import cv2
import cv2.aruco as aruco
import numpy as np
import itertools

def load_camera_parameters(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {yml_path}")
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, dist_coeffs

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

def get_marker_center(corner):
    c = corner.reshape((4, 2))
    center = np.mean(c, axis=0).astype(int)
    return tuple(center)

def draw_comparison_box(frame, marker_id, t_raw, t_filtered, center):
    marker_x, marker_y = center
    text_lines = [
        f"ID: {marker_id}",
        f"Raw: X:{t_raw[0]:.2f} Y:{t_raw[1]:.2f} Z:{t_raw[2]:.2f}",
        f"Kal: X:{t_filtered[0]:.2f} Y:{t_filtered[1]:.2f} Z:{t_filtered[2]:.2f}"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
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
    frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    for j, line in enumerate(text_lines):
        tx = box_x + padding
        ty = box_y + padding + j * line_height
        cv2.putText(frame, line, (tx, ty), font, font_scale, (0, 0, 0), thickness)

def save_transformation_matrix(marker_id, T, save_path="marker_transforms.csv"):
    with open(save_path, "a") as f:
        T_flat = T.flatten()
        f.write(f"{marker_id}," + ",".join([f"{v:.6f}" for v in T_flat]) + "\n")

def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.08  # meters
    aruco_dict_type = aruco.DICT_4X4_1000

    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera not detected")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    kalman_filters = {}

    print("Press 'ESC' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_raw = frame.copy()
        frame_kalman = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_positions_raw = {}
        marker_positions_kalman = {}
        marker_centers = {}

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                t = tvecs[i][0]
                r = rvecs[i]
                c = corners[i]

                center = get_marker_center(c)
                marker_centers[marker_id] = center
                marker_positions_raw[marker_id] = t

                if marker_id not in kalman_filters:
                    kalman_filters[marker_id] = create_kalman_filter()
                kf = kalman_filters[marker_id]
                kf.predict()
                kf.correct(np.array([[t[0]], [t[1]], [t[2]]], dtype=np.float32))
                t_filtered = kf.statePost[:3].flatten()
                marker_positions_kalman[marker_id] = t_filtered

                aruco.drawDetectedMarkers(frame_raw, corners, ids)
                aruco.drawDetectedMarkers(frame_kalman, corners, ids)
                cv2.drawFrameAxes(frame_raw, camera_matrix, dist_coeffs, r, t, 0.03)
                cv2.drawFrameAxes(frame_kalman, camera_matrix, dist_coeffs, r, t, 0.03)

                draw_comparison_box(frame_raw, marker_id, t, t_filtered, center)
                draw_comparison_box(frame_kalman, marker_id, t, t_filtered, center)

                # === Compute and print transformation matrix ===
                R_mat, _ = cv2.Rodrigues(r)
                T = np.eye(4)
                T[:3, :3] = R_mat
                T[:3, 3] = t
                print(f"\nMarker ID {marker_id} - Camera to Marker Transform:\n{T}")
                save_transformation_matrix(marker_id, T)

            for id1, id2 in itertools.combinations(marker_centers, 2):
                c1 = marker_centers[id1]
                c2 = marker_centers[id2]
                mx, my = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2

                d_raw = np.linalg.norm(marker_positions_raw[id1] - marker_positions_raw[id2])
                d_kal = np.linalg.norm(marker_positions_kalman[id1] - marker_positions_kalman[id2])

                cv2.line(frame_raw, c1, c2, (0, 0, 255), 2)
                cv2.putText(frame_raw, f"{d_raw:.2f}m", (mx, my - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.line(frame_kalman, c1, c2, (0, 0, 255), 2)
                cv2.putText(frame_kalman, f"{d_kal:.2f}m", (mx, my + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        both_views = np.hstack((frame_raw, frame_kalman))
        cv2.imshow("Raw (Left) vs Kalman Filtered (Right)", both_views)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
