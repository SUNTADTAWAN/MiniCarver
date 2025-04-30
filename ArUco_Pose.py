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

# === Main ===
def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.08  # meters
    aruco_dict_type = aruco.DICT_4X4_1000

    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Press 'ESC' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_positions = {}

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                t = tvecs[i][0]
                marker_positions[marker_id] = t

                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # Floating box with position info
                marker_corner = corners[i][0][0]
                marker_x, marker_y = int(marker_corner[0]), int(marker_corner[1])
                text_lines = [
                    f"ID: {marker_id}",
                    f"X: {t[0]:.2f} m",
                    f"Y: {t[1]:.2f} m",
                    f"Z: {t[2]:.2f} m"
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

            # === Draw lines and distances between all pairs ===
            font = cv2.FONT_HERSHEY_SIMPLEX
            all_pairs = list(itertools.combinations(marker_positions.items(), 2))
            for (id1, p1), (id2, p2) in all_pairs:
                distance = np.linalg.norm(p1 - p2)

                # Project to 2D
                rvec = np.zeros((1, 3))
                tvec = np.zeros((1, 3))
                imgpt1, _ = cv2.projectPoints(np.array([p1]), rvec, tvec, camera_matrix, dist_coeffs)
                imgpt2, _ = cv2.projectPoints(np.array([p2]), rvec, tvec, camera_matrix, dist_coeffs)

                pt1 = tuple(imgpt1[0][0].astype(int))
                pt2 = tuple(imgpt2[0][0].astype(int))

                # Draw red line
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

                # Distance label at midpoint
                mx = (pt1[0] + pt2[0]) // 2
                my = (pt1[1] + pt2[1]) // 2
                label = f"{distance:.2f} m"
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 2)
                cv2.rectangle(frame, (mx - 5, my - th - 5), (mx + tw + 5, my + 5), (255, 255, 255), -1)
                cv2.putText(frame, label, (mx, my), font, 0.5, (0, 0, 0), 2)

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
