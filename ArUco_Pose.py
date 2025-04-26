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

# === Main ===
def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.05  # meters
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

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                t = tvecs[i][0]

                # === Get marker top-left corner position ===
                marker_corner = corners[i][0][0]
                marker_x, marker_y = int(marker_corner[0]), int(marker_corner[1])

                # === Prepare text lines ===
                text_lines = [
                    f"ID: {ids[i][0]}",
                    f"X: {t[0]:.2f} m",
                    f"Y: {t[1]:.2f} m",
                    f"Z: {t[2]:.2f} m"
                ]

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2
                line_height = 20  # pixels between lines

                # Calculate box size
                text_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in text_lines])
                text_height = line_height * len(text_lines)

                box_padding = 10
                box_width = text_width + box_padding * 2
                box_height = text_height + box_padding

                # Position the box offset from marker
                box_x = marker_x + 10
                box_y = marker_y - box_height - 10
                if box_y < 0:
                    box_y = marker_y + 20  # if too high, move box below marker

                # Draw semi-transparent white rectangle background
                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (box_x, box_y),
                              (box_x + box_width, box_y + box_height),
                              (255, 255, 255),
                              thickness=-1)
                alpha = 0.4
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw each line of text
                for j, line in enumerate(text_lines):
                    text_x = box_x + box_padding
                    text_y = box_y + box_padding + j * line_height
                    cv2.putText(frame, line, (text_x, text_y),
                                font, font_scale, (0, 0, 0), font_thickness)
                    



        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
