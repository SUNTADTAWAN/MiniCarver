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

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                t = tvecs[i][0]
                print(f"[ID {ids[i][0]}] Position: x={t[0]:.3f}m, y={t[1]:.3f}m, z={t[2]:.3f}m")

                # === Draw marker ID on image ===
                corner = corners[i][0]
                top_left = tuple(corner[0].astype(int))
                # Format position text
                position_text = f"ID:{ids[i][0]} x:{t[0]:.2f} y:{t[1]:.2f} z:{t[2]:.2f}"

                # Draw the text next to the top-left corner of the marker
                cv2.putText(frame, position_text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
