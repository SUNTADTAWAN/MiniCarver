import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import queue

# === Load camera intrinsics ===
def load_camera_parameters(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {yml_path}")
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, dist_coeffs

# === Convert rvec + tvec to 4x4 transformation matrix ===
def get_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# === Draw robot as square with heading arrow ===
def draw_robot(ax, x, z, yaw, size=0.2, color='blue', label=''):
    half = size / 2
    corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    rotated = (R @ corners.T).T + np.array([x, z])
    square = plt.Polygon(rotated, closed=True, fill=True, color=color, alpha=0.5, label=label)
    ax.add_patch(square)
    front = R @ np.array([half, 0]) + np.array([x, z])
    ax.arrow(x, z, front[0] - x, front[1] - z, head_width=0.03, color=color)

# === Draw axis lines ===
def draw_axis(ax, origin, rotation, scale=0.1):
    x_axis = origin + scale * rotation[[0, 2], 0]
    y_axis = origin + scale * rotation[[0, 2], 1]
    z_axis = origin + scale * rotation[[0, 2], 2]
    ax.arrow(*origin, *(x_axis - origin), color='red', head_width=0.01)
    ax.arrow(*origin, *(y_axis - origin), color='green', head_width=0.01)
    ax.arrow(*origin, *(z_axis - origin), color='blue', head_width=0.01)

# === Matplotlib update thread ===
def plot_robot_live(pose_queue):
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        try:
            x, z, yaw = pose_queue.get(timeout=1)
        except queue.Empty:
            continue

        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Top-Down View: Robot1 Pose from Robot2")

        draw_robot(ax, 0, 0, 0, size=0.2, color='blue', label='Robot2')
        draw_robot(ax, x, z, yaw, size=0.2, color='red', label='Robot1')
        draw_axis(ax, np.array([x, z]), np.eye(3))
        draw_axis(ax, np.array([0, 0]), np.eye(3))
        ax.legend()
        plt.pause(0.001)

# === Main camera loop ===
def main():
    yml_file = "camera_intrinsics.yml"
    marker_length = 0.08
    aruco_dict_type = aruco.DICT_4X4_1000

    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)
    cap = cv2.VideoCapture(0)  # change to 1, 2 if needed

    if not cap.isOpened():
        print("Camera not detected")
        return

    # Shared queue between threads
    pose_queue = queue.Queue()
    Thread(target=plot_robot_live, args=(pose_queue,), daemon=True).start()

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
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec, tvec = rvecs[i], tvecs[i]

                T_marker_camera = get_transform_matrix(rvec, tvec)
                T_camera_marker = np.linalg.inv(T_marker_camera)

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                T_marker_robot2 = np.eye(4)
                if marker_id == 451:
                    T_marker_robot2[0, 3] = -0.225
                    R, _ = cv2.Rodrigues(np.array([0, 0, 0], dtype=np.float64))
                elif marker_id == 455:
                    T_marker_robot2[0, 3] = +0.225
                    R, _ = cv2.Rodrigues(np.array([0, 0, np.pi], dtype=np.float64))
                elif marker_id == 457:
                    T_marker_robot2[1, 3] = -0.125
                    R, _ = cv2.Rodrigues(np.array([0, 0, -np.pi/2], dtype=np.float64))
                elif marker_id == 453:
                    T_marker_robot2[1, 3] = +0.125
                    R, _ = cv2.Rodrigues(np.array([0, 0, np.pi/2], dtype=np.float64))
                else:
                    continue
                T_marker_robot2[:3, :3] = R

                T_camera_robot1 = np.eye(4)
                T_camera_robot1[0, 3] = 0.0
                T_camera_robot1[2, 3] = -0.225

                T_camera_robot2 = T_camera_marker @ T_marker_robot2
                T_robot1_robot2 = T_camera_robot2 @ np.linalg.inv(T_camera_robot1)
                T_robot2_robot1 = np.linalg.inv(T_robot1_robot2)

                x = T_robot2_robot1[0, 3]
                z = T_robot2_robot1[2, 3]
                yaw = np.arctan2(-T_robot2_robot1[0, 2], T_robot2_robot1[2, 2])

                pose_queue.queue.clear()
                pose_queue.put((x, z, yaw))

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
