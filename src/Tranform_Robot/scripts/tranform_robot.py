#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from tf2_ros import TransformBroadcaster

def load_camera_parameters(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {yml_path}")
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    return camera_matrix, dist_coeffs

def get_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

class PosePublisher(Node):
    def __init__(self):
        super().__init__('aruco_pose_publisher')
        self.pose_pub = self.create_publisher(PoseStamped, '/robot1_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish_pose_and_tf(self, x, z, yaw):
        # Publish PoseStamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "robot2_base"

        pose.pose.position.x = x
        pose.pose.position.y = 0.0
        pose.pose.position.z = z

        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pose_pub.publish(pose)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = pose.header.stamp
        t.header.frame_id = "robot2_base"
        t.child_frame_id = "robot1"

        t.transform.translation.x = x
        t.transform.translation.y = 0.0
        t.transform.translation.z = z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = PosePublisher()

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

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec, tvec = rvecs[i], tvecs[i]

                T_marker_camera = get_transform_matrix(rvec, tvec)
                T_camera_marker = np.linalg.inv(T_marker_camera)

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
                T_camera_robot1[2, 3] = -0.225

                T_camera_robot2 = T_camera_marker @ T_marker_robot2
                T_robot1_robot2 = T_camera_robot2 @ np.linalg.inv(T_camera_robot1)
                T_robot2_robot1 = np.linalg.inv(T_robot1_robot2)

                x = T_robot2_robot1[0, 3]
                z = T_robot2_robot1[2, 3]
                yaw = math.atan2(-T_robot2_robot1[0, 2], T_robot2_robot1[2, 2])

                node.publish_pose_and_tf(x, z, yaw)

        rclpy.spin_once(node, timeout_sec=0.01)
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
