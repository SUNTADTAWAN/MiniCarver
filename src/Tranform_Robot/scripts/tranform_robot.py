#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
        self.marker_pub = self.create_publisher(Marker, '/robot2_direction', 1)
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        self.bridge = CvBridge()

    def publish_pose_and_tf(self, x, z, yaw):
        now = self.get_clock().now().to_msg()
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "robot2_base"
        pose.pose.position.x = x
        pose.pose.position.y = 0.0
        pose.pose.position.z = z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pose_pub.publish(pose)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "robot2_base"
        tf.child_frame_id = "robot1"
        tf.transform.translation.x = x
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = z
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def publish_robot2_arrow(self):
        marker = Marker()
        marker.header.frame_id = "robot2_base"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot2"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        self.marker_pub.publish(marker)

    def publish_image(self, frame):
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        self.image_pub.publish(msg)


def main():
    rclpy.init()
    node = PosePublisher()
    yml_file = "/home/tadtawan/MiniCarver/MiniCarver/src/Tranform_Robot/camera_intrinsics.yml"
    marker_length = 0.08
    aruco_dict_type = aruco.DICT_4X4_1000
    camera_matrix, dist_coeffs = load_camera_parameters(yml_file)
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        node.get_logger().error("❌ Camera not detected.")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Press ESC to exit.")

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

                print(f"\n>>> Marker ID: {marker_id}")
                print("rvec:", rvec.flatten())
                print("tvec:", tvec.flatten())

                T_marker_camera = get_transform_matrix(rvec, tvec)
                T_marker_robot2 = np.eye(4)

                if marker_id == 451:
                    T_marker_robot2[0, 3] = -0.225
                    rot_vec = [0, 0, np.pi]
                elif marker_id == 455:
                    T_marker_robot2[0, 3] = +0.225
                    rot_vec = [0, 0, 0]
                elif marker_id == 453:
                    T_marker_robot2[1, 3] = +0.125
                    rot_vec = [0, 0, -np.pi / 2]
                elif marker_id == 457:
                    T_marker_robot2[1, 3] = -0.125
                    rot_vec = [0, 0, np.pi / 2]
                else:
                    continue

                R, _ = cv2.Rodrigues(np.array(rot_vec, dtype=np.float64))
                T_marker_robot2[:3, :3] = R

                T_camera_robot1 = np.eye(4)
                T_camera_robot1[2, 3] = -0.225

                T_camera_robot2 = T_marker_robot2 @ np.linalg.inv(T_marker_camera)
                T_robot1_robot2 = T_camera_robot2 @ np.linalg.inv(T_camera_robot1)
                T_robot2_robot1 = np.linalg.inv(T_robot1_robot2)

                x = T_robot2_robot1[0, 3]
                z = T_robot2_robot1[2, 3]
                yaw = math.atan2(-T_robot2_robot1[0, 2], T_robot2_robot1[2, 2])

                print("Rotation matrix used for marker:\n", R)
                print("T_robot2_robot1:\n", T_robot2_robot1)
                print("Estimated yaw (deg):", math.degrees(yaw))

                node.publish_pose_and_tf(x, z, yaw)

        node.publish_robot2_arrow()
        node.publish_image(frame)

        rclpy.spin_once(node, timeout_sec=0.01)
        cv2.imshow("ArUco Debug", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
