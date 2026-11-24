import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, CameraInfo
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge

from kinova_ik import KinovaIK, rpy_matrix


TARGET_ID = 3
GAIN = 0.5
DEADBAND = 5
MAX_STEP = 0.008
DEPTH = 0.60

START_POSITION = np.array([0.066, 0.25, 0.65])

START_JOINTS = np.array([
    1.051122, 0.105117, 1.744147,
    1.570796, -1.639030, -1.051121
])


class AutonomousTracking(Node):

    def __init__(self):
        super().__init__('autonomous_tracking')

        self.ik = KinovaIK()
        self.bridge = CvBridge()

        self.target_R = rpy_matrix([
            0, 0, np.pi / 2
        ])

        self.q = None
        self.q_velocity = np.zeros(6)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.target_id = TARGET_ID
        self.enabled = True
        self.initialized = False
        self.command_time = 0.0
        self.goal_q = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary
        )

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CameraInfo,
            '/tracking_camera/camera/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            '/tracking_camera/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            "Autonomous Gazebo tracking started"
        )

    def joint_callback(self, msg):

        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity))

        if all(n in positions for n in self.ik.names):

            self.q = np.array([
                positions[n] for n in self.ik.names
            ])

            self.q_velocity = np.array([
                velocities.get(n, 0.0)
                for n in self.ik.names
            ])

    def camera_info_callback(self, msg):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def send_command(self, q, seconds):

        msg = JointTrajectory()
        msg.joint_names = self.ik.names

        point = JointTrajectoryPoint()
        point.positions = q.tolist()

        point.time_from_start = Duration(
            sec=int(seconds),
            nanosec=int((seconds % 1) * 1e9)
        )

        msg.points = [point]

        self.publisher.publish(msg)

        self.goal_q = q.copy()
        self.command_time = time.monotonic()

    def hold_position(self):

        if self.q is not None:
            self.send_command(self.q, 0.3)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg, 'bgr8'
        ).copy()

        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        corners, ids, _ = self.detector.detectMarkers(
            gray
        )

        h, w = frame.shape[:2]

        cx = int(self.cx) if self.cx is not None else w // 2
        cy = int(self.cy) if self.cy is not None else h // 2

        cv2.drawMarker(
            frame, (cx, cy), (0, 255, 255),
            cv2.MARKER_CROSS, 20, 2
        )

        error = None
        status = "Waiting for robot"

        if ids is not None:

            for corner, marker_id in zip(
                corners, ids.flatten()
            ):

                points = corner.reshape(4, 2).astype(
                    np.int32
                )

                mx, my = points.mean(axis=0).astype(int)

                tracked = marker_id == self.target_id

                color = (
                    (0, 255, 0) if tracked
                    else (255, 0, 0)
                )

                cv2.polylines(
                    frame, [points], True, color, 2
                )

                cv2.putText(
                    frame, f"ID {marker_id}",
                    (mx - 20, my - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )

                if tracked:

                    error = np.array([
                        mx - cx, my - cy
                    ], dtype=float)

                    cv2.line(
                        frame, (cx, cy), (mx, my),
                        (0, 255, 255), 2
                    )

                    cv2.circle(
                        frame, (mx, my), 4, color, -1
                    )

        velocity = np.zeros(3)
        position = START_POSITION.copy()

        if self.q is not None:

            position, R, J = self.ik.forward(self.q)

            velocity = J[:3] @ self.q_velocity

            ready = (
                self.goal_q is None
                or np.max(
                    np.abs(self.q - self.goal_q)
                ) < 0.015
            )

            ready = ready and (
                time.monotonic() - self.command_time > 0.5
            )

            if not self.enabled:

                status = "PAUSED"

            elif not self.initialized:

                if self.goal_q is None:

                    self.send_command(START_JOINTS, 8.0)

                elif ready:

                    self.initialized = True

                status = "Moving to viewing pose"

            elif error is None:

                status = "TARGET LOST - HOLDING"

                if not ready:
                    self.hold_position()

            elif np.linalg.norm(error) <= DEADBAND:

                status = "TARGET CENTRED"

                if not ready:
                    self.hold_position()

            elif not self.fx or not self.fy:

                status = "Waiting for camera calibration"

            else:

                status = "TRACKING"

                if ready:

                    dx = GAIN * DEPTH * error[0] / self.fx
                    dz = -GAIN * DEPTH * error[1] / self.fy

                    dx = np.clip(dx, -MAX_STEP, MAX_STEP)
                    dz = np.clip(dz, -MAX_STEP, MAX_STEP)

                    target = position.copy()

                    target[0] += dx
                    target[1] = START_POSITION[1]
                    target[2] += dz

                    target[0] = np.clip(
                        target[0], -0.08, 0.21
                    )

                    target[2] = np.clip(
                        target[2], 0.51, 0.79
                    )

                    q_new, success = self.ik.solve(
                        target,
                        self.target_R,
                        self.q
                    )

                    if success and np.max(
                        np.abs(q_new - self.q)
                    ) < 0.15:

                        self.send_command(q_new, 0.6)

                    else:

                        status = "IK LIMIT - HOLDING"

        cv2.rectangle(
            frame, (0, 0), (w, 85),
            (25, 25, 25), -1
        )

        error_text = (
            f"({error[0]:.0f}, {error[1]:.0f}) px"
            if error is not None else "N/A"
        )

        lines = [
            f"Target {self.target_id} | {status}",
            f"Error: {error_text}",
            f"X: {position[0]:.3f}  "
            f"Y: {position[1]:.3f}  "
            f"Z: {position[2]:.3f} m",
            f"Vx: {velocity[0]:.3f}  "
            f"Vy: {velocity[1]:.3f}  "
            f"Vz: {velocity[2]:.3f} m/s"
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                frame, line,
                (10, 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 255, 0), 1
            )

        cv2.putText(
            frame,
            "1/2/3: Target | SPACE: Pause | Q: Quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 255), 1
        )

        cv2.imshow(
            "Kinova Autonomous Tracking", frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('1'), ord('2'), ord('3')):

            self.target_id = int(chr(key))
            self.hold_position()

        elif key == ord(' '):

            self.enabled = not self.enabled

            if not self.enabled:
                self.hold_position()

        elif key == ord('q'):

            self.hold_position()
            raise KeyboardInterrupt


def main():

    rclpy.init()
    node = AutonomousTracking()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.hold_position()

    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

