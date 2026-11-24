
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

TARGET_ID = 3


class MarkerDetector(Node):

    def __init__(self):
        super().__init__('marker_detector')

        self.bridge = CvBridge()

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary
        )

        self.subscription = self.create_subscription(
            Image,
            '/tracking_camera/camera/image_raw',
            self.process_image,
            qos_profile_sensor_data
        )

        self.get_logger().info("ArUco detector started")

    def process_image(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.drawMarker(
            frame, (cx, cy), (0, 255, 255),
            cv2.MARKER_CROSS, 20, 2
        )

        status = "Target not detected"
        status_color = (0, 0, 255)

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):

                points = corner.reshape(4, 2).astype(np.int32)
                mx, my = points.mean(axis=0).astype(int)

                tracked = marker_id == TARGET_ID
                color = (0, 255, 0) if tracked else (255, 0, 0)

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
                    ex = mx - cx
                    ey = my - cy

                    cv2.line(
                        frame, (cx, cy), (mx, my),
                        (0, 255, 255), 2
                    )

                    cv2.circle(
                        frame, (mx, my), 4, color, -1
                    )

                    status = f"Target {marker_id} | Error: ({ex}, {ey}) px"
                    status_color = (0, 255, 0)

        cv2.rectangle(
            frame, (0, 0), (w, 40), (30, 30, 30), -1
        )

        cv2.putText(
            frame, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, status_color, 2
        )

        cv2.imshow("Kinova ArUco Tracking", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = MarkerDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()



