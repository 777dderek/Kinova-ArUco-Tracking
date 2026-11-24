import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis

    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    return (
        np.eye(3)
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * (K @ K)
    )


def rpy_matrix(rpy):
    r, p, y = rpy

    return (
        rotation([0, 0, 1], y)
        @ rotation([0, 1, 0], p)
        @ rotation([1, 0, 0], r)
    )


class KinovaIK:

    def __init__(self):

        path = Path(__file__).parent / "robot_description.txt"
        text = path.read_text()

        root = ET.fromstring(text[text.index("<robot"):])

        self.names = [
            f"joint_{i}" for i in range(1, 7)
        ]

        self.chain = []
        self.lower = []
        self.upper = []

        joints = {
            j.attrib["name"]: j
            for j in root.findall("joint")
        }

        for name in self.names + ["tracking_camera_joint"]:

            joint = joints[name]
            origin = joint.find("origin")

            xyz = np.fromstring(
                origin.attrib.get("xyz", "0 0 0"),
                sep=" "
            )

            rpy = np.fromstring(
                origin.attrib.get("rpy", "0 0 0"),
                sep=" "
            )

            T = np.eye(4)
            T[:3, :3] = rpy_matrix(rpy)
            T[:3, 3] = xyz

            axis = joint.find("axis")

            if name in self.names:

                a = np.fromstring(
                    axis.attrib["xyz"], sep=" "
                )

                limit = joint.find("limit")

                self.lower.append(
                    float(limit.attrib["lower"])
                )

                self.upper.append(
                    float(limit.attrib["upper"])
                )

            else:
                a = None

            self.chain.append((T, a))

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

    def forward(self, q):

        T = np.eye(4)
        origins = []
        axes = []

        for i, (origin, axis) in enumerate(self.chain):

            T = T @ origin

            if axis is not None:

                origins.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ axis)

                A = np.eye(4)
                A[:3, :3] = rotation(axis, q[i])

                T = T @ A

        position = T[:3, 3].copy()
        orientation = T[:3, :3].copy()

        J = np.zeros((6, 6))

        for i in range(6):

            J[:3, i] = np.cross(
                axes[i],
                position - origins[i]
            )

            J[3:, i] = axes[i]

        return position, orientation, J

    def solve(self, target_p, target_R, initial_q):

        q = np.array(initial_q, dtype=float).copy()

        for _ in range(400):

            p, R, J = self.forward(q)

            ep = target_p - p

            er = 0.5 * sum(
                np.cross(R[:, i], target_R[:, i])
                for i in range(3)
            )

            if (
                np.linalg.norm(ep) < 0.0005
                and np.linalg.norm(R - target_R) < 0.005
            ):
                return q, True

            error = np.concatenate((ep, er))

            damping = 0.025

            dq = J.T @ np.linalg.solve(
                J @ J.T
                + damping**2 * np.eye(6),
                error
            )

            q += np.clip(dq, -0.08, 0.08)

            q = np.clip(
                q,
                self.lower + 0.001,
                self.upper - 0.001
            )

        return q, False


if __name__ == "__main__":

    ik = KinovaIK()

    initial_q = np.array([
        1.051122,
        0.105117,
        1.744147,
        1.570796,
        -1.639030,
        -1.051121
    ])

    target_position = np.array([
        0.066, 0.25, 0.65
    ])

    target_orientation = rpy_matrix([
        0, 0, np.pi / 2
    ])

    q, success = ik.solve(
        target_position,
        target_orientation,
        initial_q
    )

    p, R, J = ik.forward(q)

    print("IK success:", success)
    print("Joint angles:", np.round(q, 5))
    print("Camera position:", np.round(p, 5))
    print("Position error:", np.linalg.norm(
        target_position - p
    ))
    print("Orientation error:", np.linalg.norm(
        target_orientation - R
    ))

