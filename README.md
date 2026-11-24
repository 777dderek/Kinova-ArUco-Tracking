# Kinova ArUco Tracking

EECS 4421 Robotics Project

## Environment
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Kinova Gen3 Lite (6 DOF)

## Features
- Wrist-mounted camera
- ArUco marker detection
- Numerical inverse kinematics
- Autonomous Cartesian X-Z tracking
- Fixed Y and camera orientation
- Live error and velocity overlays

## Controls
- 1, 2, 3: Select marker
- Space: Pause or resume
- Q: Quit

## Project folders
- ros2_kortex: Modified Kinova simulation source
- aruco_board: Detection, IK and tracking scripts
- gazebo_models: ArUco board model

Simulation tested. Physical robot integration pending.
Local ROS dependencies must be configured before building.
