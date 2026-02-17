# Gazebo Integration Guide

Integration guide for connecting Gazebo (Harmonic) with ROS 2 to the
PAI Oncology Trial FL platform for sensor-rich surgical robotics
simulation and hardware-in-the-loop testing in oncology clinical trials.

## System Requirements

| Component        | Minimum                     | Recommended                   |
|------------------|-----------------------------|-------------------------------|
| CPU              | 8 cores (x86_64)           | 16+ cores (x86_64)           |
| GPU              | OpenGL 3.3 capable         | NVIDIA RTX 3060+ (for sensors)|
| RAM              | 16 GB                       | 32 GB                         |
| OS               | Ubuntu 22.04 LTS            | Ubuntu 22.04 LTS              |
| ROS 2            | Humble Hawksbill (LTS)      | Iron Irwini                   |
| Gazebo           | Harmonic (8.x)              | Harmonic 8.7                  |
| ros_gz bridge    | 1.0+                        | 1.0+                          |
| Python           | 3.10                        | 3.10                          |

Gazebo Harmonic is the current LTS release, pairing with ROS 2 Humble
(LTS through May 2027) and ROS 2 Iron. Legacy Gazebo Classic (1-11)
is not supported. Reference: [Gazebo Releases](https://gazebosim.org/docs/harmonic/releases)

## Installation

### 1. Install ROS 2 Humble

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash
```

Reference: [ROS 2 Humble Installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

### 2. Install Gazebo Harmonic and ros_gz Bridge

```bash
sudo apt install -y gz-harmonic ros-humble-ros-gz
```

Reference: [Gazebo Harmonic Installation](https://gazebosim.org/docs/harmonic/install_ubuntu)

### 3. Install PAI Oncology Trial FL

```bash
pip install -e /path/to/pai-oncology-trial-fl
```

## ROS 2 Integration for Clinical Robotics

### Topic Architecture

All topics are namespaced under `/oncology_trial/`:

```
/oncology_trial/
  surgical_arm/
    joint_states          [sensor_msgs/msg/JointState]
    end_effector_pose     [geometry_msgs/msg/PoseStamped]
    force_torque          [geometry_msgs/msg/WrenchStamped]
  tumor_phantom/
    pose                  [geometry_msgs/msg/PoseStamped]
  sensors/
    depth_camera/image    [sensor_msgs/msg/Image]
    depth_camera/points   [sensor_msgs/msg/PointCloud2]
  telemetry/
    features              [std_msgs/msg/Float64MultiArray]
```

### Gazebo SDF World for Oncology Simulation

```xml
<?xml version="1.0" ?>
<sdf version="1.11">
  <world name="oncology_or">
    <physics type="dart">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <light type="directional" name="or_light">
      <pose>0 0 3 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>0 0 -1</direction>
      <cast_shadows>true</cast_shadows>
    </light>
    <model name="operating_table">
      <static>true</static>
      <pose>0 0 0.7 0 0 0</pose>
      <link name="table_surface">
        <collision name="col"><geometry><box><size>1.0 0.6 0.04</size></box></geometry></collision>
        <visual name="vis"><geometry><box><size>1.0 0.6 0.04</size></box></geometry></visual>
      </link>
    </model>
    <model name="depth_camera">
      <static>true</static>
      <pose>0 0 1.8 0 1.5708 0</pose>
      <link name="camera_link">
        <sensor name="rgbd_sensor" type="rgbd_camera">
          <camera>
            <horizontal_fov>1.047</horizontal_fov>
            <image><width>640</width><height>480</height></image>
            <clip><near>0.1</near><far>5.0</far></clip>
          </camera>
          <update_rate>30</update_rate>
          <topic>/oncology_trial/sensors/depth_camera</topic>
        </sensor>
      </link>
    </model>
    <include>
      <uri>model://surgical_arm_description</uri>
      <pose>0 -0.3 0.7 0 0 0</pose>
    </include>
  </world>
</sdf>
```

Reference: [SDFormat 1.11](http://sdformat.org/spec?ver=1.11)

### Launch File with ros_gz Bridge

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py",
        )),
        launch_arguments={"gz_args": "-r oncology_or.sdf"}.items(),
    )
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=[
            "/oncology_trial/surgical_arm/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/oncology_trial/sensors/depth_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
    )
    return LaunchDescription([gz_sim, bridge])
```

Reference: [ros_gz Bridge](https://github.com/gazebosim/ros_gz)

## Sensor Simulation for Oncology Environments

| Sensor             | Gazebo Plugin     | Oncology Use Case                              |
|--------------------|-------------------|-------------------------------------------------|
| RGBD Camera        | `rgbd_camera`     | Surgical field visualization, tumor segmentation |
| Force/Torque       | `force_torque`    | Tissue interaction monitoring                    |
| Contact Sensor     | `contact`         | Collision detection near critical anatomy        |
| IMU                | `imu`             | Instrument orientation tracking                  |
| GPU Lidar          | `gpu_lidar`       | OR environment mapping                           |

Reference: [Gazebo Sensors](https://gazebosim.org/docs/harmonic/sensors)

### Telemetry Collector Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import numpy as np
from physical_ai.robotic_integration import SurgicalRobotInterface, RobotType

class OncologyTelemetryCollector(Node):
    """ROS 2 node collecting surgical telemetry for FL training."""
    def __init__(self):
        super().__init__("oncology_telemetry_collector")
        self.robot = SurgicalRobotInterface(
            robot_id="site_gamma_gazebo_01", robot_type=RobotType.DA_VINCI,
        )
        self.robot.connect()
        self.create_subscription(
            JointState, "/oncology_trial/surgical_arm/joint_states",
            self._joint_cb, 10)
        self.create_subscription(
            WrenchStamped, "/oncology_trial/surgical_arm/force_torque",
            self._ft_cb, 10)
        self._joints = None
        self._force = None

    def _joint_cb(self, msg): self._joints = np.array(msg.position)
    def _ft_cb(self, msg):
        f = msg.wrench.force
        self._force = np.array([f.x, f.y, f.z])
```

## Connecting to the Federated Learning Platform

```python
from physical_ai.simulation_bridge import SimulationBridge, RobotModel, ModelFormat
from physical_ai.framework_detection import FrameworkDetector

detector = FrameworkDetector()
detector.detect()
pipeline = detector.recommend_pipeline()
# Gazebo recommended for "ros2_integration"

bridge = SimulationBridge()
surgical_arm = RobotModel(
    name="surgical_arm_gazebo",
    source_format=ModelFormat.SDF,
    num_joints=7, num_links=8, mass_kg=2.3,
    joint_limits=[
        (-3.14, 3.14), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14),
        (-1.57, 1.57), (-1.57, 1.57), (0.0, 0.04),
    ],
    metadata={"framework": "gazebo", "version": "harmonic"},
)
bridge.register_model(surgical_arm)

# Cross-framework validation via URDF conversion
converted_urdf = bridge.convert(surgical_arm, ModelFormat.URDF)
validation = bridge.validate_conversion(surgical_arm, converted_urdf)
```

## Troubleshooting

| Symptom                             | Cause                              | Fix                                                   |
|-------------------------------------|------------------------------------|-------------------------------------------------------|
| `[Err] Unable to find fuel model`   | Model URI not in `GZ_SIM_RESOURCE_PATH` | Export `GZ_SIM_RESOURCE_PATH` with model directory |
| Bridge topics not appearing         | Topic name mismatch                | Verify Gazebo topic names with `gz topic -l`          |
| Depth camera returns blank images   | GPU rendering not available        | Install NVIDIA drivers or use `ogre2` engine          |
| Physics instability with DART       | Step size too large                | Reduce `max_step_size` to 0.001                       |
| `ros2 topic echo` shows no data     | Bridge not running                 | Ensure `parameter_bridge` node is launched             |

## Source Citations

1. Gazebo Sim Documentation: https://gazebosim.org/docs/harmonic
2. Gazebo Harmonic Installation: https://gazebosim.org/docs/harmonic/install_ubuntu
3. SDFormat Specification 1.11: http://sdformat.org/spec?ver=1.11
4. ROS 2 Humble Documentation: https://docs.ros.org/en/humble/
5. ros_gz Bridge Package: https://github.com/gazebosim/ros_gz
6. Gazebo Sensor Plugins: https://gazebosim.org/docs/harmonic/sensors
7. Gazebo Releases and Compatibility: https://gazebosim.org/docs/harmonic/releases
