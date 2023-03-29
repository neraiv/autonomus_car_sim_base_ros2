# autonomus_car_sim_base_ros2
This is base simulation for autonomus car projects using Yolov5 and OpenCV. This package tested in ROS2 Galactic

# Install ROS2
This package tested in ROS2 Galactic but u can use it with other ROS2 distros after making minimal changes.
  https://docs.ros.org/en/galactic/Installation.html

# Clone repo
-> Create a workspace
  mkdir -p ros2_ws/src
  cd ros2_ws
  colcon build
  cd src
  git clone https://github.com/neraiv/autonomus_car_sim_base_ros2.git
  cd ..
  colcon build
  
# Install Yolov5 requiremnts 
Simply navigate to ../autonomus_car_sim_base_ros2/yolov5 directory and 

pip install -r requirements.txt 
 -> Some libraries may not install properly. Install them manually.
 
# Gazebo Models
This package has a car model, a race field and 17 traffic signs. Gazebo Sensors can't see gazebo models autside gazebo workspace, so to be
able to see objects there are 2 options add gazebo model path to Gazebo Model Path using .bashrc (or equal) or add models into .gazebo folder
located in Home. 
-> Navigate into ../autonomus_car_sim_base_ros2/gazebo_models directory and cut car_dae and models then paste them into .gazebo folder.
-> Gazebo creates .gazebo folder autmatically after creating first model in Gazebo.

# Now Everything Should be Ready
-> İnstall libraries if anyone missing :(

Open Three Terminals, one for simulation - one for lane_tracking and one for yolo

source the workspace on every terminal
source install/setup.bash

First launch the Sim
ros2 launch autonomus_car_sim_base_ros2 autonom_sim_base.launch.py

Start Yolo
ros2 run autonomus_car_sim_base_ros2 gazebo_yolo.py

Start Lane Tracking
ros2 run autonomus_car_sim_base_ros2 gazebo_lane_tracking.py




