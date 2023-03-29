cmake_minimum_required(VERSION 3.8)
project(autonomus_car_sim_base_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# ...
ament_python_install_package(${PROJECT_NAME})

# Python programs
install(PROGRAMS
  scripts/gazebo_lane_tracking.py
  scripts/gazebo_yolo.py
  DESTINATION lib/${PROJECT_NAME}
)

# Directories
install(DIRECTORY
  launch
  msg
  urdf
  worlds
  yolov5
  weights
  DESTINATION share/${PROJECT_NAME}/
)

# MSG files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MekatronomYolo.msg"
 )
 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
