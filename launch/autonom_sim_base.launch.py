import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess,DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,Command
from launch_ros.actions import Node

import xacro

def generate_launch_description():
  use_sim_time = LaunchConfiguration('use_sim_time', default='true')

  package_dir = get_package_share_directory('autonomus_car_sim_base_ros2')

  world_file  = os.path.join(package_dir,'worlds','robotaksi2_new')
  
  urdf_model = LaunchConfiguration('urdf_model')

  default_urdf_model_path = os.path.join(package_dir, 'urdf','merco.urdf')
  
  remappings = [('/tf', 'tf'),
                ('/tf_static', 'tf_static')]

  declare_urdf_model_path_cmd = DeclareLaunchArgument(
    name='urdf_model', 
    default_value=default_urdf_model_path,
    description='Absolute path to robot urdf file.')

  start_robot_state_publisher_cmd = Node(
    #condition=IfCondition(PythonExpression(["'", use_robot_state_pub, "' == 'True' and '", use_backward_drive, "' == 'False'"])),
    #condition=IfCondition(use_robot_state_pub),
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'use_sim_time': use_sim_time, 
    'robot_description': Command(['xacro ', urdf_model])}],
    remappings=remappings,
    arguments=[default_urdf_model_path])
  
  start_gazebo_server_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen')

  start_gazebo_client_cmd = ExecuteProcess(  
        cmd=['gzclient'],
        output='screen')

  spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'umut',
                   '-file', urdf_model,
                   '-x', '-6.22',
                   '-y', '-6.64',
                   '-z', '0.05',
                   '-Y', '-0.04'],
        output='screen')


  return LaunchDescription([
        #ExecuteProcess(
         # cmd =['gazebo','--verbose',world_file,'-s','libgazebo_ros_factory.so'],
          #output='screen'),
        declare_urdf_model_path_cmd,
        start_gazebo_server_cmd,
        start_gazebo_client_cmd,
        spawn_entity_cmd,       
        start_robot_state_publisher_cmd
        #spawn_entity_cmd
        
  ])
  
