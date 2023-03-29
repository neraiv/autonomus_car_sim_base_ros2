#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  This code detects lines using filters then implements hough lines algorithm then calclulates
#  required turn angle. Sends data to gazebo robot named umut and drives the car (umut).

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from mekatronom.msg import MekatronomYolo
from geometry_msgs.msg import Twist

import cv2
import numpy as np
import math

class LaneTracker(Node): 
    def __init__(self,image_topic_name,yolo_topic_name,lidar_topic_name,twist_topic_name):
        super().__init__("LaneTracker") 

        qos_profile = QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=5) 

        self.bridge = CvBridge() # ROS2 to OpenCV bridge or vice-versa
        
        self.twist = Twist()     # ROS msg to move objects

        # Lane tracking whic is CAM0 in URDF
        self.image_sub = self.create_subscription(
            Image, image_topic_name, self.image_callback, qos_profile=qos_profile)

        # Lidar subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan, lidar_topic_name, self.lidar_callback, qos_profile=qos_profile)
        
        # Yolov5 subscriber
        self.yolo_sub = self.create_subscription(
            MekatronomYolo, yolo_topic_name, self.yolo_callback, qos_profile=qos_profile)

        # Twist publisher
        self.twist_pub = self.create_publisher(Twist, twist_topic_name, 10)
        
        # Twist publish timer
        self.twist_pub_timer = self.create_timer(0.05, self.twist_pub_timer_callback)


    def image_callback(self,data):

        def detect_edges(frame,gray_frame):
            # White Lines Detect 
            mask_white = cv2.inRange(gray_frame, 200, 255)
            edges = cv2.Canny(mask_white, 50, 100)

            # To Detect Yellow Lines 
            # lower = np.array([0, 0, 100], dtype = "uint8") # lower limit of white color 0, 0, 100      22, 93, 0
            # upper = np.array([10, 255, 255], dtype="uint8") # upper limit of white color 10, 255, 255  45, 255, 255
            # mask_yellow = cv2.inRange(frame,lower,upper) # this mask will filter out everything but white
            # result = cv2.bitwise_and(frame, frame, mask=mask_yellow)
            # cv2.imshow('result', result)

            return edges


        def region_of_interest(edges):
            height, width = edges.shape # extract the height and width of the edges frame
            mask = np.zeros_like(edges) # make an empty matrix with same dimensions of the edges frame

            # only focus lower half of the screen
            # specify the coordinates of 4 points (lower left, upper left, upper right, lower right)

            
            # u can change as desired different roi may work better

            polygon = np.array([[                              # Sim roi
                (50, height-50),
                (170,  height/2+50),      #(0,  height/2),
                (width-170 , height/2+50),#(width , height/2)
                (width-50, height-50),
            ]], np.int32)

            # polygon = np.array([[                            # Sim roi
            #     (0, height),
            #     (350,  height/2),      #(0,  height/2),
            #     (width-350 , height/2),#(width , height/2)
            #     (width, height),
            # ]], np.int32)

            cv2.fillPoly(mask, polygon, 255) # fill the polygon with blue color
            roi = cv2.bitwise_and(edges, mask)
            return roi

        def detect_line_segments(cropped_edges):
            rho = 1
            theta = np.pi / 180
            min_threshold = 10
            line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,np.array([]), minLineLength=80, maxLineGap=50)
            
            return line_segments

        def make_points(frame, line):
            height, width, _ = frame.shape
            slope, intercept = line
            y1 = height  # bottom of the frame
            y2 = int(y1/2)  # make points from middle of the frame down
            
            if slope == 0: 
                slope = 0.1    

            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

            return [[x1, y1, x2, y2]]
        
        def average_slope_intercept(frame, line_segments):
            lane_lines = []
            print_string = ""
            global intercept
            intercept =0
            if line_segments is None:
                #print("no line segment detected")
                return lane_lines

            height, width,_ = frame.shape
            left_fit = []
            right_fit = []
            boundary = 1/2

            left_region_boundary = width * (1 - boundary) 
            right_region_boundary = width * boundary 
            
            for line_segment in line_segments:
                for x1, y1, x2, y2 in line_segment:
                    if x1 == x2:
                        print_string = "(skipping vertical lines (slope = infinity)"
                        continue
                    
                    

                    fit = np.polyfit((x1, x2), (y1, y2), 1)
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - (slope * x1)
                    #print_string += "SLOPE: " +str(slope) +"\n Lane Lines:"+ str(lane_lines) +"\n (X2-X1):"+ str(x2-x1) + "\n"


                    if slope < 0:
                        if x1 < left_region_boundary and x2 < left_region_boundary:
                            left_fit.append((slope, intercept))
            
                    else:
                        if x1 > right_region_boundary and x2 > right_region_boundary:
                            right_fit.append((slope, intercept))

            left_fit_average = np.average(left_fit, axis=0)
            if len(left_fit) > 0:
                lane_lines.append(make_points(frame, left_fit_average))

            right_fit_average = np.average(right_fit, axis=0)
            if len(right_fit) > 0:
                lane_lines.append(make_points(frame, right_fit_average))

            #print(print_string)

            return lane_lines

        def display_lines(frame, lines, line_color=(255, 0, 0), line_width=12): # line color (B,G,R)
            line_image = np.zeros_like(frame)

            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

            line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  
            return line_image
        
        def get_steering_angle(frame, lane_lines):
            height, width, _ = frame.shape
        
            if len(lane_lines) == 2: # if two lane lines are detected

                _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
                _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane_lines array
                mid = int(width / 2)
                x_offset = (left_x2 + right_x2) / 2 #- mid
                y_offset = int(height/2)  

            elif len(lane_lines) == 1: # if only one line is detected
                x1, _, x2, _ = lane_lines[0][0]
                x_offset = x2 - x1
                y_offset = int(height / 2)


            elif len(lane_lines) == 0: # if no line is detected
                x_offset = 0
                y_offset = int(height / 2)

            angle_to_mid_radian = math.atan(x_offset / y_offset)
            angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
            
            steering_angle = int(angle_to_mid_deg + 90)


            #print(steering_angle)

            return steering_angle 

        def display_heading_line(frame, steering_angle, line_color=(0, 80, 255), line_width=5):

            heading_image = np.zeros_like(frame)
            height, width, _ = frame.shape

            steering_angle_radian = steering_angle / 180.0 * math.pi
            x1 = int(width / 2)
            y1 = height
            x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
            y2 = int(height/2)
            

            cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

            heading_image = cv2.addWeighted(frame, 0.6, heading_image, 1, 1)

            return heading_image 

        def haraket(steering_angle):
            
            # You can write ur traffic sign conditions under this.
            if 1 == 0:
                pass

            # This code works when there is no traffic sign and when no need to turn. Just keeps the car in lines.
            else:
                speed = 0.3 # m/s

                # This statement needed when lanetracker detects 1 line as 2 lines. 
                if intercept > 700 and intercept < 1200:                    
                    turn = -0.2 # radian
                elif steering_angle == 90:   
                    turn = 0.0
                elif steering_angle < 90:                 
                    turn  = 0.2
                elif steering_angle >90:
                    turn  = -0.2
                else:
                    turn = 0.0
            
            
            print(steering_angle)
            self.twist.linear.x = float(speed)
            self.twist.angular.z = float(turn)
            

        img = self.bridge.imgmsg_to_cv2(data)
        img_resized = cv2.resize(img, (540,500))
        gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        

        edges = detect_edges(img_resized,gray_image) 
        roi = region_of_interest(edges)
        
        print(roi.argmax())
        line_segments = detect_line_segments(roi)
        lane_lines = average_slope_intercept(rgb_image,line_segments)
        lane_lines_image = display_lines(rgb_image,lane_lines)
        steering_angle = get_steering_angle(rgb_image, lane_lines)
        #heading_image = display_heading_line(lane_lines_image,steering_angle)

        heading_line_segments = detect_line_segments(edges)
        heading_lane_lines = average_slope_intercept(rgb_image,heading_line_segments)
        heading_lane_lines_image= display_lines(rgb_image,heading_lane_lines)
        heading_image = display_heading_line(heading_lane_lines_image,steering_angle)

        
        haraket(steering_angle)

        # cv2.imshow("edges",edges)
        # cv2.imshow("roi",roi)
        # cv2.imshow("heading_image",heading_image)
        
        vis = np.concatenate((cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB), heading_image), axis=1)
        cv2.imshow("vis",vis)
        
        # edges = cv2.resize(edges,([540,500]))
        # roi = cv2.resize(roi,([540,500]))
        # heading_image = cv2.resize(heading_image,([540,500]))
        # img_stacked = stackImages(0.8,([edges,roi,heading_image]))
        # cv2.imshow("img_stacked",img_stacked)
        cv2.waitKey(3)


    def lidar_callback(self,data):
        pass

    def yolo_callback(self,data):
        if len(data.object) != 0:
            print(data.object)
            print(str(data.depth[0]))
            print(str(data.size[0]))
        pass

    def twist_pub_timer_callback(self):

        self.twist_pub.publish(self.twist)


def main(args=None):
    rclpy.init(args=args) # Node u başlat ros2 python kütüpyhanesi

    lane_tracker = LaneTracker(image_topic_name='/camera1/image_raw',  
                               yolo_topic_name ='/mekatronom/yolov5',
                               lidar_topic_name='/scan',
                               twist_topic_name='/umut/cmd_vel')

    try:
        rclpy.spin(lane_tracker)
    except KeyboardInterrupt:
        lane_tracker.twist.angular.z = float(0)
        lane_tracker.twist.linear.x = float(0)
        lane_tracker.twist_pub.publish(lane_tracker.twist)

    cv2.destroyAllWindows()
    lane_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()