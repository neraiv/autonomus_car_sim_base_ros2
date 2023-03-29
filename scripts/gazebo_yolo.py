#!/usr/bin/env python3
# This code runs Yolov5 model and sends detected objects in MekatronomYolo.msg format
# U can change add ur model into autonomus_car_sim_base_ros2/weights and change the
# name of model in main (model_path = 'your_model_name'). This code may not work on 
# different model types like TensorRT or ONNX.

import imp
import os, sys
from tkinter import Image
import cv2
import pyrealsense2 as rs2
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math
import argparse


import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from ament_index_python.packages import get_package_share_directory

from autonomus_car_sim_base_ros2.msg import MekatronomYolo
from std_msgs.msg import String
from sensor_msgs.msg import Image as msg_Image,CompressedImage as msg_CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

PATH_PKG = get_package_share_directory('autonomus_car_sim_base_ros2')

ROOT = os.path.join(PATH_PKG,'yolov5') # YOLOv5 root directory
WEIGHTS_PATH = os.path.join(PATH_PKG,'weights')
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WEIGHTS_PATH = Path(os.path.relpath(WEIGHTS_PATH, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class detector(Node):
    # Ağağıdaki blogğun büyük bı kısmı modeli yolo ya yüklemek ve çalıştrımak la alakalı. Bizim eklediğimi kısmı belirttim.
    def __init__(self,model_path,publisher_topic,image_topic,color_info_topic,depth_topic,depth_info_topic):
        super().__init__('mekatronom_yolov5')
        qos_profile = QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=5)
        self.object_coordinates = []
        self.delete_objects = None
        self.weights=WEIGHTS_PATH / model_path  # model.pt path(s)
        self.source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
        self.data=ROOT / 'custom.yaml'  # dataset.yaml path
        self.imgsz=(480, 640)  # inference size (height, width)
        self.conf_thres=0.70  # confidence threshold             #yüzde kaç doğruluğun üstünde olanları alma 0.0 / %0 ,1.0 / %100 
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=5 # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference

        self.source = str(self.source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.streams') or (self.is_url and not self.is_file)
        self.screenshot = self.source.lower().startswith('screen')
        if self.is_url and self.is_file:
            self.source = check_file(self.source)  # download

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir


        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if self.device.type != 'cpu':
            self.model.warmup(imgsz=(1 if self.stride or self.model.triton else bs, 3, *imgsz))  # run once
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

        ################################## BURASI BİZİM EKLEDĞİMİZ KISIM #########################################
        self.bridge = CvBridge() # OpenCV and ROS msg bridge
        
        self.yolom = MekatronomYolo() # My msg file to make communcation clear
        
        #self.intrinsics = None

        self.img_cb_done = False      # IMG callback check flag
        self.depth_cb_done = False    # Depth callback check flag

        # Yolo topic publisher to use other codes
        self.yolo_publisher = self.create_publisher(MekatronomYolo,publisher_topic,10)
        

        # IMG and DEPTH subscribtion
        self.depth_sub = self.create_subscription(msg_Image,depth_topic,self.depthCallback1,qos_profile=qos_profile) 
        self.image_sub = self.create_subscription(msg_Image,image_topic,self.imageCallback1,qos_profile=qos_profile) 
        
        # I didnt send img to yolov5 in img callback, becuse when i tried 2 cameras in a different code and ona of them was slower
        # when i send img in img callback. So instead of img callback i used camera info topic.
        self.ana_callback =  self.create_subscription(CameraInfo,color_info_topic,self.yolo,qos_profile=qos_profile) 

        #variables
        self.start_time = time.time()

    def yolo(self,data):
        
        try:
            Yolomsg = MekatronomYolo()    
            if not self.img_cb_done  or not self.depth_cb_done:
                print(self.img_cb_done,self.depth_cb_done)
                return
            
            color_frame = self.color_frame
            depth_frame = self.depth_frame

            img = np.asanyarray(color_frame)
            im = img
            depth_image = np.asanyarray(depth_frame)

            # Letterbox
            im0 = im.copy()
            im = im[np.newaxis, :, :, :]        

            # Stack
            im = np.stack(im, 0)

            # Convert
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)

        

            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            self.dt[0] += t2 - t1

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=self.augment, visualize=False)
            t3 = time_sync()
            self.dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            self.dt[2] += time_sync() - t3
            
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                bbox_areas = ""
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    self.delete_objects = True
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label2 = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]}')
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                        x = int((xyxy[0] + xyxy[2])/2)
                        y = int((xyxy[1] + xyxy[3])/2)
                        lengthx = int((xyxy[0] + xyxy[2]))
                        lengthy = int((xyxy[1] + xyxy[3]))
                        dist = depth_frame[int(y),int(x)]
                        dist2= 4*dist
                        Xtarget = dist2*(x)
                        Ytarget = dist2*(y)
                        
                        bbox_areas = (int((xyxy[2]- xyxy[0])))        
                        #self.object_coordinates.append([Xtarget,Ytarget,Ztarget])



                        if dist2 < 13: 
                            Yolomsg.depth.append(dist2)   
                            Yolomsg.object.append(label2)
                            Yolomsg.size.append(bbox_areas)
                            if (Xtarget < 0):
                                Side = "Sol"
                            elif (Xtarget > 0):
                                Side = "Sag"
                            elif(Xtarget == 0):
                                Side = "Orta"
                            Yolomsg.side.append(Side)
                        #print(lengthx,dist)
                        #Yolomsg.coordinates.append([Xtarget,Ytarget,Ztarget])

                        cv_font = cv2.FONT_HERSHEY_SIMPLEX
                        try:
                            coordinate_text = "(" + str(round(dist2)) + ")"
                        except OverflowError as e:
                            print(e)
                            coordinate_text = "(" + str(round(50)) + ")"
                        cv2.putText(im0, text=coordinate_text, org=(int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)),
                        fontFace = cv_font, fontScale = 0.7, color=(0, 232, 124), thickness=2, lineType = cv2.LINE_AA)
                    
            if(len(Yolomsg.object) != 0 and dist2 <13):
                self.get_logger().info( "%s" "%s" "%s"% (Yolomsg.depth,Yolomsg.object,Yolomsg.side))
                self.yolo_publisher.publish(Yolomsg)
                print("-------------------------")

            end_time = time.time()
            fps = 1 / np.round(end_time - self.start_time, 2)      
            cv2.putText(im0, "FPS: " + str(int(fps)), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
                          
            
            cv2.imshow("stacked", cv2.resize(im0,[640,480]))
            self.start_time = time.time()
            cv2.waitKey(3)

        except CvBridgeError as e:
            print(e)
            return

        # gelen görüntüyü ROS_Img topicten opencv img ine dönüştür.
    def imageCallback1(self,msg):
        try:
            self.img_cb_done = True
            self.color_frame = self.bridge.imgmsg_to_cv2(msg,"bgr8")              
        except CvBridgeError as e:
            print(e)
            return
         
    def depthCallback1(self,msg):
        try:
           self.depth_cb_done = True
           self.depth_frame = self.bridge.imgmsg_to_cv2(msg,msg.encoding)
           self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_frame, alpha=0.08), cv2.COLORMAP_JET)
        except CvBridgeError as e:
            print(e)
            return

def main(args=None):
    rclpy.init(args=args)
    model_path = 'best.pt'

    publisher_topic = '/mekatronom/yolov5'

    camera = 'camera2'
    image_topic = '/'+camera+'/image_raw'
    color_info_topic = '/'+camera+'/camera_info'
    depth_topic = '/'+camera+'/depth/image_raw'
    depth_info_topic = '/'+camera+'/depth/camera_info'

    Detect = detector(model_path,publisher_topic,image_topic,color_info_topic,depth_topic,depth_info_topic)
    
    try: 
        rclpy.spin(Detect)
    except KeyboardInterrupt: 
        cv2.destroyAllWindows()

    Detect.destroy_node()
    rclpy.shutdown()

    
 
if __name__ == '__main__':
    main(sys.argv)
