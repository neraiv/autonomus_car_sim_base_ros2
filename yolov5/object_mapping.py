# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
from tkinter import ttk
import tkinter as tk
import tkinter.font as font
import threading
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

cv_font = cv2.FONT_HERSHEY_SIMPLEX

object_coordinates = [] #label, x, y
delete_objects = False

#@torch.no_grad()
def run():
    global object_coordinates, delete_objects
    weights=ROOT / 'yolov5s.pt'  # model.pt path(s)
    source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz=(416, 224)  # inference size (height, width)
    conf_thres=0.70  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=5 # maximum detections per image
    device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    print("device_num : " + str(device_num))
    device = select_device(device_num)
    print("device : " + str(device))
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    config = rs.config()
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    sayac = 0
    while(True):
        t0 = time.time()

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        im = img[8:232,4:420]
        depth_image = np.asanyarray(depth_frame.get_data())

        # Letterbox
        im0 = im.copy()
        im = im[np.newaxis, :, :, :]        

        # Stack
        im = np.stack(im, 0)

        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                delete_objects = True
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    if(label != 'dining table'):
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        x = int((xyxy[0] + xyxy[2])/2)
                        y = int((xyxy[1] + xyxy[3])/2)
                        dist = depth_frame.get_distance(x + 4, y + 8)*1000
                        Xtarget = dist*(x+4 - intr.ppx)/intr.fx - 35 #the distance from RGB camera to realsense center
                        Ytarget = dist*(y+8 - intr.ppy)/intr.fy
                        Ztarget = dist

                        dist2 = math.sqrt(math.pow(Xtarget,2)+math.pow(Ytarget,2)+math.pow(Ztarget,2))
                        
                        if (sayac < 20):
                            sayac = sayac +1
                        else:

                    
                           sayac = 0
                           print(dist2)

                        object_coordinates.append([label, Xtarget, Ztarget])

                    coordinate_text2 = "(" + str(round(dist2)) + ")"
                    #coordinate_text = "(" + str(round(Xtarget)) + "," + str(round(Ytarget)) + "," + str(round(Ztarget)) + ")"
                    #cv2.putText(im0, text=coordinate_text, org=(int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)),fontFace = cv_font, fontScale = 0.7, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
                    cv2.putText(im0, text=coordinate_text2,
                                org=(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)),
                                fontFace=cv_font, fontScale=0.7, color=(255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)

        cv2.imshow("IMAGE", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 





def animate():
    global object_coordinates, delete_objects
    circle_radius = 5
    scale = 1.25 # mm/pixel
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        canvas1.delete("all")
        for i in range(len(object_coordinates)):
            cx = object_coordinates[i][1]/scale + 200
            cy = 560 - object_coordinates[i][2]/scale
            x1 = cx - circle_radius
            x2 = cx + circle_radius
            y1 = cy - circle_radius
            y2 = cy + circle_radius   
            circle1 = canvas1.create_oval(x1, y1, x2, y2, fill = "green", tag=str(i))         
            canvas1.coords(circle1, x1, y1, x2, y2)
        if(delete_objects):
            object_coordinates.clear()
            delete_objects=False

def thread_start():
    thread_1.start() 
    thread_2.start()


#make a window
root = tk.Tk()

t = tk.StringVar()

thread_1 = threading.Thread(target=run)
thread_2 = threading.Thread(target=animate)
thread_1.daemon = True # die when the main thread dies
thread_2.daemon = True

#window title
root.title('mapping')
root.geometry("400x640")
my_font = font.Font(root, family="System", size = 28, weight="bold")

canvas1 = tk.Canvas(root, bg="white", height=560, width=400)
canvas1.place(x=0, y=0)
Start_button = tk.Button(text='START', font=my_font, command = thread_start)
Start_button.place(x = 120, y = 580, height = 40, width = 160)


root.mainloop()
