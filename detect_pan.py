import argparse
import os
import platform
import sys
from pathlib import Path
from typing import List

import torch

# A library for scientific computing
import numpy as np
# Library for serial communication with Arduino
import serial
# Library for image processing and face detection (OpenCV)
import cv2
# Library for managing binary data(π.χ. convert values ​​to Byte)
import struct
# Library for pausing the program at a specific time
from time import sleep

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xyxy2xywhn, xywhn2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# This function takes two parameters, pan and tilt, which represent the angles to which the servos should move. It checks the values ​​to make sure they don't exceed the limits and then sends the values ​​to the Arduino via serial port as bytes.
def setServo(pan, tilt):

   # Some sanity checks to make sure we don't go over what our servo can do
   if pan < 0: pan = 0
   if pan > 180: pan = 180
   if tilt < 0: tilt = 0
   if tilt > 180: tilt = 180

   # We're reading these values as bytes on the arduino, why make it convert
   # strings to ints over there when we can just sent them natively from here?
   # So we're going to pack them and send as bytes

   ser.write(struct.pack('>BBBB',pan_chan,pan,tilt_chan,tilt))

# Calls setServo with calculated new values ​​for pan and tilt based on the given direction and number of degrees.
def moveServo(direction, degrees):
   # Calculate our new servo position and send it to setServo to move it

   print ("Direction: %s  Degrees: %d" % (direction, degrees))
   global cur_pan
   global cur_tilt
   if direction=="LEFT":
       cur_pan = cur_pan + degrees
   if direction=="RIGHT":
       cur_pan = cur_pan - degrees
   if direction=="UP":
       cur_tilt = cur_tilt - degrees
   if direction=="DOWN":
       cur_tilt = cur_tilt + degrees

   setServo(cur_pan,cur_tilt)

# He moves the servos to extreme positions and then brings them back to the center as part of an initial test.
def exerciseServo():
    # Exercise the servo then set it to the middle
    setServo(0,90)
    sleep(1)
    setServo(180,0)
    sleep(1)
    setServo(180,180)
    sleep(1)
    setServo(0,180)
    sleep(1)
    setServo(90,75)



@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global xywh, xyxy, det
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # Try Pandora
                    xywh_test: list[int] = [4]
                    xywh_test = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                    xywh_test = [x * 1000 for x in xywh_test]
                    print("x1,y1,x2,y2:", xywh_test)
                    x1 = xywh_test[0] * 0.64
                    y1 = xywh_test[1] * 0.48
                    x2 = xywh_test[2] * 0.64
                    y2 = xywh_test[3] * 0.48
                    print("/ width & height:", x1, y1, x2, y2)
                    x1 = round(x1, 1)
                    y1 = round(y1, 1)
                    x2 = round(x2, 1)
                    y2 = round(y2, 1)
                    print("round x1,y1,x2,y2:", x1, y1, x2, y2)

                    i = 0
                    while any(j is None for j in xywh_test):
                        i = i + 30
                        for j in range(0, 180, 10):
                            setServo(j, i)
                            print("setServo(", j, ",", i, ")")

                    width = x2 - x1
                    height = y1 - y2
                    print("width & height:", width, height)
                    cam_midpoint_x = 320
                    cam_midpoint_y = 240
                    camera_fov_x = 42  # Cameras view in x degrees
                    camera_fov_y = 50  # Cameras view in y degrees
                    # Detection returns the upper left of the bounding box of the face
                    # However we want the middle of the face, thankfully we get the width and height
                    center_x = x1 + (width / 2)
                    center_y = y2 + (height / 2)
                    print("center_x,y:", center_x, center_y)
                    #We're going to calculate where the camera needs to point based on a couple things and then move it
                    # if center_y < cam_midpoint_y && y1 == 0:
                    if center_y < cam_midpoint_y:
                        degrees_per_move = int((cam_midpoint_y - center_y) / (camera_fov_y / 2))
                        moveServo("UP", degrees_per_move)
                    elif center_y > cam_midpoint_y:
                        degrees_per_move = int((center_y - cam_midpoint_y) / (camera_fov_y / 2))
                        moveServo("DOWN", degrees_per_move)

                    if center_x > cam_midpoint_x:
                        degrees_per_move = int((center_x - cam_midpoint_x) / (camera_fov_x / 2))
                        moveServo("RIGHT", degrees_per_move)
                    elif center_x < cam_midpoint_x:
                        degrees_per_move = int((cam_midpoint_x - center_x) / (camera_fov_x / 2))
                        moveServo("LEFT", degrees_per_move)


            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    # Declaration of the port and speed of communication with the Arduino
    # Setup our Serial Port
    serial_port = "/dev/ttyACM0"  # "/dev/ttyUSB0"  # Serial port the Arduino is attached to
    serial_speed = 9600  # Speed we're talking to it

    # Initial values ​​are set for pan and tilt angles, image center tolerance, and camera field of view
    tilt_chan = 0  # This is the channel 'Tilt' is on when talking to Arduino
    pan_chan = 1  # This is the channel 'Pan' is on when talking to Arduino
    cur_pan = 90  # Sane defaults for the current pan
    cur_tilt = 70  # Sane defaults for the current tile

    # Opens the serial port for communication
    ser = serial.Serial(serial_port, serial_speed)
    exerciseServo()

    opt = parse_opt()
    main(opt)
