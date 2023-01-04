import pyrealsense2 as rs
import numpy as np
import cv2
import os
import yolov7_traffic_cone_box_center as yolo
import random
import onnxruntime as ort
import serial
import time

# Setup inference for ONNX model
cuda = True
w = "traffic_cone_yolov7.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
names = ['blue cone', 'traffic cone', 'yellow cone']
colors = {name: [random.randint(0, 128) for _ in range(3)] for i, name in enumerate(names)}

# img = cv2.imread('../Resources/cones2.jpg')
# print(img.shape)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

# curr_frame = 0

# init x y position to collect depth
x=320
y=240

# init serial communication to control servos
# ser = serial.Serial('COM9',baudrate=9600,timeout=1)
time.sleep(0.5)
posX = 90
posY = 90

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if not depth_frame or not color_frame:
        if not color_frame:
            print("No depth frame or color frame")
            continue

        # Intrinsics & Extrinsics
        # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # print(depth_intrin.ppx, depth_intrin.ppy)

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # print(len(color_image))

        cone_data = yolo.inference(color_image)
        center = []

        # cone_coors = []
        # closest_cone_coor = []

        if len(cone_data):
            for i,([x0,y0,x1,y1],name,color) in enumerate(cone_data):
                try:
                    cv2.rectangle(color_image, (x0, y0), (x1, y1), color, 2)
                    cv2.rectangle(color_image, (x0, y0 - 25), (x1, y0), color, -1)
                    cv2.putText(color_image, name, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
                    center.append([(x0 + x1) // 2, (y0 + y1) // 2])

                    # depth = depth_frame.get_distance(x, y)
                    # depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
                    # cone_coors.append([round(depth_point[0]*100,3), round(depth_point[2]*100,3), round(-depth_point[1]*100,3)])
                    # text = "%.5lf, %.5lf, %.5lf\n" % (depth_point[0]*100, depth_point[1]*100, depth_point[2]*100)
                    # print(text)
                except:
                    pass

            center = np.average(center, axis=0)
            x = int(center[0])
            y = int(center[1])

            # closest_cone_coor = cone_coors[0]
            # for i, (x, y, z) in enumerate(cone_coors):
            #     if closest_cone_coor[2] > z:
            #         closest_cone_coor = cone_coors[i]
        if x and y:
            errorPan = (x) - 640 / 2
            print('errorPan', errorPan)
            # print(type(errorPan))
            if abs(errorPan) > 20:
                posX = posX - errorPan / 30
                print(type(posX))
            if posX > 160:
                posX = 160
                print("Out of range")
            if posX < 20:
                posX = 20
                print("out of range")

            errorTilt = (y) - 480 / 2
            print('errorTilt', errorTilt)
            # print(type(errorPan))
            if abs(errorTilt) > 20:
                posY = posY - errorTilt / 30
                print(type(posY))
            if posY > 160:
                posY = 160
                print("Out of range")
            if posY < 20:
                posY = 20
                print("out of range")

            servoPos = F"X{int(posX)}Y{int(posY)}Z"

            # ser.write(servoPos.encode())
            print('servoPos = ', servoPos)

        # print(closest_cone_coor)

        # arm control
        # if len(closest_cone_coor):
        #     if distanse > ??:
        #         get arm coor
        #         calculate and get target position
        #         pan arm
        #     else:
        #         save arm location
        #         collect cone
        #         pick up and bring to storage zone
        #         return to saved location



        # cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

        # curr_frame += 1
finally:

    # Stop streaming
    pipeline.stop()