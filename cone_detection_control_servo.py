import serial
import time
import numpy as np
import cv2

import pyrealsense2
from realsense_depth import *

from roboflow import Roboflow
rf = Roboflow(api_key="W3yN6Ivo4oDFuV85x0XF")
project = rf.workspace().project("traffic_cone-b2tsm")
model = project.version(2).model

# Initialize Camera Intel Realsense
dc = DepthCamera()

ser = serial.Serial('COM9',baudrate=9600,timeout=1)
time.sleep(0.5)
posX = 90
posY = 90
# print(type(pos))

# cam = cv2.VideoCapture(0)

while True:
    # ret, img = cam.read()
    ret, depth_frame, color_frame = dc.get_frame()

    x, y, w, h = [], [], [], []
    arr = model.predict(color_frame, confidence=40, overlap=30).json().get('predictions')

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    if len(arr) > 0:
        for i in range(len(arr)):
            x.append(arr[i].get('x'))
            y.append(arr[i].get('y'))
            # w.append(arr[i].get('width'))
            # h.append(arr[i].get('height'))

    x = np.average(x)
    y = np.average(y)
    print(f"x: {x}  y:{y}")
    # w = np.average(w)
    # h = np.average(h)

    if x and y:
        errorPan = (x) - 640/2
        print('errorPan', errorPan)
        # print(type(errorPan))
        if abs(errorPan) > 20:
            posX = posX - errorPan/30
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


        ser.write(servoPos.encode())
        print('servoPos = ', servoPos)
        # print(type(pos))
        # time.sleep(0.1)

    cv2.imshow('MBS3523 Webcam', color_frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

ser.close()
# cam.release()
dc.release()
cv2.destroyAllWindows()