# Inference for ONNX model
import cv2
import random
import numpy as np
import onnxruntime as ort

cuda = True
w = "traffic_cone_yolov7.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
names = ['blue cone', 'traffic cone', 'yellow cone']
colors = {name: [random.randint(0, 128) for _ in range(3)] for i, name in enumerate(names)}

img = cv2.imread('../Resources/cones2.jpg')
# print(img.shape)

# HSV boundaries
hueLow = 0
hueHigh = 6
satLow = 220
satHigh = 255
valueLow = 105
valueHigh = 255

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # print(shape)
    # print(new_shape)
    # Scale ratio (new / old)

    r = np.min([new_shape[0] / shape[0], new_shape[1] / shape[1]])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = np.min([r, 1.0])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def inference(img):

    img_ori = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    # im.shape

    outname = [i.name for i in session.get_outputs()]
    # outname

    inname = [i.name for i in session.get_inputs()]
    # inname

    inp = {inname[0]: im}

    # ONNX inference
    outputs = session.run(outname, inp)[0]
    # outputs
    # print(list(enumerate(outputs)))

    # output image
    # ori_images = [img.copy()]



    conePos = []

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        # print(i,(batch_id,x0,y0,x1,y1,cls_id,score))
        # image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()

        # print(f"box    : {box}")
        # print(f"box[:2]: {box[:2]}")
        # print(f"box[2:]: {box[2:]}")

        # create bounding box
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        # cv2.rectangle(image,box[:2],box[2:],color,2)
        # cv2.rectangle(image,(box[0], box[1]-25),(box[2], box[1]),color,-1)
        # cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        conePos.append([box,name,color])






    # inference result
    # print(coneXYPos)
    # for a,b in coneXYPos:
    #     print(a, b)
    # cv2.imshow("mask",mask)
    # cv2.imshow("mask1",mask1)
    # Image.fromarray(ori_images[0])
    # img2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Traffic Cone",img2)
    # cv2.waitKey(0)
    return conePos


if __name__ == '__main__':
    result = inference(img)
    print(result)
    center = []
    for i, ([x0,y0,x1,y1],name,color) in enumerate(result):
        print(i,x0,y0,x1,y1,name,color)
        cv2.rectangle(img,(x0,y0),(x1,y1),color,2)
        cv2.rectangle(img,(x0,y0-25),(x1,y0),color,-1)
        cv2.putText(img,name,(x0,y0 - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        center.append([(x0+x1)//2,(y0+y1)//2])
    print(center)
    print(np.average(center,axis=0))
    center = np.average(center,axis=0)
    print(f'X: {int(center[0])}')
    print(f'Y: {int(center[1])}')
    cv2.imshow("Traffic Cone", img)
    cv2.waitKey(0)