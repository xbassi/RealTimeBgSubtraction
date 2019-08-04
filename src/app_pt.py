#!/usr/bin/env python


import os
from io import BytesIO

import numpy as np
from PIL import Image,ImageFilter

import sys
import datetime
import cv2
import time
import imutils
import sys
sys.path.insert(0, './src/matting')

import matting.camera as camera_input


import argparse

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./src/matting/pre_trained/erd_seg_matting/model/model_obj.pth', help='preTrained model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')

args = parser.parse_args()

# print(args)

model = camera_input.load_model()


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    # print(height,width)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    # print(bound_w, bound_h)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat







from flask import Flask, render_template, Response
# from camera_opencv import Camera



app = Flask(__name__)



# Camera.set_video_source("./data/small.mp4")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return render_template('video.html')


@app.route('/fps')
def fps():
    return str(int(FPS))


def gen():
    count = 0
    camera = cv2.VideoCapture(0)

    offset = 1
    seg_map = None
    save_size = None
    kernel = np.ones((4,4), np.uint8) 
    while True:
        flag = 0
        # frame = camera.get_frame()
        start = time.time()
        ret, frame = camera.read()
        # frame = cv2.resize(frame,(256,256))

        if ret != None:

          frame = cv2.flip(frame,1)
          # jpeg_str = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # jpeg_str = rotate_image(jpeg_str, -90)
          # shape = jpeg_str.shape
          # orignal_im = Image.fromarray(jpeg_str)

          # if count % offset == 0:
          #   flag = 1
          frame = camera_input.seg_process(args, frame, model)
            # save_size = orignal_im.size
          

          # seg_map = np.array(seg_map).astype(np.uint8)

          # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(seg_map, connectivity=4)
          # sizes = stats[1:, -1]; nb_components = nb_components - 1

          # if len(sizes) != 0:

          #   seg_map = np.zeros((output.shape))
          #   max_e = sizes.tolist().index(max(sizes.tolist()))
          #   seg_map[output == max_e + 1] = 255

          # # seg_map_cv2 = np.array(seg_map).astype(np.float32)

          # ret_frame = drawSegment(orignal_im.resize(save_size), seg_map, count, shape)
          # ret_frame = np.array(ret_frame) 
          # ret_frame = ret_frame[:, :, ::-1].copy()
          # frame = cv2.resize(frame,())
          frame = imutils.resize(frame, width=512)
          frame = cv2.imencode('.jpg', frame)[1].tobytes() 
          # frame = ret_frame

          # print(count)
          count = count + 1

          end  = time.time()
          
          global FPS

          FPS = 1 / (end - start)
          print(FPS, end = "\r")

          yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    FPS = -1

    # model_file = "/home/prince/Downloads/deeplabv3_257_mv_gpu.tflite"
    # interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    # interpreter.allocate_tensors()
    
    app.run(host='0.0.0.0', debug=True)

