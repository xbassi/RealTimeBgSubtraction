#!/usr/bin/env python


##Deep

import os
from io import BytesIO

import numpy as np
from PIL import Image,ImageFilter

import tensorflow as tf
import sys
import datetime
import cv2
import time


# from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper



##

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 256
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    self.sess = tf.Session(config=config, graph=self.graph)

  def run(self, image):
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    # print("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map


def drawSegment(baseImg, matImg, count, shape):
    # width, height = baseImg.size

    mask = np.array(matImg)
    # mask = mask[:, :, ::-1].copy() 

    aimg = np.array(baseImg)
    # aimg = aimg[:, :, ::-1].copy() 

    # mask_inv = cv2.bitwise_not(mask)

    # aimg = cv2.bitwise_and(aimg,aimg,mask_inv)
    # aimg = cv2.cvtColor(aimg,cv2.COLOR_BGR2RGB)

    # mask = np.invert(mask.astype(np.uint8))
    mask = 1 - mask
    mask[mask == 1] = 255
    # print(mask)

    aimg[:,:,0] = np.where(mask>0,mask[:,:],aimg[:,:,0])
    aimg[:,:,1] = np.where(mask>0,mask[:,:],aimg[:,:,1])
    aimg[:,:,2] = np.where(mask>0,mask[:,:],aimg[:,:,2])

    # dummyImg = np.clip(aimg, 0, 255).astype(np.uint8)

    # print("Hereeee")
    # dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    # for x in range(width):
    #           for y in range(height):
    #               color = matImg[y,x]
    #               (r,g,b) = baseImg.getpixel((x,y))
    #               if color == 0:
    #                   dummyImg[y,x,3] = 255
    #                   dummyImg[y,x,2] = 255
    #                   dummyImg[y,x,1] = 255
    #                   dummyImg[y,x,0] = 255
    #               else :
    #                   dummyImg[y,x] = [r,g,b,255]
    img = Image.fromarray(aimg)
    # img = img.convert("RGB")
    #img = img.resize((shape[0], shape[1]), Image.ANTIALIAS) 
    #img.save(outputFilePath+str(count).zfill(8)+'.jpg')
    return img


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


def gen():
    count = 0
    camera = cv2.VideoCapture(0)

    offset = 4
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

          jpeg_str = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # jpeg_str = rotate_image(jpeg_str, -90)
          shape = jpeg_str.shape
          orignal_im = Image.fromarray(jpeg_str)

          if count % offset == 0:
            flag = 1
            orignal_im, seg_map = MODEL.run(orignal_im)
            save_size = orignal_im.size
          

          seg_map = np.array(seg_map).astype(np.uint8)

          nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(seg_map, connectivity=4)
          #connectedComponentswithStats yields every seperated component with information on each of them, such as size
          #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
          sizes = stats[1:, -1]; nb_components = nb_components - 1

          # minimum size of particles we want to keep (number of pixels)
          # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
          # min_size = 300  

          #your answer image
          seg_map = np.zeros((output.shape))

          max_e = sizes.tolist().index(max(sizes.tolist()))

          seg_map[output == max_e + 1] = 255

          #for every component in the image, you keep it only if it's above min_size
          # for i in range(0, nb_components):
          #     if sizes[i] >= min_size:
          #         seg_map[output == i + 1] = 255

          # seg_map_cv2 = np.array(seg_map).astype(np.uint8)          
          # seg_map = Image.fromarray(seg_map)
          # seg_map = (seg_map.filter(ImageFilter.MinFilter(3)))


          seg_map_cv2 = np.array(seg_map).astype(np.float32)



          # seg_map_cv2 = cv2.dilate(seg_map_cv2, kernel, iterations=1)

          ret_frame = drawSegment(orignal_im.resize(save_size), seg_map, count, shape)
          ret_frame = np.array(ret_frame) 
          ret_frame = ret_frame[:, :, ::-1].copy()
          ret_frame = cv2.imencode('.jpg', ret_frame)[1].tobytes() 
          frame = ret_frame

          # print(count)
          count = count + 1

          end  = time.time()
          
          if count % offset != 0 and flag == 1:
            print(1 / (end - start), end = "\r")

          yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    modelType = "mobile_net_model"
    if len(sys.argv) > 3 and sys.argv[3] == "1":
      modelType = "xception_model"

    MODEL = DeepLabModel(modelType)
    # model_file = "/home/prince/Downloads/deeplabv3_257_mv_gpu.tflite"
    # interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    # interpreter.allocate_tensors()
    

    app.run(host='0.0.0.0', debug=True)
