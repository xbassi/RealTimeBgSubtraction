"""test_2_rpi_send_images.py -- send PiCamera image stream.

A Raspberry Pi test program that uses imagezmq to send image frames from the
PiCamera continuously to a receiving program on a Mac that will display the
images as a video stream.

This program requires that the image receiving program be running first. Brief
test instructions are in that program: test_2_mac_receive_images.py.
"""

# import imagezmq from parent directory
import sys
sys.path.insert(0, '../imagezmq')  # imagezmq.py is in ../imagezmq

import socket
import time
import cv2
# from imutils.video import VideoStream
import imagezmq

# use either of the formats below to specifiy address of display computer
# sender = imagezmq.ImageSender(connect_to='tcp://jeff-macbook:5555')
sender = imagezmq.ImageSender(connect_to='tcp://127.0.0.1:5555')

rpi_name = socket.gethostname()  # send RPi hostname with each image

cap = cv2.VideoCapture('s.mp4')

time.sleep(2.0)  # allow camera sensor to warm up
while cap.isOpened():  # send images as stream until Ctrl-C
	
	ret, image = cap.read()

	if image is None:
		break

	image = cv2.resize(image,(320,320))

	sender.send_image(rpi_name, image)



cap.release()
