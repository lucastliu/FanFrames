# USAGE
# python client.py --server-ip SERVER_IP

# import the necessary packages
from imutils.video import VideoStream, WebcamVideoStream, FPS 
import imagezmq
import argparse
import socket
import time
import cv2
import numpy as np 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
ap.add_argument("-ms", "--messaging", default=0, type=int,
	help="type of messaging (default is 0: REQ/REP; 1 is PUB/SUB, 2 is REQ/REP + PUB/SUB)")
ap.add_argument("-t", "--multithread", default=1, type=int,
	help="single/multithreading (default is 1: multithreaded; 0 single threaded)")

args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
# initialize the ImageHub object
if args["messaging"] == 0:
	sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
		args["server_ip"]))
elif args["messaging"] == 1:
	sender = imagezmq.ImageSender(connect_to="tcp://*:5588", REQ_REP = False)
elif args["messaging"] == 2:
	sender = imagezmq.ImageSender(connect_to="tcp://*:5566", REQ_REP = False)
	rr_sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
		args["server_ip"]))
else:
	raise ValueError("messaging input value must be 0, 1, or 2")


# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
clientName = socket.gethostname()
clientIP = socket.gethostbyname(clientName)

if args["multithread"] == 1:
	vs = WebcamVideoStream(src=0).start()
elif args["multithread"] == 0:
	vs = VideoStream(src=0).start()
else:
	raise ValueError("multithread input value must be 0 or 1")

time.sleep(2.0)

count=0
fps = FPS().start()
prev_frame = None
 
while True:
	# read the frame from the camera and send it to the server
	frame = vs.read()
	if prev_frame is None:
		prev_frame = np.zeros(frame.shape)
	comp = prev_frame == frame
	if comp.all():
		# print("matched")
		continue
	else:
		prev_frame = frame 
	if count == 0 and args["messaging"] ==2:
		rr_sender.send_image(clientIP, frame)
	# print("about to send")
	sender.send_image(clientIP, frame)
	# print('sent')
	if count<100:
		count +=1 
		fps.update()
	if count==100:
		fps.stop()
		print("[INFO] elasped time to send 100 frames: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		count +=1
	