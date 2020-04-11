# USAGE
# python client.py --server-ip SERVER_IP --messaging 1 --exp_num whatever#youwantthatmatchesserver

# import the necessary packages
from imutils.video import VideoStream, WebcamVideoStream, FPS 
import imagezmq
import argparse
import socket
import time
import cv2
import numpy as np 
import csv
from datetime import datetime   



def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

blur_threshold=100
light_lower = 100
light_upper = 150

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
ap.add_argument("-ms", "--messaging", default=0, type=int,
	help="type of messaging (default is 0: REQ/REP; 1 is PUB/SUB, 2 is REQ/REP + PUB/SUB)")
ap.add_argument("-t", "--multithread", default=1, type=int,
	help="single/multithreading (default is 1: multithreaded; 0 single threaded)")
ap.add_argument("-bl", "--bl_li", default=0, type=int,
	help="run blur and lighting checks on client")
ap.add_argument("-ex", "--exp_num", required=True,
	help="for naming files")

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
time_data = ['']*1000

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

	if args["bl_li"] == 1:
		
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		fm = variance_of_laplacian(frame_gray)
		blur_text = "Not Blurry"

		if fm < blur_threshold:
			blur_text = "Blurry"

		light_text = 'Good Lighting '
		mean, std = cv2.meanStdDev(frame_gray)
		mean = mean[0][0]
		std = std[0][0]
		if mean > light_upper:
			light_text = "Too Bright"
		elif mean < light_lower:
			light_text = 'Too Dark'


	if count<=999:
		dateTimeObj = datetime.now()
		time_data[count] = [str(count), str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + 
							':' + str(dateTimeObj.second) + '.' + str(dateTimeObj.microsecond), frame]

	if count == 0 and args["messaging"] ==2:
		rr_sender.send_image(clientIP, frame)

	sender.send_image(clientIP, frame)

	if count<99:
		fps.update()

	elif count==99:
		fps.stop()
		print("[INFO] elasped time to send 100 frames: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		
	elif count == 999:
		filename = 'client_time_data_'+args['exp_num']+'.csv'
		with open(filename, 'w') as csvfile:  
			# creating a csv writer object  
			csvwriter = csv.writer(csvfile)  
			# writing the headers
			csvwriter.writerow(['count', 'time', 'frame']) 	
			# writing the data rows  
			csvwriter.writerows(time_data) 
		print('Wrote to csv')

	count +=1 
	