# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2 --messaging 1 --exp_num whatever#youwantthatmatchesclient

# import the necessary packages
from imutils import build_montages
from imutils.video import FPS
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import csv 


#-- load face detection files
face_cascade_name = 'face_detection/data/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


blur_threshold=100
light_lower = 100
light_upper = 150


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-mW", "--montageW", required=True, type=int,
	help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int,
	help="montage frame height")
ap.add_argument("-ms", "--messaging", default=0, type=int,
	help="type of messaging (default is 0: REQ/REP; 1 is PUB/SUB, 2 is REQ/REP + PUB/SUB)")
ap.add_argument("-bl", "--bl_li", default=1, type=int,
	help="run blur and lighting checks on server")
ap.add_argument("-ex", "--exp_num", required=True, 
	help="for naming files")

args = vars(ap.parse_args())

# initialize the ImageHub object
if args["messaging"] == 0:
	imageHub = imagezmq.ImageHub()
elif args["messaging"] == 1:
	# imageHub = imagezmq.ImageHub(open_port='tcp://localhost:5556', REQ_REP = False)
	imageHub = imagezmq.ImageHub(open_port='tcp://192.168.0.145:5588', REQ_REP = False) #Ryan's laptop
elif args["messaging"] == 2:
	imageHub_rr = imagezmq.ImageHub()
else:
	raise ValueError("messaging input value must be 0, 1, or 2")

count=0
time_data = ['']*1000

# start looping over all the frames
while True:
	# receive client name and frame from the client and acknowledge
	# the receipt
	if args["messaging"] == 2 and count == 0:
		(clientIP, frame) = imageHub_rr.recv_image()
		fps = FPS().start()
		imageHub_rr.send_reply(b'OK')
		imageHub = imagezmq.ImageHub(open_port='tcp://{}:5566'.format(clientIP), REQ_REP = False)

	else:
		(clientIP, frame) = imageHub.recv_image()

		if count == 0:
			fps = FPS().start()
		if args["messaging"] == 0:
			imageHub.send_reply(b'OK')
	
	if args["bl_li"] == 1:

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame_gray = cv2.equalizeHist(frame_gray)

		fm = variance_of_laplacian(frame_gray)
		blur_text = "Not Blurry"

		passes = True

		if fm < blur_threshold:
			blur_text = "Blurry"
			passes = False

		light_text = 'Good Lighting '
		mean, std = cv2.meanStdDev(frame_gray)
		mean = mean[0][0]
		std = std[0][0]
		if mean > light_upper:
			light_text = "Too Bright"
			passes = False
		elif mean < light_lower:
			light_text = 'Too Dark'
			passes = False

	frame2 = frame.copy()
	frame2 = cv2.putText(frame2, "{}".format(blur_text), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	frame2 = cv2.putText(frame2, "{}".format(light_text), (10, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	if passes:
		#-- Detect faces
		faces = face_cascade.detectMultiScale(frame_gray)
		for (x,y,w,h) in faces:
			center = (x + w//2, y + h//2)
			frame2 = cv2.ellipse(frame2, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) # this line updates the frame with the drawing
			faceROI = frame_gray[y:y+h,x:x+w]
		#-- end face detection

	# # display image with face detections and blur/lighting status
	# cv2.imshow('frame',frame2)
	# k = cv2.waitKey(30) & 0xff
	# if k == 27:
	# 	break

	if count<99:
		fps.update()

	if count<=999:
		dateTimeObj = datetime.now()
		time_data[count] = [str(count), str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + 
							':' + str(dateTimeObj.second) + '.' + str(dateTimeObj.microsecond), frame]

	if count==99:
		fps.stop()
		print("[INFO] elasped time to receive 100 frames: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
	if count == 999:
		filename = 'server_time_data_'+args['exp_num']+'.csv'
		with open(filename, 'w') as csvfile:  
			# creating a csv writer object  
			csvwriter = csv.writer(csvfile)  
				
			# writing the headers
			csvwriter.writerow(['count', 'time', 'frame']) 

			# writing the data rows  
			csvwriter.writerows(time_data) 
		print('Wrote to csv')


	count +=1 

	
# do a bit of cleanup
cv2.destroyAllWindows()