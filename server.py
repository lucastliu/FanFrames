# USAGE
# python server.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --montageW 2 --montageH 2

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

args = vars(ap.parse_args())

# initialize the ImageHub object
if args["messaging"] == 0:
	imageHub = imagezmq.ImageHub()
elif args["messaging"] == 1:
	# imageHub = imagezmq.ImageHub(open_port='tcp://localhost:5556', REQ_REP = False)
	imageHub = imagezmq.ImageHub(open_port='tcp://192.168.0.140:5588', REQ_REP = False) #Ryan's laptop
elif args["messaging"] == 2:
	imageHub_rr = imagezmq.ImageHub()
else:
	raise ValueError("messaging input value must be 0, 1, or 2")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
CONSIDER = set(["dog", "person", "car"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# assign montage width and height so we can view all incoming frames
# in a single "dashboard"
mW = args["montageW"]
mH = args["montageH"]
print("[INFO] detecting: {}...".format(", ".join(obj for obj in
	CONSIDER)))



count=0
time_data = ['']*100

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
		# print("ready to receive")
		(clientIP, frame) = imageHub.recv_image()
		# print("received image")
		if count == 0:
			fps = FPS().start()
		if args["messaging"] == 0:
			imageHub.send_reply(b'OK')
	
	if args["bl_li"] == 1:

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame_gray = cv2.equalizeHist(frame_gray)
		#-- Detect faces
		faces = face_cascade.detectMultiScale(frame_gray)
		for (x,y,w,h) in faces:
			center = (x + w//2, y + h//2)
			frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) # this line actually modifies the frame with the drawing
			faceROI = frame_gray[y:y+h,x:x+w]
		#-- end face detection

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



	if count<99:
		count +=1 
		fps.update()
	if count==99:
		fps.stop()
		print("[INFO] elasped time to receive 100 frames: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		count +=1
		filename = 'server_time_data.csv'
		with open(filename, 'w') as csvfile:  
			# creating a csv writer object  
			csvwriter = csv.writer(csvfile)  
				
			# writing the headers
			csvwriter.writerow(['count', 'time', 'frame']) 

			# writing the data rows  
			csvwriter.writerows(time_data) 

	# if a device is not in the last active dictionary then it means
	# that its a newly connected device
	if clientIP not in lastActive.keys():
		print("[INFO] receiving data from {}...".format(clientIP))

	# record the last active time for the device from which we just
	# received a frame
	lastActive[clientIP] = datetime.now()

	# resize the frame to have a maximum width of 400 pixels, then
	# grab the frame dimensions and construct a blob
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# # reset the object count for each object in the CONSIDER set
	# objCount = {obj: 0 for obj in CONSIDER}

	# # loop over the detections
	# for i in np.arange(0, detections.shape[2]):
	# 	# extract the confidence (i.e., probability) associated with
	# 	# the prediction
	# 	confidence = detections[0, 0, i, 2]

	# 	# filter out weak detections by ensuring the confidence is
	# 	# greater than the minimum confidence
	# 	if confidence > args["confidence"]:
	# 		# extract the index of the class label from the
	# 		# detections
	# 		idx = int(detections[0, 0, i, 1])

	# 		# check to see if the predicted class is in the set of
	# 		# classes that need to be considered
	# 		if CLASSES[idx] in CONSIDER:
	# 			# increment the count of the particular object
	# 			# detected in the frame
	# 			objCount[CLASSES[idx]] += 1

	# # 			# compute the (x, y)-coordinates of the bounding box
	# # 			# for the object
	# # 			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	# # 			(startX, startY, endX, endY) = box.astype("int")

	# # 			# draw the bounding box around the detected object on
	# # 			# the frame
	# # 			cv2.rectangle(frame, (startX, startY), (endX, endY),
	# # 				(255, 0, 0), 2)

	# # # draw the sending device name on the frame
	# # cv2.putText(frame, clientIP, (10, 25),
	# # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# # # draw the object count on the frame
	# # label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
	# # 	objCount.items())
	# # cv2.putText(frame, label, (10, h - 20),
	# # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

	# # update the new frame in the frame dictionary
	# frameDict[clientIP] = frame

	# # build a montage using images in the frame dictionary
	# montages = build_montages(frameDict.values(), (w, h), (mW, mH))

	# # display the montage(s) on the screen
	# for (i, montage) in enumerate(montages):
	# 	cv2.imshow("Video received from client ({})".format(i),
	# 		montage)

	if count<=99:
		dateTimeObj = datetime.now()
		time_data[count] = [str(count), str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + 
							':' + str(dateTimeObj.second) + '.' + str(dateTimeObj.microsecond), frame]

	# # detect any kepresses
	# key = cv2.waitKey(1) & 0xFF

	# # if current time *minus* last time when the active device check
	# # was made is greater than the threshold set then do a check
	# if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
	# 	# loop over all previously active devices
	# 	for (clientIP, ts) in list(lastActive.items()):
	# 		# remove the client from the last active and frame
	# 		# dictionaries if the device hasn't been active recently
	# 		if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
	# 			print("[INFO] lost connection to {}".format(clientIP))
	# 			lastActive.pop(clientIP)
	# 			frameDict.pop(clientIP)

	# 	# set the last active check time as current time
	# 	lastActiveCheck = datetime.now()

	# # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break

# do a bit of cleanup
cv2.destroyAllWindows()