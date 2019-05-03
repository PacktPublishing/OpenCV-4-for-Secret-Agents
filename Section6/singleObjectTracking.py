#imports
from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2
import dlib


#We're using SSD with MobileNet for classification. These are the 
#classes of MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

#load pre-trained network
net = cv2.dnn.readNetFromCaffe("deploy.prototxt","mobilenet_iter_73000.caffemodel")

#Setup video stream from file
vs = FileVideoStream("Video05.mp4").start()

#create our tracker
#tracker = cv2.TrackerKCF_create()
tracker = dlib.correlation_tracker()

probability_treshold = 0.95
objectDetected = False;
# loop over the frames from the video stream
while True:
	# read from stream
	frame = vs.read()
	#resize preserving aspect ratio
	frame = imutils.resize(frame, width=600)
	frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

	if objectDetected is False:
		# convert to blob
		(height, width) = frame.shape[:2]
		#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		#	0.007843, (300, 300), 127.5)
		blob = cv2.dnn.blobFromImage(frame,
			0.007843, (width, height), 127.5)

		# pass the blob through the network and obtain the predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# get the probability
			probability = detections[0, 0, i, 2]

			# thresholding
			if probability > probability_treshold:
				objectDetected = True
				# get the index of the detected class
				idx = int(detections[0, 0, i, 1])
				# get the location of the object
				box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
				(startX, startY, endX, endY) = box.astype("int")

				#init tracker
				track_rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(frame_RGB, track_rect)

				# display the prediction
				label = "{}: {:.2f}%".format(CLASSES[idx],
					probability * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0,255,0), 2)
				
				cv2.putText(frame, label, (startX, startY),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

				break;

	else:
		#update tracker and get new position

		tracker.update(frame_RGB)
		new_pos = tracker.get_position()

		startX = int(new_pos.left())
		startY = int(new_pos.top())
		endX = int(new_pos.right())
		endY = int(new_pos.bottom())

		# draw the bounding box from the correlation object tracker
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv2.putText(frame, label, (startX, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break






# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


