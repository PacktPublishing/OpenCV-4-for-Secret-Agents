# import the necessary packages
from imutils.video import FileVideoStream

import numpy as np
import imutils
import cv2

#We're using SSD with MobileNet for classification. These are the 
#classes of MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

#random color choices for each class so its easy to distinguish
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load pre-trained model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# setup video stream from file
vs = FileVideoStream("video_01.mp4").start()

probability_treshold = 0.76
# loop over the frames from the video stream
while True:
	# read from stream
	frame = vs.read()

	# convert to blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# get the probability
		probability = detections[0, 0, i, 2]

		# thresholding
		if probability > probability_treshold:
			# get the index of the detected class
			idx = int(detections[0, 0, i, 1])
			# get the location of the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx],
				probability * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			
			cv2.putText(frame, label, (startX, startY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break






# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()