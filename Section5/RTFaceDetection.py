# import the necessary packages
from imutils.video import FileVideoStream
import numpy as np
import imutils
import cv2

#load pre trained model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# setup video stream from file
vs = FileVideoStream("Video Of People Walking.mp4").start()

probability_trshold = 0.4

# loop over the frames from the video stream
while True:
	# read the frame
	frame = vs.read()
	frame = cv2.resize(frame, (300,300))
	#  convert to blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# set the input
	net.setInput(blob)
	#get the detections
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
	# extract the probability 
		probability = detections[0, 0, i, 2]

		#thresholding
		if probability > probability_trshold:
			# get the location
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# display location and probability
			text = "{:.2f}%".format(probability * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, startY),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()