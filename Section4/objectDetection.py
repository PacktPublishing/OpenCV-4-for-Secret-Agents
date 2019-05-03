import cv2
import numpy as np

#We're using SSD with MobileNet for classification. These are the 
#classes of MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

#read pre trained network
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

image = cv2.imread("images/image_04.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# set input
net.setInput(blob)

#pass through network to get predictions
detections = net.forward()

probability_threshold = 0.2
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	probability = detections[0, 0, i, 2]

	# thresholding
	if probability > probability_threshold:
		# get the index of the class label
		idx = int(detections[0, 0, i, 1])
		# get the location of the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], probability * 100)
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0,255,0), 2)
		cv2.putText(image, label, (startX, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)