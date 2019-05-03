import numpy as np
import cv2

#load input
image = cv2.imread("Lenna.png")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

#load model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")


net.setInput(blob)
detections = net.forward()

probability_trshold = 0.5 
# loop over the detections
for i in range(0, detections.shape[2]):
	# get the probability 
	probability = detections[0, 0, i, 2]

	# thresholding
	if probability > probability_trshold:
		# get the location
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# display the bounding box
		text = "{:.2f}%".format(probability * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)