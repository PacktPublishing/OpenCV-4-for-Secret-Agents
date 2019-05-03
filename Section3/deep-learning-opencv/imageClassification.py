import numpy as np
import cv2
import time

# load the input image from disk
image = cv2.imread("images/image_02.jpg")

# load the class labels from disk
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]


blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

net.setInput(blob)
preds = net.forward()

#sort the predictions in order of probability and grab the first 3.
idxs = np.argsort(preds[0])[::-1][:3]

for (i, idx) in enumerate(idxs):
	# draw the top prediction on the input image
	if i == 0: #the first in the list will be the prediction with most probability
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

	# display 
	print("Predicted label {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))

cv2.imshow("Image",image)
cv2.waitKey(0)