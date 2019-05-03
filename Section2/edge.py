import cv2

image = cv2.imread("coins.png")
cv2.imshow("Original", image)

edged = cv2.Canny(image, 200, 250) #cv2.Canny(input, lower threshold, uppoer threshold)
cv2.imshow("Edged", edged)

cv2.waitKey(0)