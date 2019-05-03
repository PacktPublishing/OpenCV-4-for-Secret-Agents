import cv2

image = cv2.imread("image_01.png")

#resize  cv2.resize(image, (width, height))
output = cv2.resize(image, (200,200))
cv2.imshow("resize",output)
cv2.waitKey(0)
