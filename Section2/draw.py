import cv2

image = cv2.imread("image_01.png")

output = image.copy()

#cv2.rectangle(image, top left corner, bottom right corner, color, thickness)
cv2.rectangle(output, (320, 30), (480, 180), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)
