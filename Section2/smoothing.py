import cv2

image = cv2.imread("noisy.png")

cv2.imshow("Original", image) 

smooth = cv2.GaussianBlur(image, (11,11), 1) #cv2.GaussianBlur(input, kernel size, sigma)

cv2.imshow("De-noised", smooth)
cv2.waitKey(0)