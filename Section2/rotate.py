import cv2

image = cv2.imread("image_01.png")
(h,w,d) = image.shape

center = (w//2,h//2)
Rot_M = cv2.getRotationMatrix2D(center,-45, 1.0)
rotated = cv2.warpAffine(image,Rot_M,(w,h))
cv2.imshow("rotated",rotated)
cv2.waitKey(0)