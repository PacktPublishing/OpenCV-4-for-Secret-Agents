import cv2

image = cv2.imread("image_01.png")

cv2.imshow("Image",image)

(h,w,d) = image.shape
print("h = {},w = {}, d={}".format(h,w,d))


(B,G,R) = image[200,200]
print("B = {}, G = {}, R={}".format(B,G,R))

cv2.waitKey(0)

