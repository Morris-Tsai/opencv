import cv2
import numpy as np

img1 = cv2.imread('opening.png',0)
img2 = cv2.imread('closing.png',0)

kernel = np.ones((5,5),np.uint8)

opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
cv2.imshow("img1", img1)
cv2.imshow("Opening", opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
cv2.imshow("img2", img2)
cv2.imshow("Closing", closing)
cv2.waitKey(0)

