import numpy as np
import cv2 as cv

# Load an color image in grayscale
img = cv.imread('mybaby.jpg',cv.IMREAD_GRAYSCALE)

rows,cols = img.shape

# cols-1 and rows-1 are the coordinate limits
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('Rotate image',dst)
cv.waitKey(0)
cv.destroyAllWindows()

dst = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imshow('Rotate 90',dst)
dst = cv.rotate(img, cv.ROTATE_180)
cv.imshow('Rotate 180',dst)
dst = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
cv.imshow('Rotate 270',dst)
cv.waitKey(0)
cv.destroyAllWindows()
