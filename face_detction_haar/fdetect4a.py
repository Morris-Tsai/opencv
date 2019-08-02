import cv2
import time
import sys
import numpy as np
from picamera import PiCamera
import imutils
from imutils.video import VideoStream
from gpiozero import LED

# LED on BCM17, set off by default
red = LED(17)  

img_name='/home/pi/image.jpg'

# initialize the video stream, allow the cammera sensor to warmup,
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# wait for press key
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        img = None
        break

    if key & 0xFF == ord('p'):
        img = frame
        break

# stop video stream
vs.stop()
cv2.destroyAllWindows()

if img is None:  # press 'q' to quit
    sys.exit(2)

# convert the image to gray  
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml')

# let's detect multiscale (some images may be closer to camera than others) images 
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  

# print the number of faces found
print('Faces found: ', len(faces))

if len(faces):
    print('LDE is ON...')
    red.on()

# list of faces and draw them as rectangles on original colored
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

# Turn off the LED
red.off()
