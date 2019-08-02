import numpy as np
import imutils
from imutils.video import VideoStream
import cv2 as cv
import sys
import time

if len(sys.argv) != 2:
    print('Usage:',sys.argv[0],'<image name>')
    sys.exit(2)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs1 = VideoStream(src=0).start()
vs2 = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
    
while True:
    # grab the frame from the threaded video stream
    frame1 = vs1.read()

    # Display the resulting frame
    cv.imshow('frame1',frame1)
    
    # grab the frame from the threaded video stream
    frame2 = vs2.read()

    # Display the resulting frame
    cv.imshow('frame2',frame2)
    
    key = cv.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('p'):
        cv.imwrite(sys.argv[1],frame2)
        break

# do a bit of cleanup
vs1.stop()
vs2.stop()

# When everything done, release the capture
cv.destroyAllWindows()
