# lsusb -v
# Actions Microelectronics Co. Display capture-UVC05

# ls /dev/video*
# ls /sys/class/video4linux
# v4l2-ctl --list-devices

# get resolution/modes
# v4l2-ctl --list-formats-ext

# video2&3 are capture card.
import cv2

# can't you reset?
camera_id = 2 # use first device 
# CV_CAP_PROP_FRAME_WIDTH: '640.0'
# CV_CAP_PROP_FRAME_HEIGHT : '480.0'
# CV_CAP_PROP_FRAME_FPS : '30.0'
# CV_CAP_PROP_FRAME_FOURCC : '1196444237.0'
# CV_CAP_PROP_FRAME_FORMAT : '16.0'

# camera_id = 0
# cam 1,3 not working.

cap = cv2.VideoCapture(camera_id)

# resolution not right...

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# cannot strentch!
# CV_CAP_PROP_FRAME_WIDTH: '1280.0'
# CV_CAP_PROP_FRAME_HEIGHT : '1024.0'

# showing values of the properties
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CV_CAP_PROP_FRAME_FPS : '{}'".format(cap.get(cv2.CAP_PROP_FPS)))
print("CV_CAP_PROP_FRAME_FOURCC : '{}'".format(cap.get(cv2.CAP_PROP_FOURCC)))
print("CV_CAP_PROP_FRAME_FORMAT : '{}'".format(cap.get(cv2.CAP_PROP_FORMAT)))
# cv2.CAP_PROP_SETTINGS

ret, frame = cap.read()

# Display the resulting frame
cv2.imwrite(output_path:="output.png",frame) # from "no_signal" to something!
import os
os.system(f"ffplay {output_path}")
# cv2.imshow('frame', frame)
# cv2.waitKey(0)