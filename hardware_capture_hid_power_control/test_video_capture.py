# lsusb -v
# Actions Microelectronics Co. Display capture-UVC05

# ls /dev/video*
# ls /sys/class/video4linux
# v4l2-ctl --list-devices

# get resolution/modes
# v4l2-ctl --list-formats-ext

# video2&3 are capture card.
import cv2

camera_id = 2
cap = cv2.VideoCapture(camera_id)


# showing values of the properties
print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CV_CAP_PROP_FRAME_FPS : '{}'".format(cap.get(cv2.CAP_PROP_FPS)))
print("CV_CAP_PROP_FRAME_FOURCC : '{}'".format(cap.get(cv2.CAP_PROP_FOURCC)))
print("CV_CAP_PROP_FRAME_FORMAT : '{}'".format(cap.get(cv2.CAP_PROP_FORMAT)))
# cv2.CAP_PROP_SETTINGS

ret, frame = cap.read()

# Display the resulting frame
cv2.imwrite("output.png",)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)