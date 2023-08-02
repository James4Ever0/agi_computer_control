# lsusb -v
# Actions Microelectronics Co. Display capture-UVC05

# ls /dev/video*
# ls /sys/class/video4linux
# v4l2-ctl --list-devices

# video2&3 are capture card.
import cv2

cap = cv2.VideoCapture(2)


ret, frame = cap.read()

# Display the resulting frame
cv2.imshow('frame', frame)
cv2.waitKey(0)