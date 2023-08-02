v4l2-ctl -d 2 --set-fmt-video=width=1920,height=1080,pixelformat=MJPG

# get fps
# v4l2-ctl -P -d 2

# set fps
# v4l2-ctl -p 60 -d 2

# get all help
# v4l2-ctl --help-all