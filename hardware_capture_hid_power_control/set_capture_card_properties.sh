# maybe you shall install python bindings
# pip3 install v4l2-python3
# pip3 install v4l2py

# fuck windows. how can i change capture device specs without using obs?
# maybe you should spin up obs for cross-platform integration.

# for macos (maybe?): https://github.com/jtfrey/uvc-util

v4l2-ctl -d 2 --set-fmt-video=width=1920,height=1080,pixelformat=MJPG

# get fps
# v4l2-ctl -P -d 2

# set fps
# v4l2-ctl -p 60 -d 2

# get all help
# v4l2-ctl --help-all