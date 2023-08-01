# vboxmanage startvm "12c0e77b-5f4a-4d30-b19b-1b105d2042cf" # Ubuntu 16.04
nohup xvfb-run --server-num=10 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority vboxsdl --startvm "Ubuntu 16.04" &
# start in headless mode
# you may not live stream this one! but vrde is available! and you can use ffmpeg for bridging!
# vboxmanage startvm "Ubuntu 16.04" --type headless