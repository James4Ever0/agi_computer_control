import subprocess

# run this using xvfb.

"xvfb-run --server-num=10 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority vboxsdl --startvm "Ubuntu 16.04""
