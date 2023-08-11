cd . # FIX: input/output error
ps aux | grep -i xvfb | grep 11 | grep -v grep | awk '{print $2}' | xargs -iabc kill -s TERM abc
xvfb-run --server-num=11 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority1 bash x11grab.sh
# xvfb-run --server-num=11 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority1 bash x11grab.sh | python3 stop_on_ffmpeg_x11_error.py