# IMAGE_NAME=x11vnc/docker-desktop:zh_CN
# NOVNC_PORT=6080

# Note: the manual built image has a deficiency: there is something unwanted in the /tmp
# namely: /tmp/.X0-lock
# the socket file /tmp/.X11-unix/X0 is not in the image.
IMAGE_NAME=cybergod_worker_gui:remote-base
NOVNC_PORT=6081

echo "Using docker image: $IMAGE_NAME"
echo "noVNC port in the container: $NOVNC_PORT"

echo "noVNC Proxy WebUI will be hosted at: http://localhost:16080/vnc.html?resize=downscale&autoconnect=1&password=secret (on host machine)"


# the python script is different from bash.
# it takes resolution from command line argument "--resolution" instead of environment variable "RESOLUT"
docker run --rm --tty -e VNCPASS=secret -p 16080:$NOVNC_PORT -v novnc_test:/home/ubuntu/project -v ./startvnc.py:/usr/local/bin/startvnc.py:ro --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --entrypoint /usr/bin/env $IMAGE_NAME python3 /usr/local/bin/startvnc.py --resolution 800x600