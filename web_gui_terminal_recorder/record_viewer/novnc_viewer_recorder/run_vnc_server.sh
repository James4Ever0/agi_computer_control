IMAGE_NAME=x11vnc/docker-desktop:zh_CN

echo "VNC server will be hosted at: http://localhost:15900 (on host machine)"

docker run --rm --tty -e RESOLUT=800x600 -e VNCPASS=secret -p 15900:5900 -v novnc_test:/home/ubuntu/project --security-opt seccomp=unconfined --cap-add=SYS_PTRACE $IMAGE_NAME startvnc.sh