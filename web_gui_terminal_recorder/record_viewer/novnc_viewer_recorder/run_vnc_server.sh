IMAGE_NAME=x11vnc/docker-desktop:zh_CN

docker run --rm --tty -e RESOLUT=800x600 -e VNCPASS=secret -p 15900:5900 -v novnc_test:/home/ubuntu/project --security-opt seccomp=unconfined --cap-add=SYS_PTRACE $IMAGE_NAME startvnc.sh