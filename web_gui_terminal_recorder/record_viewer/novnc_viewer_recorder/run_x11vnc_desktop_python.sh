IMAGE_NAME=x11vnc/docker-desktop:zh_CN

echo "NoVNC Proxy WebUI will be hosted at: http://localhost:16080 (on host machine)"

docker run --rm --tty -e RESOLUT=800x600 -e VNCPASS=secret -p 16080:6080 -v novnc_test:/home/ubuntu/project -v ./startvnc.py:/usr/local/bin/startvnc.py:ro --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --entrypoint /usr/bin/env $IMAGE_NAME bash /usr/local/bin/startvnc.py