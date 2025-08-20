echo "Container using host network"
echo "noVNC proxy will be hosted at: http://localhost:6080/vnc.html"

docker run --rm --name novnc -e AUTOCONNECT=true -e VNC_PASSWORD=secret -e VNC_SERVER=127.0.0.1:15900 --network host bonigarcia/novnc:1.2.0

# try running with root if the network is not host
# or just create a dedicated network for the interconnected container groups
# assign each container with a name, then connect with the name as hostname