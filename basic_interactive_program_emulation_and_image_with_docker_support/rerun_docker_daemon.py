MACOS_DOCKER_APP_BINARY = "/Applications/Docker.app/Contents/MacOS/Docker"
# killall Docker && open --background -a Docker

LINUX_RESTART_DOCKER_COMMAND = "sudo systemctl restart docker"

WINDOWS_RESTART_DOCKER_COMMAND = 'powershell -Command "Restart-Service -Name *docker*"'
# Stop-Service & Start-Service
# net stop com.docker.service/docker & net start com.docker.service/docker
import platform

sysname = platform.system()