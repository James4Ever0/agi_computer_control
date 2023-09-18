MACOS_DOCKER_APP_BINARY = "/Applications/Docker.app/Contents/MacOS/Docker"
# killall Docker && open --background -a Docker

LINUX_RESTART_DOCKER_COMMAND = "sudo systemctl restart docker"

# DOES NOT WORK ON WIN11
# kill com.docker.backend.exe? seems to be hanging
WINDOWS_RESTART_DOCKER_COMMAND = 'powershell -Command "Restart-Service -Name *docker*"' # need elevated permissions
# Stop-Service & Start-Service
# net stop com.docker.service/docker & net start com.docker.service/docker
import platform

sysname = platform.system()