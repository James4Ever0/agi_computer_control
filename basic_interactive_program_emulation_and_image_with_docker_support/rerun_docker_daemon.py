MACOS_DOCKER_APP_BINARY = "/Applications/Docker.app/Contents/MacOS/Docker"
# killall Docker && open --background -a Docker

LINUX_RESTART_DOCKER_COMMAND = "sudo systemctl restart docker"

# DOES NOT WORK ON WIN11
# kill com.docker.backend.exe? seems to be hanging
WINDOWS_KILL_DOCKER_COMMAND = 'taskkill /FI "IMAGENAME eq Docker*" /F'
# start program minimized? instead use pygetwindow to hide the window once found.
# find 'Docker Desktop.exe'
# which docker -> ../../ -> 'Docker Desktop.exe'
# WINDOWS_RESTART_DOCKER_COMMAND = 'powershell -Command "Restart-Service -Name *docker*"' # need elevated permissions
# Stop-Service & Start-Service
# net stop com.docker.service/docker & net start com.docker.service/docker
import platform

sysname = platform.system()