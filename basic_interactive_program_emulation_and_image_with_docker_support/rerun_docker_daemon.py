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
import elevate

sysname = platform.system()

REQUIRED_BINARIES = ['docker']
elevate_needed = False

if sysname == 'Windows':
    REQUIRED_BINARIES.append('taskkill')
elif sysname == 'Linux':
    REQUIRED_BINARIES.append('systemctl')
    elevate_needed = True
elif sysname == 'Darwin':
    REQUIRED_BINARIES.append('killall')
    REQUIRED_BINARIES.append('open')
    REQUIRED_BINARIES.append(MACOS_DOCKER_APP_BINARY)
else:
    raise Exception('Unknown platform: {}'.format(sysname))

import shutil

for name in REQUIRED_BINARIES:
    assert shutil.which(name), f"{name} is not available in PATH."