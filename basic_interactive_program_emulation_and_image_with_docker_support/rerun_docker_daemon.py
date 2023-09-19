MACOS_DOCKER_APP_BINARY = "/Applications/Docker.app/Contents/MacOS/Docker"
# killall Docker && open --background -a Docker
# ps aux | grep Docker.app | grep -v grep | awk '{print $2}' | xargs -Iabc kill abc
# killall docker
import subprocess

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
import shutil
import os

sysname = platform.system()

REQUIRED_BINARIES = ['docker']
elevate_needed = False

DOCKER_DESKTOP_EXE = 'Docker Desktop.exe'

def execute_os_command_and_assert_safe_exit(cmd:str):
    ret = os.system(cmd)
    assert ret == 0, f"Abnormal exit code {ret} while executing following command:\n{cmd}"

if sysname == 'Windows':
    REQUIRED_BINARIES.append('taskkill')
    docker_path = shutil.which('docker')
    docker_bin_path = os.path.dirname(docker_path)
    docker_desktop_dir_path = os.path.split(os.path.split(docker_bin_path)[0])[0]
    docker_desktop_exe_path = os.path.join(docker_desktop_dir_path, DOCKER_DESKTOP_EXE)
    assert os.path.exists(docker_desktop_exe_path), f'Failed to find docker desktop executable at: "{docker_desktop_exe_path}"'
    def kill_docker():
        execute_os_command_and_assert_safe_exit(WINDOWS_KILL_DOCKER_COMMAND)
    def start_docker():
        execute_os_command_and_assert_safe_exit(f'start "{docker_desktop_exe_path}"')
elif sysname == 'Linux':
    REQUIRED_BINARIES.append('systemctl')
    elevate_needed = True
    def kill_docker():
        ...
    def start_docker():
        ...
elif sysname == 'Darwin':
    REQUIRED_BINARIES.append('killall')
    REQUIRED_BINARIES.append('open')
    REQUIRED_BINARIES.append(MACOS_DOCKER_APP_BINARY)
    def kill_docker():
        ...
    def start_docker():
        ...
else:
    raise Exception(f'Unknown platform: {sysname}')


DOCKER_KILLED_KW = 'Cannot connect to the Docker daemon'
def verify_docker_killed(timeout = 5, encoding='utf-8'):
    output = subprocess.Popen(['docker', 'ps'], stdout=subprocess.PIPE).communicate(timeout=timeout)[0].decode(encoding)
    killed = DOCKER_KILLED_KW in output
    if not killed:
        raise Exception(f'Docker not killed.\nCaptured output from command `docker ps`:\n{output}')

def restart_docker():
    kill_docker()
    verify_docker_killed()
    start_docker()

import shutil

for name in REQUIRED_BINARIES:
    assert shutil.which(name), f"{name} is not available in PATH."


if __name__ == '__main__':
    # kill & perform checks if you really have killed docker.
    # restart & check if restart is successful.
    # do it once more.
    if elevate_needed:
        elevate.elevate(graphical=False)
    