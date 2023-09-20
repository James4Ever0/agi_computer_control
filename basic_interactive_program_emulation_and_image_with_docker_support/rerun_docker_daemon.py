# TODO: eliminate stale containers by restarting docker every 10 sessions.
MACOS_DOCKER_APP_BINARY = "/Applications/Docker.app/Contents/MacOS/Docker"
# killall Docker && open -j -a Docker
# ps aux | grep Docker.app | grep -v grep | awk '{print $2}' | xargs -Iabc kill abc
HIDE_DOCKER_ASCRIPT = """
tell application "System Events"
    set visible of processes where name is "Docker Desktop" to false
end tell
"""
WINDOW_TITLE_KW = "Docker Desktop"
# killall docker
MACOS_KILL_DOCKER_APP = """ bash -c 'ps aux | grep Docker.app | grep -v grep | awk "{print \\$2}" | xargs -I abc kill abc' """
import subprocess

LINUX_CONTROL_DOCKER_SERVICE_CMDGEN = lambda action: f"sudo systemctl {action} docker"
LINUX_RESTART_DOCKER_COMMAND = LINUX_CONTROL_DOCKER_SERVICE_CMDGEN("restart")
LINUX_STOP_DOCKER_COMMAND = LINUX_CONTROL_DOCKER_SERVICE_CMDGEN("stop")
LINUX_START_DOCKER_COMMAND = LINUX_CONTROL_DOCKER_SERVICE_CMDGEN("start")

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

kill_safe_codes = [0]
start_safe_codes = [0]
sysname = platform.system()

REQUIRED_BINARIES = ["docker"]
elevate_needed = False

DOCKER_DESKTOP_EXE = "Docker Desktop.exe"

from typing import List


def execute_os_command_and_assert_safe_exit(cmd: str, safe_codes: List[int] = [0]):
    ret = os.system(cmd)  # use cmd.exe on windows.
    assert (
        ret in safe_codes
    ), f"Abnormal exit code {ret} while executing following command:\n{cmd}"


kill_docker_cmds = []
start_docker_cmds = []
if sysname == "Windows":
    import pygetwindow

    REQUIRED_BINARIES.append("taskkill")

    docker_path = shutil.which("docker")
    docker_bin_path = os.path.dirname(docker_path)
    docker_desktop_dir_path = os.path.split(os.path.split(docker_bin_path)[0])[0]
    docker_desktop_exe_path = os.path.join(docker_desktop_dir_path, DOCKER_DESKTOP_EXE)

    assert os.path.exists(
        docker_desktop_exe_path
    ), f'Failed to find docker desktop executable at: "{docker_desktop_exe_path}"'

    kill_docker_cmds.append(WINDOWS_KILL_DOCKER_COMMAND)
    start_docker_cmds.append(f'start "" "{docker_desktop_exe_path}"')  # bloody chatgpt.

    def hide_docker():
        ...

elif sysname == "Linux":
    REQUIRED_BINARIES.append("systemctl")
    elevate_needed = True

    kill_docker_cmds.append(LINUX_STOP_DOCKER_COMMAND)
    start_docker_cmds.append(LINUX_START_DOCKER_COMMAND)

    def hide_docker():
        ...

elif sysname == "Darwin":
    # import pygetwindow
    import applescript
    HIDE_DOCKER_ASCRIPT_OBJ = applescript.AppleScript(HIDE_DOCKER_ASCRIPT)
    kill_safe_codes.append(256)
    REQUIRED_BINARIES.extend(["killall", "open", MACOS_DOCKER_APP_BINARY])

    kill_docker_cmds.extend(["killall Docker", "killall docker", MACOS_KILL_DOCKER_APP])
    # start_docker_cmds.append(MACOS_DOCKER_APP_BINARY)
    start_docker_cmds.append(f"open -j -a {MACOS_DOCKER_APP_BINARY}")

    def hide_docker():
        HIDE_DOCKER_ASCRIPT_OBJ.run()
        # pygetwindow.getWindowsWithTitle(WINDOW_TITLE_KW)

else:
    raise Exception(f"Unknown platform: {sysname}")


def kill_docker():
    for cmd in kill_docker_cmds:
        execute_os_command_and_assert_safe_exit(cmd, kill_safe_codes)


def start_docker():
    for cmd in start_docker_cmds:
        execute_os_command_and_assert_safe_exit(cmd, start_safe_codes)


DOCKER_KILLED_KWS = [
    "the docker daemon is not running",
    "Cannot connect to the Docker daemon",
    "error during connect",
]


def verify_docker_killed(timeout=5, encoding="utf-8", inverse: bool = False):
    output = (
        subprocess.Popen(
            ["docker", "ps"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        .communicate(timeout=timeout)[1]
        .decode(encoding)
    )
    killed = any([kw in output for kw in DOCKER_KILLED_KWS])
    if not inverse:
        if not killed:
            raise Exception(
                f"Docker not killed.\nCaptured output from command `docker ps`:\n{output}"
            )
    else:
        if killed:
            raise Exception(
                f"Docker not started.\nCaptured output from command `docker ps`:\n{output}"
            )


import time


def verify_docker_launched(retries=7, sleep=3):
    success = False
    for i in range(retries):
        try:
            verify_docker_killed(inverse=True)
            success = True
            break
        except Exception as e:
            if i < retries - 1:
                print(f"Retrying in {sleep} seconds...")
                time.sleep(sleep)
            else:
                raise e
    return success


def restart_docker():
    check_required_binaries()
    print("prerequisites checked")
    kill_docker()
    print("docker killed")
    verify_docker_killed()
    print("kill has been verified")
    start_docker()
    print("docker restarted")


import shutil


def check_required_binaries():
    for name in REQUIRED_BINARIES:
        resolved_path = shutil.which(name)
        assert resolved_path, f"{name} is not available in PATH."
        assert os.path.exists(
            resolved_path
        ), f"{name} does not exist.\nfilepath: {resolved_path}"
        print(f"'{name}' found")


# working!
def restart_and_verify():
    restart_docker()
    verify_docker_launched()
    print("docker restart verified")
    hide_docker()
    print("docker window minimized")


if elevate_needed:
    elevate.elevate(graphical=False)

if __name__ == "__main__":
    # kill & perform checks if you really have killed docker.
    # restart & check if restart is successful.
    # do it once more.
    for i in range(2):
        print(f"trial #{i}")
        restart_and_verify()
        time.sleep(3)  # m1 is running too damn fast. or is it?
