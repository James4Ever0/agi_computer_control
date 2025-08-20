#!/usr/bin/env python3

# TODO: verify its effectiveness via bash run_x11vnc_desktop_python.sh
# TODO: make its launch behavior configurable, remote into other machines using tigervnc/ssh from the start, with some checks to ensure the connectivity exists first, and exit the entire script/container if the connection is closed
# TODO: use shlex to separate commands passed via environment variable

import os
import sys
import re
import subprocess
import signal
import atexit
import time
import random
import string
import tempfile
import argparse
from pathlib import Path
import shlex

# Global variables to track processes and state
processes: list[subprocess.Popen] = []
disp = None
vnc_port = None
web_port = None
local_ssh_agent = False
novnc_process = None
x11vnc_process = None
stop_file = None

print("Notice: You are using the Python 3 variant of startvnc.sh")


def cleanup():
    """Clean up function to kill child processes and remove lock files"""
    global local_ssh_agent

    # Kill locally started ssh-agent
    if local_ssh_agent and "SSH_AGENT_PID" in os.environ:
        try:
            subprocess.run(
                ["ssh-agent", "-k"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    # Remove X lock file
    if disp is not None:
        lock_file = Path(f"/tmp/.X{disp}-lock")
        if lock_file.exists():
            lock_file.unlink()

    # Terminate all child processes
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                time.sleep(0.5)
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass


def start_xorg(disp, resolution):
    """Start Xorg server with specified display and resolution"""
    print("Starting Xorg server...")
    print("Resolution:", resolution)
    log_dir = Path.home() / ".log"
    log_dir.mkdir(exist_ok=True, parents=True)

    config_file = Path.home() / f".config/xorg_X{disp}.conf"
    config_file.parent.mkdir(exist_ok=True, parents=True)

    # Copy base config and update resolution
    subprocess.run(["cp", "/etc/X11/xorg.conf", config_file], check=True)

    if resolution != "1920x1080":
        width, height = resolution.split("x")
        sed_cmd = f"s/Virtual 1920 1080/Virtual {width} {height}/"
        subprocess.run(["sed", "-i", "-e", sed_cmd, config_file], check=False)

    # Start Xorg
    log_file = log_dir / f"Xorg_X{disp}.log"
    err_log = log_dir / f"Xorg_X{disp}_err.log"

    with open(log_file, "w") as log, open(err_log, "w") as err:
        cmd = [
            "Xorg",
            "-noreset",
            "+extension",
            "GLX",
            "+extension",
            "RANDR",
            "+extension",
            "RENDER",
            "-logfile",
            str(log_file),
            "-config",
            str(config_file),
            f":{disp}",
        ]
        p = subprocess.Popen(cmd, stdout=log, stderr=err)
        processes.append(p)

    # Check if Xorg started successfully
    time.sleep(1)
    if p.poll() is not None:
        with open(err_log) as f:
            print(f.read(), file=sys.stderr)
        sys.exit(1)

    return p


def start_novnc(vnc_port, web_port, disp):
    """Start noVNC proxy"""
    log_file = Path.home() / f".log/novnc_X{disp}.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

    with open(log_file, "w") as log:
        cmd = [
            "/usr/local/noVNC/utils/launch.sh",
            "--web",
            "/usr/local/noVNC",
            "--vnc",
            f"localhost:{vnc_port}",
            "--listen",
            str(web_port),
        ]
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        processes.append(p)

    # Check if noVNC started
    time.sleep(0.5)
    if p.poll() is not None:
        with open(log_file) as f:
            print(f.read(), file=sys.stderr)
        sys.exit(1)

    return p


def start_x11vnc(disp, vnc_port, password):
    """Start x11vnc server with specified password"""
    log_file = Path.home() / f".log/x11vnc_X{disp}.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

    # Store password
    passwd_file = Path.home() / f".vnc/passwd{disp}"
    passwd_file.parent.mkdir(exist_ok=True, parents=True)

    subprocess.run(
        ["x11vnc", "-storepasswd", password, passwd_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Start x11vnc
    with open(log_file, "a") as log:
        cmd = [
            "x11vnc",
            "-display",
            f":{disp}",
            "-rfbport",
            str(vnc_port),
            "-xkb",
            "-repeat",
            "-skip_dups",
            "-forever",
            "-shared",
            "-rfbauth",
            str(passwd_file),
        ]
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        processes.append(p)

    return p


def start_application(
    command: str,
    log_file: Path,
    processes: list[subprocess.Popen],
    check_started_timeout=0.5,
):
    print("Starting application")
    print("App start command:", command)
    cmd_list = shlex.split(command)

    # Start app
    with open(log_file, "w") as log:
        p = subprocess.Popen(
            cmd_list,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(p)

    # Check app started
    time.sleep(check_started_timeout)
    if p.poll() is not None:
        with open(log_file) as f:
            print(f.read(), file=sys.stderr)
        sys.exit(1)


# lxterminal has toolbar on top. We don't want that.
# want something like xterm but with full unicode support
# alternatives: xterm(1), gnome-terminal(1), konsole(1), terminator(1), urxvt(1), qterminal(1)
def start_lxterminal(
    log_file: Path, processes: list[subprocess.Popen], init_command: str = "bash"
):
    # https://linuxcommandlibrary.com/man/lxterminal
    # window class: lxterminal
    # lxterminal configuration file in container: /home/ubuntu/.config/lxterminal/lxterminal.conf
    # TODO: make it full screen
    command: str = "lxterminal --command '%s'" % init_command
    start_application(command, log_file, processes)
    maximize_window(window_class="lxterminal", window_count=2)


def start_tigervnc_client(
    log_file: Path,
    processes: list[subprocess.Popen],
    vnc_password: str,
    vnc_host: str,
    vnc_port: str,
):
    # generate the password file
    passwd_file = "/tmp/vncpasswd"
    subprocess.run(
        ["bash", "-c", f"echo {vnc_password} | tightvncpasswd -f > {passwd_file}"]
    )
    command = f"vncviewer -passwd {passwd_file} -FullScreen {vnc_host}:{vnc_port}"
    start_application(command, log_file, processes)


def maximize_window(window_class: str, window_count: int):
    # kiosk mode
    # https://unix.stackexchange.com/questions/613782/running-x-appplication-without-desktop-environment-in-full-screen
    # https://unix.stackexchange.com/questions/237763/graphical-application-kiosk-mode-fullscreen
    while True:
        # find the window id
        command = f'xdotool search --class "{window_class}"'
        window_id = subprocess.run(
            command, shell=True, capture_output=True, text=True
        ).stdout.strip()
        if window_id:
            # count window id
            current_window_count = len(window_id.split())
            if current_window_count < window_count:  # wait for the window to appear
                time.sleep(0.1)
                continue
            else:
                break
        time.sleep(0.1)
    command = f'xdotool search --class "{window_class}" | xargs -Iabc xdotool windowsize abc 100% 100%'
    # print("Running window maximization command:", "sh -c '%s'" % command)
    subprocess.run(["sh", "-c", command])


def start_lxde(log_file: Path, processes: list[subprocess.Popen]):
    command: str = "lxsession -s LXDE -e LXDE"
    start_application(command, log_file, processes)


def start_ssh_agent():
    local_ssh_agent = False
    if "SSH_AUTH_SOCK" not in os.environ:
        local_ssh_agent = True
        output = subprocess.check_output(["ssh-agent", "-s"], text=True)
        for line in output.splitlines():
            match = re.match(r"(\w+)=([^;]+);", line)
            if match:
                os.environ[match.group(1)] = match.group(2)
    return local_ssh_agent


def main():
    global disp, vnc_port, web_port, local_ssh_agent, novnc_process, x11vnc_process, stop_file

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Start VNC server and desktop environment",
        epilog="Connects via VNC viewer at port 5900+DISPLAY or web browser at port 6080+DISPLAY",
    )
    parser.add_argument(
        "-s",
        "--resolution",
        default="1920x1080",
        help="Screen resolution (e.g., 1280x720)",
    )
    args = parser.parse_args()

    # Initialize environment
    subprocess.run(["/usr/local/bin/init_vnc"], check=True)
    subprocess.run(["sync"], check=True)

    # Unset desktop-related environment variables
    for var in list(os.environ.keys()):
        if re.match(
            r"^XDG|SESSION|^GTK|XKEYS|^WLS|WINDOWMANAGER|WAYLAND_DISPLAY|BROWSER", var
        ):
            os.environ.pop(var, None)

    # Find available display
    for i in range(10):
        lock_file = Path(f"/tmp/.X{i}-lock")
        socket_file = Path(f"/tmp/.X11-unix/X{i}")
        if not lock_file.exists() and not socket_file.exists():
            disp = i
            break

    if disp is None:
        print("Error: No available display port (0-9)", file=sys.stderr)
        sys.exit(1)

    # Set ports and environment
    vnc_port = 5900 + disp
    web_port = 6080 + disp
    os.environ["XDG_RUNTIME_DIR"] = tempfile.mkdtemp(
        prefix=f"runtime-{os.getenv('USER')}-"
    )
    os.environ["DISPLAY"] = f":{disp}.0"
    os.environ["NO_AT_BRIDGE"] = "1"

    # Setup cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    # Start Xorg
    start_xorg(disp, args.resolution)

    # Start ssh-agent if needed
    local_ssh_agent = start_ssh_agent()

    # TODO: this 'lxde' part we might want to swap with tigervnc, lxterminal etc
    # log_file = Path.home() / f".log/lxsession_X{disp}.log"
    # start_lxde(log_file, processes)

    # for persistance, we should place logs under /home/ubuntu/project/.log (create the folder first)
    # since the docker volume novnc_test is mounted at /home/ubuntu/project
    os.makedirs("/home/ubuntu/project/.log", exist_ok=True)

    log_file = f"/home/ubuntu/project/.log/lxterminal_X{disp}.log"

    log_file = Path(log_file)

    start_lxterminal(log_file, processes)

    # Generate or use existing VNC password
    password = os.getenv("VNCPASS")
    if not password:
        password = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        os.environ["VNCPASS"] = password

    # Start noVNC and x11vnc
    novnc_process = start_novnc(vnc_port, web_port, disp)
    time.sleep(1)  # Extra delay for Singularity containers
    x11vnc_process = start_x11vnc(disp, vnc_port, password)

    # Print connection info
    print("\nOpen your web browser with URL:")
    print(
        f"    http://localhost:{web_port}/vnc.html?resize=downscale&autoconnect=1&password={password}"
    )
    print(
        f"or connect your VNC viewer to localhost:{vnc_port} with password {password}\n"
    )

    # Fix Shift-Tab key mapping
    subprocess.run(["xmodmap", "-e", "keycode 23 = Tab"], check=False)

    # Main monitoring loop
    stop_file = Path.home() / f'.log/stopvnc{os.getenv("DISPLAY")}'
    restart_count = 0

    while restart_count < 100:
        # Write PID file
        pid_file = Path.home() / f".log/x11vnc_pid_X{disp}"
        with open(pid_file, "w") as f:
            f.write(str(x11vnc_process.pid))

        # Wait for x11vnc to exit
        x11vnc_process.wait()

        # Check for stop file
        if stop_file.exists():
            stop_file.unlink()
            break

        # Restart noVNC and x11vnc
        novnc_process.terminate()
        novnc_process = start_novnc(vnc_port, web_port, disp)
        x11vnc_process = start_x11vnc(disp, vnc_port, password)

        if restart_count == 0:
            print("\nX11vnc was restarted probably due to screen-resolution change.")
            print("Please refresh the web browser or reconnect your VNC viewer.\n")

        restart_count += 1


if __name__ == "__main__":
    main()
