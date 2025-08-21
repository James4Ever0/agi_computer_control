import pathlib
import fastapi
from fastapi.responses import HTMLResponse
import uvicorn
import subprocess
import os
import shutil
import datetime
import socket
import asyncio
from typing import Union
# TODO: store beginning and ending timestamp of recording to record folder

app = fastapi.FastAPI()

EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST", "http://127.0.0.1")

print("Using external host: %s" % EXTERNAL_HOST)


async def wait_for_connection(host: str, port: int, timeout: int):
    connection_timeout = True
    for i in range(timeout):
        try:
            conn = socket.create_connection((host, port))
            print("Port %s is available after %s seconds" % (port, i))
            connection_timeout = False
            conn.close()
            break
        except:
            await asyncio.sleep(1)
    if connection_timeout:
        print("Port port is not available in %s seconds" % timeout)
    return connection_timeout


# actually the base gui recorder can be tuned into a remote ssh client.
# the recording will be directly available as a series of screenshots, mouse and keyboard events

# return html
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
    <head>
        <title>Web GUI Terminal Recorder</title>
    </head>
    <body>
        <h1>Web GUI/Terminal Recorder</h1>
        <p>Choose a recorder type:</p>
        <ul>
            <li><a href="/recorder/terminal">Terminal Recorder</a></li>
            <li><a href="/recorder/gui">GUI Recorder</a></li>
            <li><a href="/recorder/remote-gui">Remote GUI Recorder</a></li>
            <li><a href="/recorder/remote-terminal">Remote Terminal Recorder</a></li>
        </ul>
    </body>
    </html>
    """

@app.get("/recorder/remote-terminal", response_class=HTMLResponse)
def read_remote_terminal_recorder():
    # ssh into the remote machine
    # user needs to provide the ip address, port, username, and password
    ...

@app.get("/recorder/remote-gui", response_class=HTMLResponse)
def read_remote_gui_recorder():
    # vnc into the remote machine
    # user needs to provide the ip address, port and password
    ...

@app.get("/recorder/general", response_class=HTMLResponse)
def read_general_recorder(
    title="General Recorder",
    iframe_link="",
    iframe_width="100%",
    iframe_height="100%",
    javascript="""
    function startRecording() {
        console.log("Recording started (stub)")
    }
    function stopRecording() {
        console.log("Recording stopped (stub)")
    }
    """,
):

    
    reload_iframe_javascript = """
    function reloadIFrame(){
        // this function does not work in firefox. do we need to use nginx for mapping all urls into the same host and port? or is there some wrong header in the ttyd respose?
        const iframe = document.getElementById("recorder_iframe");
        iframe.contentWindow.location.reload();
    }
    function reloadPage(){
        location.reload();
    }
    """
    window_onbeforeunload_javascript = """
    window.onbeforeunload = function () {
     // Your Code here 
      return null;  // return null to avoid pop up
    }
    """
    if iframe_link:
        iframe_elem = f"""<iframe id="recorder_iframe" width="{iframe_width}" height="{iframe_height}" src="{iframe_link}" ></iframe>"""
    else:
        iframe_elem = ""

    return f"""
    <!-- kind of weird. do not add the following thing into the html file, otherwise the height of novnc iframe will not work -->
    <!-- <!DOCTYPE html> -->
    <html>
    <head>
        <title>{title}</title>
    </head>
    <script>
    {javascript}
    {reload_iframe_javascript}
    {window_onbeforeunload_javascript}
    </script>
    <body>
        <h1>{title}</h1>
        <div>
        <textarea id="description" placeholder="description"></textarea>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
        <button onclick="reloadPage()">Reload Page</button>
        </div>
        {iframe_elem}
    </body>
    </html>
    """


@app.get("/recorder/terminal", response_class=HTMLResponse)
def read_terminal_recorder(show_iframe: bool = False, user_agent: Union[str, None] = fastapi.Header(default=None)):

    terminal_iframe_link = f"{EXTERNAL_HOST}:8080" if show_iframe else ""
    javascript = """
    function startRecording() {
        fetch("/start/terminal").then(() => {
            // change to "show_iframe=True" in url, then reload
            const url = new URL(window.location.href);
            url.searchParams.set("show_iframe", "true");
            window.location.href = url.toString();
        });
    }
    function stopRecording() {
        const description = document.getElementById("description").value;
        if (!description) {
            alert("Please enter a description");
            return;
        }

        // send the description to /stop/terminal as query params
        window.onbeforeunload = null;

        fetch("/stop/terminal?description=" + encodeURIComponent(description)).then(() => {
            // change to "show_iframe=False" in url, then reload
            const url = new URL(window.location.href);
            url.searchParams.set("show_iframe", "false");
            document.getElementById("description").value = "";
            // remove the iframe
            document.getElementById("recorder_iframe").remove();
            window.location.href = url.toString();
        });
    }
    """
    # for 80x25 terminal, width=615px, height=416px (firefox)
    # in chrome it is different. at least it is fixed sized in the same session
    print("User agent:", user_agent)

    iframe_size = dict(
        iframe_width="600px",
        iframe_height="400px"
    )

    if user_agent:
        if "Firefox/" in user_agent:
            print("Using firefox iframe size")
            iframe_size = dict(
            iframe_width="615px",
            iframe_height="416px",)
        elif "Chrome/" in user_agent:
            print("Using chrome iframe size")
            iframe_size = dict(
            iframe_width="615px",
            iframe_height="405px",)
        else:
            print("Unknown user agent, using default iframe size")
    else:
        print("No user agent, using default iframe size")

    # finding alternative solutions
    return read_general_recorder(
        title="Terminal Recorder",
        iframe_link=terminal_iframe_link,
        **iframe_size,
        javascript=javascript,
    )


@app.get("/recorder/gui", response_class=HTMLResponse)
def read_gui_recorder(show_iframe: bool = False):
    # set resize=scale to make it fit into window
    # ref: https://novnc.com/noVNC/docs/EMBEDDING.html
    gui_iframe_link = (
        f"{EXTERNAL_HOST}:8081/vnc.html?password=password&autoconnect=1&resize=scale&reconnect=1&reconnect_delay=1000"
        if show_iframe
        else ""
    )
    javascript = """
    function startRecording() {
        fetch("/start/gui").then(() => {
            // change to "show_iframe=True" in url, then reload
            const url = new URL(window.location.href);
            url.searchParams.set("show_iframe", "true");
            window.location.href = url.toString();
        });
    }
    function stopRecording() {
        const description = document.getElementById("description").value;
        if (!description) {
            alert("Please enter a description");
            return;
        }
        

        fetch("/stop/gui?description=" + encodeURIComponent(description)).then(() => {
            // change to "show_iframe=False" in url, then reload
            const url = new URL(window.location.href);
            url.searchParams.set("show_iframe", "false");
            document.getElementById("description").value = "";
            document.getElementById("recorder_iframe").remove();
            window.location.href = url.toString();
        });
    }
    """
    return read_general_recorder(
        title="GUI Recorder",
        iframe_link=gui_iframe_link,
        iframe_width="100%",
        iframe_height="100%",
        javascript=javascript,
    )


def start_ttyd():
    import time
    import json

    tmpdir_path = "/tmp/cybergod_terminal_recorder_worker_tempdir"
    try:
        stop_ttyd()
        # wrap all subcomponent in quotes
    finally:
        if os.path.exists(tmpdir_path):
            shutil.rmtree(tmpdir_path)
    pathlib.Path(tmpdir_path).mkdir(parents=True, exist_ok=True)
    begin_recording_file = os.path.join(tmpdir_path, "begin_recording.txt")
    with open(begin_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "begin_recording"}))
    # poc_ttyd_command = ["ttyd", "-p", "8080", "--once", "asciinema", "rec", "-c", "bash", "-t", "Terminal Recorder", "-y", "%s/terminal.cast" % tmpdir_path, "--overwrite"]
    image_name = "cybergod_worker_terminal"
    container_name = "terminal_recorder_ttyd"
    # web terminal apps on xterm.js official website: https://xtermjs.org/
    # TODO: ttyd uses xterm.js. maybe we can tweak there for 80x25 fixed size terminal
    # hint: run "window.term.resize(80, 25)" then get the actual size of the terminal element later
    # TODO: state is not persisted. may use ssh to connect to a persistant machine

    # gotty supports fixed column and height
    # ref: https://github.com/sorenisanerd/gotty
    docker_ttyd_command = [
        "docker",
        "run",
        "--rm",
        "--tty",
        "-d",
        "--publish",
        "8080:8080",
        "--name",
        container_name,
        "-v",
        "%s:/tmp" % tmpdir_path,
        "--entrypoint",
        "ttyd",
        image_name,
        "-p",
        "8080",
        "--once",
        "asciinema",
        "rec",
        "-c",
        "bash",
        "-t",
        "TerminalRecorder",
        "-y",
        "/tmp/terminal.cast",
        "--overwrite",
    ]

    print("Executing command:")
    print(" ".join(docker_ttyd_command))
    subprocess.call(docker_ttyd_command)
    # pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    # pid = p.pid
    # with open(pidfile, "w") as f:
    #     f.write(str(pid))
    # p.wait()
    # return pid


def stop_docker_container(container_name: str):
    subprocess.call(["docker", "stop", container_name])


def stop_ttyd():
    stop_docker_container("terminal_recorder_ttyd")


def save_ttyd_recording(description: str):
    import time
    import json

    # pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    tmpdir_path = "/tmp/cybergod_terminal_recorder_worker_tempdir"
    tmp_outputfile = os.path.join(tmpdir_path, "terminal.cast")
    if not os.path.exists(tmp_outputfile):
        print("Output file %s not found" % tmp_outputfile)
        return
    # docker stop terminal_recorder_ttyd
    stop_ttyd()
    stop_recording_file = os.path.join(tmpdir_path, "stop_recording.txt")
    with open(stop_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "stop_recording"}))

    # try:
    #     os.kill(int(pid), signal.SIGTERM)
    #     print("Terminated ttyd process %s" % pid)
    # except ProcessLookupError:
    #     print("Process %s not found" % pid)
    # os.remove(pidfile)
    # copy the output file to destination

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = "./record/terminal/terminal_record_%s" % timestamp
    description_savepath = os.path.join(destination, "description.txt")
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.copytree(tmpdir_path, destination, dirs_exist_ok=True)
    with open(description_savepath, "w") as f:
        f.write(description)
    print("Copied %s to %s" % (tmp_outputfile, destination))
    os.remove(tmp_outputfile)
    print("Removed %s" % tmp_outputfile)


@app.get("/start/terminal")
async def start_terminal_recording():
    # start the ttyd process
    start_ttyd()
    connection_timeout = await wait_for_connection(
        host="127.0.0.1", port=8080, timeout=3
    )
    if connection_timeout:
        return "Terminal recording started, but port 8080 is not available"
    return "Terminal recording started"


@app.get("/stop/terminal")
def stop_terminal_recording(description: str):
    # stop the ttyd process
    # just read the ttyd PID and terminate it.
    save_ttyd_recording(description)
    return "Terminal recording stopped"


async def start_novnc():
    import json
    import time

    image_name = "cybergod_worker_gui"
    container_name = "gui_recorder_novnc"
    volume_name = "cybergod_gui_recorder_x11vnc_project"

    tmpdir_path = "/tmp/cybergod_gui_recorder_worker_tempdir"
    try:
        stop_novnc()
    finally:
        if os.path.exists(tmpdir_path):
            shutil.rmtree(tmpdir_path)
    container_gui_record_path = "/tmp/gui_record_data"
    pathlib.Path(tmpdir_path).mkdir(parents=True, exist_ok=True)

    begin_recording_file = os.path.join(tmpdir_path, "begin_recording.txt")

    screenshot_metadata_file = os.path.join(tmpdir_path, "screenshot_metadata.json")
    with open(begin_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "begin_recording"}))

    # the docker command here is to be cross-referenced with x11vnc-docker launch script (python)
    # ref: https://github.com/x11vnc/x11vnc-desktop/blob/main/x11vnc_desktop.py
    vnc_password = "password"
    # too many local directories mounted to this container using the original python script
    # docker run -d --rm --name x11vnc-zrvgaz --shm-size 2g -p 6080:6080 -p 5950:5900 --hostname x11vnc-zrvgaz --env VNCPASS= --env RESOLUT=1920x1080 --env HOST_UID=1000 --env HOST_GID=1000 -p 2222:22 -v /media/jamesbrown/Ventoy/agi_computer_control/web_gui_terminal_recorder:/home/ubuntu/shared -v x11vnc_zh_CN_config:/home/ubuntu/.config -v /home/jamesbrown/.gnupg:/home/ubuntu/.gnupg -v /home/jamesbrown/.gitconfig:/home/ubuntu/.gitconfig_host -v cybergod_gui_recorder_x11vnc_project:/home/ubuntu/project -w /home/ubuntu/project -v /home/jamesbrown/.ssh:/home/ubuntu/.ssh --security-opt seccomp=unconfined --cap-add=SYS_PTRACE x11vnc/docker-desktop:zh_CN startvnc.sh >> /home/ubuntu/.log/vnc.log

    # the resolution must be one in the xrandr output, or it will fall back to 1920x1080
    # resolution = "1920x1080"
    resolution = "800x600" 
    # resolution = "1280x800"

    with open(screenshot_metadata_file, 'w+') as f:
        f.write(json.dumps(dict(presumed_resolution=resolution)))
    docker_novnc_command = [
        "docker",
        "run",
        "--rm",
        "--tty",
        "-d",
        "-e",
        "RESOLUT=%s" % resolution,
        "-e",
        "VNCPASS=%s" % vnc_password,
        "--publish",
        "8081:6080",
        "--publish",
        "8950:5900",
        "--name",
        container_name,
        "-v",
        "%s:%s" % (tmpdir_path, container_gui_record_path),
        "--security-opt",
        "seccomp=unconfined",
        "--cap-add=SYS_PTRACE",
        "-v",
        "%s:%s" % (volume_name, "/home/ubuntu/project"),
        image_name,
        "startvnc.sh",
    ]
    print("Running command:")
    print(" ".join(docker_novnc_command))
    subprocess.call(docker_novnc_command)

    # call the recorder only if the vnc server is ready
    # wait for port 8081 to be available, for most 5 seconds

    # novnc_connection_timeout=await wait_for_connection(host="127.0.0.1", port =8081, timeout = 5)
    # vnc_connection_timeout=await wait_for_connection(host="127.0.0.1", port =8950, timeout = 5)
    # connection_timeout= vnc_connection_timeout or novnc_connection_timeout
    view_log_cmd = ["docker", "logs", container_name]
    ready = False
    # timeout in 7 seconds.
    for i in range(7):
        container_log_output = subprocess.check_output(view_log_cmd)
        if "Open your web browser with URL:" in container_log_output.decode():
            ready = True
            print("novnc service ready at %s seconds" % i)
            break
        await asyncio.sleep(1)
    if not ready:
        return "GUI recording started, but not all services available"
    else:
        display_name = ":0.0"
        docker_novnc_recorder_command = [
            "docker",
            "exec",
            "-d",
            "-e",
            "DISPLAY=%s" % display_name,
            container_name,
            "python3",
            "/home/ubuntu/worker_gui.py",
            "--output_dir",
            container_gui_record_path,
        ]
        print("Executing command:")
        print(" ".join(docker_novnc_recorder_command))
        subprocess.call(docker_novnc_recorder_command)
        return "GUI recording started"


def stop_novnc():
    container_name = "gui_recorder_novnc"
    stop_docker_container(container_name)
    volume_name = "cybergod_gui_recorder_x11vnc_project"
    print("Removing docker volume:", volume_name)
    cmd = ["docker", "volume", "rm", "-f", volume_name]
    subprocess.call(cmd)


def save_novnc_recording(description: str):
    import json
    import time

    stop_novnc()
    tmpdir_path = "/tmp/cybergod_gui_recorder_worker_tempdir"
    stop_recording_file = os.path.join(tmpdir_path, "stop_recording.txt")
    with open(stop_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "stop_recording"}))
    savepath = "./record/gui"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = os.path.join(savepath, "gui_record_%s" % timestamp)
    description_savepath = os.path.join(destination, "description.txt")
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.copytree(tmpdir_path, destination, dirs_exist_ok=True)
    print("Copied %s to %s" % (tmpdir_path, destination))
    with open(description_savepath, "w") as f:
        f.write(description)
    print("Saved description to %s" % description_savepath)
    shutil.rmtree(tmpdir_path)
    print("Removed %s" % tmpdir_path)


@app.get("/start/gui")
async def start_gui_recording():
    # start the novnc process
    await start_novnc()
    return "GUI recording started"


@app.get("/stop/gui")
def stop_gui_recording(description: str):
    save_novnc_recording(description)
    return "GUI recording stopped"


def check_is_root():
    import sys

    if os.geteuid() != 0:
        print("This script must be run as root")
        sys.exit(1)


def main():
    # need to be root to run this script, otherwise some files may not get removed
    # TODO: use python managed tempdir instead (will it solves the issue of files in docker volume being owned by root?)
    check_is_root()
    try:
        uvicorn.run(app, host="0.0.0.0", port=9001)
    except KeyboardInterrupt:
        print("User interrputed")
    finally:
        print("Running cleanup jobs")
        # cleanup jobs
        stop_ttyd()
        stop_novnc()


if __name__ == "__main__":
    main()
