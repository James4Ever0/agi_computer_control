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
from typing import Union, Optional
import shlex

# TODO: store beginning and ending timestamp of recording to record folder
# TODO: check port 8080-8085 are free before launching the server

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
            <li><a href="/recorder/terminal-in-gui">Terminal in GUI Recorder</a></li>
            <li><a href="/recorder/remote-gui/login">Remote GUI Recorder Login</a></li>
            <li><a href="/recorder/remote-terminal/login">Remote Terminal Recorder Login</a></li>
            <li><a href="/recorder/remote-terminal-in-gui/login">Remote Terminal in GUI Recorder Login</a></li>
        </ul>
    </body>
    </html>
    """


@app.get("/recorder/remote-terminal-in-gui", response_class=HTMLResponse)
def read_remote_terminal_in_gui_recorder(
    ip_address: str, port: int, username: str, password: str, show_iframe: bool = False
):  # port: 8084

    gui_iframe_link = (
        f"{EXTERNAL_HOST}:8084/vnc.html?password=password&autoconnect=1&resize=scale&reconnect=1&reconnect_delay=1000"
        if show_iframe
        else ""
    )

    javascript = """
    function startRecording() {
        const url = new URL(window.location.href);

        const ip_address = url.searchParams.get("ip_address");
        const port = url.searchParams.get("port");
        const username = url.searchParams.get("username");
        const password = url.searchParams.get("password");

        // urlencode all params
        const start_gui_recording_url = `/recorder/remote-terminal-in-gui/start?ip_address=${encodeURIComponent(ip_address)}&port=${encodeURIComponent(port)}&username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`;

        fetch(start_gui_recording_url).then(() => {
            // change to "show_iframe=True" in url, then reload
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
        
        fetch("/recorder/remote-terminal-in-gui/stop?description=" + encodeURIComponent(description)).then(() => {
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
        title="Remote terminal in GUI Recorder",
        iframe_link=gui_iframe_link,
        iframe_width="100%",
        iframe_height="100%",
        javascript=javascript,
    )


@app.get("/recorder/terminal-in-gui", response_class=HTMLResponse)
def read_terminal_in_gui_recorder(show_iframe: bool = False):  # port: 8085

    gui_iframe_link = (
        f"{EXTERNAL_HOST}:8085/vnc.html?password=password&autoconnect=1&resize=scale&reconnect=1&reconnect_delay=1000"
        if show_iframe
        else ""
    )

    javascript = """
    function startRecording() {
        const url = new URL(window.location.href);

        const start_gui_recording_url = "/recorder/terminal-in-gui/start";

        fetch(start_gui_recording_url).then(() => {
            // change to "show_iframe=True" in url, then reload
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
        
        fetch("/recorder/terminal-in-gui/stop?description=" + encodeURIComponent(description)).then(() => {
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
        title="Terminal in GUI Recorder",
        iframe_link=gui_iframe_link,
        iframe_width="100%",
        iframe_height="100%",
        javascript=javascript,
    )


@app.get("/recorder/remote-terminal-in-gui/login", response_class=HTMLResponse)
def read_remote_terminal_in_gui_recorder_login():  # port: 8084
    ret = """
<html>
<head>
    <title>Login: Remote terminal in GUI recorder</title>
</head>
<body>
    <h1>Login: Remote terminal in GUI recorder</h1>
    <!-- use query param uri component encode -->
    <form class="form-container" action="/recorder/remote-terminal-in-gui" enctype="application/x-www-form-urlencoded" method="get">
      <div class="form-group">
        <label for="ip_address">IP address:</label>
        <input type="text" id="ip_address" name="ip_address" required>
      </div>

      <div class="form-group">
        <label for="port">Port:</label>
        <input type="text" id="port" name="port" required>
      </div>

      <div class="form-group">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required>
      </div>

      <div class="form-group">
        <label for="password">Password:</label>
        <input type="text" id="password" name="password" required>
      </div>
       <button type="submit" class="submit-btn">Login</button>
    </form>
</body>
</html>
"""
    return ret


@app.get("/recorder/remote-terminal/login", response_class=HTMLResponse)
def read_remote_terminal_recorder_login():
    # collect ip_address, port, username, password with form
    # show_iframe will be false
    # use javascript to GET page /recorder/remote-terminal with encoded parameters, when the "connect" button is clicked
    ret = """
<html>
<head>
    <title>Login: Remote terminal recorder</title>
</head>
<body>
    <h1>Login: Remote terminal recorder</h1>
    <!-- use query param uri component encode -->
    <form class="form-container" action="/recorder/remote-terminal" enctype="application/x-www-form-urlencoded" method="get">
      <div class="form-group">
        <label for="ip_address">IP address:</label>
        <input type="text" id="ip_address" name="ip_address" required>
      </div>

      <div class="form-group">
        <label for="port">Port:</label>
        <input type="text" id="port" name="port" required>
      </div>

      <div class="form-group">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required>
      </div>

      <div class="form-group">
        <label for="password">Password:</label>
        <input type="text" id="password" name="password" required>
      </div>
       <button type="submit" class="submit-btn">Login</button>
    </form>
</body>
</html>
"""
    return ret


@app.get("/recorder/remote-gui/login", response_class=HTMLResponse)
def read_remote_gui_recorder_login():
    ret = """
<html>
<head>
    <title>Login: Remote GUI recorder</title>
</head>
<body>
    <h1>Login: Remote GUI recorder</h1>
    <!-- use query param uri component encode -->
    <form class="form-container" action="/recorder/remote-gui" enctype="application/x-www-form-urlencoded" method="get">
      <div class="form-group">
        <label for="ip_address">IP address:</label>
        <input type="text" id="ip_address" name="ip_address" required>
      </div>

      <div class="form-group">
        <label for="port">Port:</label>
        <input type="text" id="port" name="port" required>
      </div>

      <div class="form-group">
        <label for="password">Password:</label>
        <input type="text" id="password" name="password" required>
      </div>
       <button type="submit" class="submit-btn">Login</button>
    </form>
</body>
</html>    
"""
    return ret


@app.get("/recorder/remote-terminal", response_class=HTMLResponse)
def read_remote_terminal_recorder(
    ip_address: str,
    port: int,
    username: str,
    password: str,
    show_iframe: bool = False,
    user_agent: Union[str, None] = fastapi.Header(default=None),
):  # port: 8082
    """
    Webpage for remote terminal recording.

    This endpoint is used to record the terminal activity on a remote machine.
    The user needs to provide the ip address, port, username, and password of
    the remote machine.

    The `show_iframe` parameter determines whether the iframe with the
    terminal recording is shown in the response. If `show_iframe` is True,
    the iframe is shown. Otherwise, the iframe is not shown.

    The `user_agent` parameter is used to determine the size of the iframe
    based on the user agent. If the user agent is not provided, the iframe
    size is determined based on the default user agent.

    The response is an HTML page with a button to start and stop the
    recording. When the button is clicked, the iframe is shown or hidden
    depending on the value of `show_iframe`.

    The start recording button is implemented as a JavaScript function that
    is called when the button is clicked. The function sends a GET request to
    `/recorder/remote-terminal/start` with the parameters `ip_address`, `port`,
    `username`, and `password`. The request is sent with the `fetch` API.

    is called when the button is clicked. The function sends a GET request to
    `/recorder/remote-terminal/stop` with the parameter `description`. The request is
    sent with the `fetch` API.

    The response is an HTML page with a form to enter the description of the
    recording. When the form is submitted, the description is sent to
    `/recorder/remote-terminal/stop` as a query parameter.

    The iframe size is determined based on the user agent. If the user agent
    is not provided, the iframe size is determined based on the default user
    """
    # user needs to provide the ip address, port, username, and password
    terminal_iframe_link = f"{EXTERNAL_HOST}:8082" if show_iframe else ""

    # TODO: encode connection info into description

    javascript = """
    function startRecording() {
        const url = new URL(window.location.href);

        const ip_address = url.searchParams.get("ip_address");
        const port = url.searchParams.get("port");
        const username = url.searchParams.get("username");
        const password = url.searchParams.get("password");

        // urlencode all params
        const start_terminal_recording_url = `/recorder/remote-terminal/start?ip_address=${encodeURIComponent(ip_address)}&port=${encodeURIComponent(port)}&username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`

        fetch(start_terminal_recording_url).then(() => {
            // change to "show_iframe=True" in url, then reload
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

        // send the description to /recorder/remote-terminal/stop as query params
        window.onbeforeunload = null;

        fetch("/recorder/remote-terminal/stop?description=" + encodeURIComponent(description)).then(() => {
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
    iframe_size = get_terminal_iframe_size_based_on_user_agent(user_agent)

    # finding alternative solutions
    return read_general_recorder(
        title="Terminal Recorder",
        iframe_link=terminal_iframe_link,
        **iframe_size,
        javascript=javascript,
    )


@app.get("/recorder/remote-gui", response_class=HTMLResponse)
def read_remote_gui_recorder(
    ip_address: str, port: int, password: str, show_iframe: bool = False
):  # port: 8083
    # vnc into the remote machine
    # user needs to provide the ip address, port and password
    gui_iframe_link = (
        f"{EXTERNAL_HOST}:8083/vnc.html?password=password&autoconnect=1&resize=scale&reconnect=1&reconnect_delay=1000"
        if show_iframe
        else ""
    )

    javascript = """
    function startRecording() {
        const url = new URL(window.location.href);

        const ip_address = url.searchParams.get("ip_address");
        const port = url.searchParams.get("port");
        const password = url.searchParams.get("password");

        // urlencode all params
        const start_gui_recording_url = `/recorder/remote-gui/start?ip_address=${encodeURIComponent(ip_address)}&port=${encodeURIComponent(port)}&password=${encodeURIComponent(password)}`;

        fetch(start_gui_recording_url).then(() => {
            // change to "show_iframe=True" in url, then reload
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
        
        fetch("/recorder/remote-gui/stop?description=" + encodeURIComponent(description)).then(() => {
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
        title="Remote GUI Recorder",
        iframe_link=gui_iframe_link,
        iframe_width="100%",
        iframe_height="100%",
        javascript=javascript,
    )


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
def read_terminal_recorder(
    show_iframe: bool = False,
    user_agent: Union[str, None] = fastapi.Header(default=None),
):

    terminal_iframe_link = f"{EXTERNAL_HOST}:8080" if show_iframe else ""
    javascript = """
    function startRecording() {
        fetch("/recorder/terminal/start").then(() => {
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

        // send the description to /recorder/terminal/stop as query params
        window.onbeforeunload = null;

        fetch("/recorder/terminal/stop?description=" + encodeURIComponent(description)).then(() => {
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
    iframe_size = get_terminal_iframe_size_based_on_user_agent(user_agent)

    # finding alternative solutions
    return read_general_recorder(
        title="Terminal Recorder",
        iframe_link=terminal_iframe_link,
        **iframe_size,
        javascript=javascript,
    )


def get_terminal_iframe_size_based_on_user_agent(user_agent):
    # for 80x25 terminal, width=615px, height=416px (firefox)
    # in chrome it is different. at least it is fixed sized in the same session
    print("User agent:", user_agent)

    iframe_size = dict(iframe_width="600px", iframe_height="400px")

    if user_agent:
        if "Firefox/" in user_agent:
            print("Using firefox iframe size")
            iframe_size = dict(
                iframe_width="615px",
                iframe_height="416px",
            )
        elif "Chrome/" in user_agent:
            print("Using chrome iframe size")
            iframe_size = dict(
                iframe_width="615px",
                iframe_height="405px",
            )
        else:
            print("Unknown user agent, using default iframe size")
    else:
        print("No user agent, using default iframe size")

    return iframe_size


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
        fetch("/recorder/gui/start").then(() => {
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
        

        fetch("/recorder/gui/stop?description=" + encodeURIComponent(description)).then(() => {
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


def start_ttyd(
    image_name="cybergod_worker_terminal",
    container_name="terminal_recorder_ttyd",
    tmpdir_path="/tmp/cybergod_terminal_recorder_worker_tempdir",
    asciinema_command="bash",
    public_port=8080,
):
    import time
    import json

    try:
        stop_docker_container(container_name)
        # wrap all subcomponent in quotes
    finally:
        if os.path.exists(tmpdir_path):
            shutil.rmtree(tmpdir_path)
    pathlib.Path(tmpdir_path).mkdir(parents=True, exist_ok=True)
    begin_recording_file = os.path.join(tmpdir_path, "begin_recording.txt")
    with open(begin_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "begin_recording"}))
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
        "%s:8080" % public_port,
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
        asciinema_command,
        "-t",
        "TerminalRecorder",
        "-y",
        "/tmp/terminal.cast",
        "--overwrite",
    ]

    print("Executing command:")
    print(" ".join(docker_ttyd_command))
    subprocess.call(docker_ttyd_command)


def stop_docker_container(container_name: str):
    subprocess.call(["docker", "stop", container_name])


def stop_ttyd():
    stop_docker_container("terminal_recorder_ttyd")


def stop_ttyd_remote():
    stop_docker_container("terminal_recorder_ttyd_remote")


def save_ttyd_remote_recording(description: str):
    tmpdir_path = "/tmp/cybergod_terminal_recorder_worker_tempdir_remote"
    container_name = "terminal_recorder_ttyd_remote"
    save_ttyd_recording(
        description, tmpdir_path=tmpdir_path, container_name=container_name
    )


def save_ttyd_recording(
    description: str,
    tmpdir_path="/tmp/cybergod_terminal_recorder_worker_tempdir",
    container_name="terminal_recorder_ttyd",
):
    import time
    import json

    tmp_outputfile = os.path.join(tmpdir_path, "terminal.cast")
    if not os.path.exists(tmp_outputfile):
        print("Output file %s not found" % tmp_outputfile)
        return
    # docker stop <container_name>
    stop_docker_container(container_name)
    stop_recording_file = os.path.join(tmpdir_path, "stop_recording.txt")
    with open(stop_recording_file, "w") as f:
        f.write(json.dumps({"timestamp": time.time(), "event": "stop_recording"}))

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


@app.get("/recorder/remote-terminal/start")
async def start_remote_terminal_recording(
    ip_address: str, port: int, username: str, password: str
):
    asciinema_command = "sshpass -p '%s' ssh -o StrictHostKeyChecking=no %s@%s -p %s" % (
        password,
        username,
        ip_address,
        port,
    )
    start_ttyd(
        image_name="cybergod_worker_terminal:remote-base",
        container_name="terminal_recorder_ttyd_remote",
        tmpdir_path="/tmp/cybergod_terminal_recorder_worker_tempdir_remote",
        asciinema_command=asciinema_command,
        public_port=8082,
    )

    connection_timeout = await wait_for_connection(
        host="127.0.0.1", port=8082, timeout=3
    )
    if connection_timeout:
        return "Remote terminal recording started, but port 8082 is not available"
    return "Remote terminal recording started"


@app.get("/recorder/terminal/start")
async def start_terminal_recording():
    # start the ttyd process
    start_ttyd()
    connection_timeout = await wait_for_connection(
        host="127.0.0.1", port=8080, timeout=3
    )
    if connection_timeout:
        return "Terminal recording started, but port 8080 is not available"
    return "Terminal recording started"


@app.get("/recorder/remote-terminal/stop")
def stop_remote_terminal_recording(description: str):
    save_ttyd_remote_recording(description)
    return "Terminal recording stopped"


@app.get("/recorder/terminal/stop")
def stop_terminal_recording(description: str):
    # stop the ttyd process
    # just read the ttyd PID and terminate it.
    save_ttyd_recording(description)
    return "Terminal recording stopped"


async def start_novnc(
    container_name="gui_recorder_novnc",
    volume_name="cybergod_gui_recorder_x11vnc_project",
    image_name="cybergod_worker_gui",
    tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir",
    run_command=["startvnc.sh"],
    public_port=8081,
    entrypoint: Optional[str] = None,
    extra_run_options: list[str] = [],
):
    import json
    import time

    try:
        stop_novnc(container_name=container_name, volume_name=volume_name)
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

    # the resolution must be one in the xrandr output, or it will fall back to 1920x1080
    resolution = "800x600"

    with open(screenshot_metadata_file, "w+") as f:
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
        "%s:6080" % public_port,
        "--name",
        container_name,
        "-v",
        "%s:%s" % (tmpdir_path, container_gui_record_path),
        "--security-opt",
        "seccomp=unconfined",
        "--cap-add=SYS_PTRACE",
        "-v",
        "%s:%s" % (volume_name, "/home/ubuntu/project"),
        *extra_run_options,
        *(["--entrypoint", entrypoint] if entrypoint else []),
        image_name,
        *run_command,
    ]
    print("Running command:")
    print(" ".join(docker_novnc_command))
    subprocess.call(docker_novnc_command)

    # call the recorder only if the vnc server is ready
    # wait for port 8081 to be available, for most 5 seconds

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


def stop_novnc(
    container_name="gui_recorder_novnc",
    volume_name="cybergod_gui_recorder_x11vnc_project",
):
    stop_docker_container(container_name)
    print("Removing docker volume:", volume_name)
    cmd = ["docker", "volume", "rm", "-f", volume_name]
    subprocess.call(cmd)


def save_novnc_recording(
    description: str, tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir", **kwargs
):
    import json
    import time

    stop_novnc(**kwargs)
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


async def start_novnc_remote(ip_address: str, port: int, password: str):
    script_mountpoint_base = "./record_viewer/novnc_viewer_recorder"
    vncviewer_environments = f"-e VNCVIEWER_VNC_PORT={port} -e VNCVIEWER_VNC_HOST={ip_address} -e VNCVIEWER_VNC_PASSWORD={password}"
    assert os.path.isdir(script_mountpoint_base)
    extra_run_options = shlex.split(
        f"-v {script_mountpoint_base}/startvnc.py:/usr/local/bin/startvnc.py:ro -v {script_mountpoint_base}/lxterminal-cybergod.conf:/home/ubuntu/.config/lxterminal/lxterminal.conf:ro {vncviewer_environments}"
    ) # TODO: remove excessive lxterminal.conf mount

    run_command = shlex.split(
        "python3 /usr/local/bin/startvnc.py --resolution 800x600 --main vncviewer"
    )


    await start_novnc(
        container_name="gui_recorder_novnc_remote",
        volume_name="cybergod_gui_recorder_x11vnc_project_remote",
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_remote",
        image_name="cybergod_worker_gui:remote-base",
        run_command=run_command,
        public_port=8083,
        entrypoint="/usr/bin/env",
        extra_run_options=extra_run_options,
    )


async def start_novnc_remote_terminal(
    ip_address: str, port: int, username: str, password: str
):
    script_mountpoint_base = "./record_viewer/novnc_viewer_recorder"
    # to simplify hostkey checking, we add "-o StrictHostKeyChecking=no" to the ssh command
    lxterminal_init_command = (
        f'"sshpass -p {password} ssh -o StrictHostKeyChecking=no {username}@{ip_address} -p {port}"'
    )
    assert os.path.isdir(script_mountpoint_base)
    extra_run_options = shlex.split(
        f"-v {script_mountpoint_base}/startvnc.py:/usr/local/bin/startvnc.py:ro -v {script_mountpoint_base}/lxterminal-cybergod.conf:/home/ubuntu/.config/lxterminal/lxterminal.conf:ro -e LXTERMINAL_INIT_COMMAND={lxterminal_init_command}"
    )
    run_command = shlex.split(
        "python3 /usr/local/bin/startvnc.py --resolution 640x480 --main lxterminal"
    )

    await start_novnc(
        container_name="gui_recorder_novnc_remote_terminal",
        volume_name="cybergod_gui_recorder_x11vnc_project_remote_terminal",
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_remote_terminal",
        image_name="cybergod_worker_gui:remote-base",
        run_command=run_command,
        public_port=8084,
        entrypoint="/usr/bin/env",
        extra_run_options=extra_run_options,
    )


async def start_novnc_terminal():
    script_mountpoint_base = "./record_viewer/novnc_viewer_recorder"
    assert os.path.isdir(script_mountpoint_base)
    extra_run_options = shlex.split(
        f"-v {script_mountpoint_base}/startvnc.py:/usr/local/bin/startvnc.py:ro -v {script_mountpoint_base}/lxterminal-cybergod.conf:/home/ubuntu/.config/lxterminal/lxterminal.conf:ro"
    )
    run_command = shlex.split(
        "python3 /usr/local/bin/startvnc.py --resolution 640x480 --main lxterminal"
    )
    await start_novnc(
        container_name="gui_recorder_novnc_terminal",
        volume_name="cybergod_gui_recorder_x11vnc_project_terminal",
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_terminal",
        image_name="cybergod_worker_gui:remote-base",
        run_command=run_command,
        public_port=8085,
        entrypoint="/usr/bin/env",
        extra_run_options=extra_run_options,
    )



@app.get("/recorder/gui/start")
async def start_gui_recording():
    # start the novnc process
    await start_novnc()
    return "GUI recording started"


@app.get("/recorder/terminal-in-gui/start")
async def start_terminal_in_gui_recording():
    await start_novnc_terminal()
    return "Terminal in GUI recording started"


@app.get("/recorder/remote-terminal-in-gui/start")
async def start_remote_terminal_in_gui_recording(
    ip_address: str, port: int, username: str, password: str
):
    await start_novnc_remote_terminal(
        ip_address=ip_address, port=port, username=username, password=password
    )
    return "Remote terminal in GUI recording started"


@app.get("/recorder/remote-gui/start")
async def start_remote_gui_recording(ip_address: str, port: int, password: str):
    await start_novnc_remote(ip_address=ip_address, port=port, password=password)
    return "Remote GUI recording started"


@app.get("/recorder/remote-terminal-in-gui/stop")
def stop_remote_terminal_in_gui_recording(description: str):
    save_novnc_recording(
        description,
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_remote_terminal",
        container_name="gui_recorder_novnc_remote_terminal",
        volume_name="cybergod_gui_recorder_x11vnc_project_remote_terminal",
    )
    return "Remote terminal in GUI recording stopped"


@app.get("/recorder/terminal-in-gui/stop")
def stop_terminal_in_gui_recording(description: str):
    save_novnc_recording(
        description,
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_terminal",
        container_name="gui_recorder_novnc_terminal",
        volume_name="cybergod_gui_recorder_x11vnc_project_terminal",
    )
    return "Terminal in GUI recording stopped"


@app.get("/recorder/gui/stop")
def stop_gui_recording(description: str):
    save_novnc_recording(description)
    return "GUI recording stopped"


@app.get("/recorder/remote-gui/stop")
def stop_remote_gui_recording(description: str):
    save_novnc_recording(
        description,
        tmpdir_path="/tmp/cybergod_gui_recorder_worker_tempdir_remote",
        container_name="gui_recorder_novnc_remote",
        volume_name="cybergod_gui_recorder_x11vnc_project_remote",
    )
    return "Remote GUI recording stopped"


def check_is_root():
    import sys

    if os.geteuid() != 0:
        print("This script must be run as root")
        sys.exit(1)


def stop_novnc_terminal():
    stop_novnc(
        container_name="gui_recorder_novnc_terminal",
        volume_name="cybergod_gui_recorder_x11vnc_project_terminal",
    )


def stop_novnc_remote():
    stop_novnc(
        container_name="gui_recorder_novnc_remote",
        volume_name="cybergod_gui_recorder_x11vnc_project_remote",
    )


def stop_novnc_terminal_remote():
    stop_novnc(
        container_name="gui_recorder_novnc_terminal_remote",
        volume_name="cybergod_gui_recorder_x11vnc_project_terminal_remote",
    )


def stop_all_workers():
    stop_ttyd()
    stop_novnc()
    stop_novnc_terminal()
    stop_ttyd_remote()
    stop_novnc_remote()
    stop_novnc_terminal_remote()


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
        stop_all_workers()


if __name__ == "__main__":
    main()
