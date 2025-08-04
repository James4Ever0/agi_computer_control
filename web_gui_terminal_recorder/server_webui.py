import pathlib
import fastapi
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi import BackgroundTasks
import subprocess
import os
import signal
import shutil
import datetime
import socket
import asyncio

app = fastapi.FastAPI()

EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST", "http://localhost")

print("Using external host: %s" % EXTERNAL_HOST)

# return html
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
    <head>
        <title>Web GUI Terminal Recorder</title>
    </head>
    <body>
        <h1>Web GUI Terminal Recorder</h1>
        <p>Choose a recorder type:</p>
        <ul>
            <li><a href="/recorder/terminal">Terminal Recorder</a></li>
            <li><a href="/recorder/gui">GUI Recorder</a></li>
        </ul>
    </body>
    </html>
    """

@app.get("/recorder/general", response_class=HTMLResponse)
def read_general_recorder(title = "General Recorder", iframe_link = "", javascript = """
    function startRecording() {
        console.log("Recording started (stub)")
    }
    function stopRecording() {
        console.log("Recording stopped (stub)")
    }
    """):

    reload_iframe_javascript = """
    function reloadIFrame(){
        // this function does not work in javascript. do we need to use nginx?
        const iframe = document.getElementById("recorder_iframe");
        iframe.contentWindow.location.reload();
    }
    function reloadPage(){
        location.reload();
    }
    """
    if iframe_link:
        iframe_elem = f"<iframe id='recorder_iframe' src='{iframe_link}' width='100%' height='100%'></iframe>"
    else:
        iframe_elem = ""
    
    return f"""
    <html>
    <head>
        <title>{title}</title>
    </head>
    <script>
    {javascript}
    {reload_iframe_javascript}
    </script>
    <body>
        <h1>{title}</h1>
        <textarea id="description"></textarea>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
        <button onclick="reloadPage()">Reload Page</button>
        {iframe_elem}
    </body>
    </html>
    """

@app.get("/recorder/terminal", response_class=HTMLResponse)
def read_terminal_recorder(show_iframe:bool=False):

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

        fetch("/stop/terminal?description=" + encodeURIComponent(description)).then(() => {
            // change to "show_iframe=False" in url, then reload
            const url = new URL(window.location.href);
            url.searchParams.set("show_iframe", "false");
            window.location.href = url.toString();
        });
    }
    """
    return read_general_recorder(title="Terminal Recorder", iframe_link=terminal_iframe_link, javascript=javascript)

@app.get("/recorder/gui", response_class=HTMLResponse)
def read_gui_recorder(show_iframe=False):
    gui_iframe_link = f"{EXTERNAL_HOST}:8081" if show_iframe else ""
    javascript = """
    function startRecording() {
        // there is no need to reload
        fetch("/start/gui");
    }
    function stopRecording() {
        fetch("/stop/gui");
    }
    """
    return read_general_recorder(title="GUI Recorder", iframe_link=gui_iframe_link, javascript=javascript)

def start_ttyd():
    stop_ttyd()
    # wrap all subcomponent in quotes
    tmpdir_path = "/tmp/cybergod_terminal_recorder_worker_tempdir"
    pathlib.Path(tmpdir_path).mkdir(parents=True, exist_ok=True)
    # poc_ttyd_command = ["ttyd", "-p", "8080", "--once", "asciinema", "rec", "-c", "bash", "-t", "Terminal Recorder", "-y", "%s/terminal.cast" % tmpdir_path, "--overwrite"]
    image_name = "cybergod_worker_terminal"
    container_name = "terminal_recorder_ttyd"
    docker_ttyd_command = ["docker", "run", "--rm", "--tty", "-d", "--publish", "8080:8080", "--name", container_name, "-v", "%s:/tmp" % tmpdir_path,"--entrypoint", "ttyd", image_name, "-p", "8080", "--once", "asciinema", "rec", "-c", "bash", "-t", "TerminalRecorder", "-y", "/tmp/terminal.cast", "--overwrite"]
    print("Executing command:", " ".join(docker_ttyd_command))
    subprocess.call(docker_ttyd_command)
    # pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    # pid = p.pid
    # with open(pidfile, "w") as f:
    #     f.write(str(pid))
    # p.wait()
    # return pid

def stop_ttyd():
    subprocess.call(["docker", "stop", "terminal_recorder_ttyd"])

def save_ttyd_recording(description:str):

    # pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    tmpdir_path = "/tmp/cybergod_terminal_recorder_worker_tempdir"
    outputfile = os.path.join(tmpdir_path, "terminal.cast")
    if not os.path.exists(outputfile):
        print("Output file %s not found" % outputfile)
        return
    # docker stop terminal_recorder_ttyd
    stop_ttyd()
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
    shutil.copy(outputfile, destination)
    with open(description_savepath, "w") as f:
        f.write(description)
    print("Copied %s to %s" % (outputfile, destination))

@app.get("/start/terminal") # use fastapi background process
async def start_terminal_recording():
    # start the ttyd process
    start_ttyd()
    connection_timeout=True
    # wait for port 8080 to be available, for most 3 seconds
    timeout = 3
    for i in range(timeout):
        try:
            conn = socket.create_connection(("127.0.0.1", 8080))
            print("Port 8080 is available after %s seconds" % (i))
            connection_timeout=False
            conn.close()
            break
        except:
            await asyncio.sleep(1)
    if connection_timeout:
        print("Port 8080 is not available in %s seconds" % timeout)
        return "Terminal recording started, but port 8080 is not available"
    return "Terminal recording started"

@app.get("/stop/terminal")
def stop_terminal_recording(description:str):
    # stop the ttyd process
    # just read the ttyd PID and terminate it.
    save_ttyd_recording(description)
    return "Terminal recording stopped"

@app.get("/start/gui")
def start_gui_recording():
    # start the novnc process
    return "GUI recording started"

@app.get("/stop/gui")
def stop_gui_recording():
    # stop the novnc process
    return "GUI recording stopped"

def main():
    try:
        uvicorn.run(app, host="0.0.0.0", port=9001)
    finally:
        print("Server down.")
        print("Running cleanup jobs")
        # cleanup jobs
        stop_ttyd()

if __name__ == "__main__":
    main()