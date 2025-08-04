import fastapi
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi import BackgroundTasks
import subprocess
import os
import signal

app = fastapi.FastAPI()

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
        const iframe = document.getElementById("recorder_iframe");
        iframe.contentWindow.location.reload();
    }
    """
    
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
        <!-- <button onclick="reloadIFrame()">Reload Iframe</button> -->
        <iframe id="recorder_iframe" src="{iframe_link}" width="100%" height="100%"></iframe>
    </body>
    </html>
    """

@app.get("/recorder/terminal", response_class=HTMLResponse)
def read_terminal_recorder():

    terminal_iframe_link = "http://localhost:8080"
    javascript = """
    function startRecording() {
        fetch("/start/terminal").then(() => {
            location.reload()
        });
    }
    function stopRecording() {
        fetch("/stop/terminal");
    }
    """
    return read_general_recorder(title="Terminal Recorder", iframe_link=terminal_iframe_link, javascript=javascript)

@app.get("/recorder/gui", response_class=HTMLResponse)
def read_gui_recorder():
    gui_iframe_link = "http://localhost:8081"
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
    # wrap all subcomponent in quotes
    p = subprocess.Popen(["ttyd", "-p", "8080", "--once", "asciinema", "rec", "-c", "bash", "-t", "Terminal Recorder", "-y", "/tmp/terminal.cast", "--overwrite"])
    pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    pid = p.pid
    with open(pidfile, "w") as f:
        f.write(str(pid))
    p.wait()
    return pid

def stop_ttyd():
    pidfile = "/tmp/terminal_recorder_managed_ttyd.pid"
    if not os.path.exists(pidfile):
        print("Pidfile %s not found" % pidfile)
        return
    with open(pidfile, "r") as f:
        pid = f.read()
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        print("Process %s not found" % pid)
    os.remove(pidfile)

@app.get("/start/terminal") # use fastapi background process
def start_terminal_recording(background_tasks: BackgroundTasks):
    # start the ttyd process
    background_tasks.add_task(start_ttyd)
    return "Terminal recording started"

@app.get("/stop/terminal")
def stop_terminal_recording(background_tasks: BackgroundTasks):
    # stop the ttyd process
    # just read the ttyd PID and terminate it.
    background_tasks.add_task(stop_ttyd)
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
    uvicorn.run(app, host="0.0.0.0", port=9001)

if __name__ == "__main__":
    main()