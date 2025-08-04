import fastapi
from fastapi.responses import HTMLResponse
import uvicorn

app = fastapi.FastAPI()

# return html
@app.get("/", response_model=HTMLResponse)
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

@app.get("/recorder/general", response_model=HTMLResponse)
def read_general_recorder(title = "General Recorder", iframe_link = "", javascript = """
    function startRecording() {
        
    }
    function stopRecording() {
        
    }
    """):
    
    return f"""
    <html>
    <head>
        <title>{title}</title>
    </head>
    <script>
    {javascript}
    </script>
    <body>
        <h1>{title}</h1>
        <iframe src="{iframe_link}"></iframe>
        <textarea id="description"></textarea>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
    </body>
    </html>
    """

@app.get("/recorder/terminal", response_model=HTMLResponse)
def read_terminal_recorder():
    return read_general_recorder("Terminal Recorder", "http://localhost:8080")

@app.get("/recorder/gui", response_model=HTMLResponse)
def read_gui_recorder():
    return read_general_recorder("GUI Recorder", "http://localhost:8081")

def main():
    uvicorn.run(app, host="0.0.0.0", port=9001)

if __name__ == "__main__":
    main()