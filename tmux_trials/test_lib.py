from lib import TmuxServer, TmuxSession
import threading
import time
import traceback
import sys

SERVER_NAME = "test_server"
SESSION_NAME = "test_session"
SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 ubuntu:22.04"
PREVIEW_FILEPATH = "/tmp/test_session_preview.log"
PREVIEW_INTERVAL = 2

STDOUT_FILEPATH="/tmp/test_tmux_stdout.log"

def write_session_preview_with_cursor_periodically(session: TmuxSession):
    while True:
        try:
            preview = session.preview(show_cursor=True)
            if preview is None:
                preview = "Empty preview for session: " + session.name
            with open(PREVIEW_FILEPATH, "w+") as f:
                f.write(preview)
        except:
            exception_log = traceback.format_exc()
            with open(PREVIEW_FILEPATH, "w+") as f:
                f.write("Failed to fetch preview for session: " + session.name + "\n")
                f.write(exception_log)
        finally:
            time.sleep(PREVIEW_INTERVAL)


server = TmuxServer(SERVER_NAME)
env = server.create_env(SESSION_NAME, SESSION_COMMAND)

threading.Thread(
    target=write_session_preview_with_cursor_periodically,
    args=(env.session,),
    daemon=True,
).start()

viewer = env.session.create_viewer()
viewer.add_cmd_pane(f"watch -n {PREVIEW_INTERVAL} cat {PREVIEW_FILEPATH}", "PREVIEW_WITH_CURSOR")
viewer.add_cmd_pane(f"watch -n {PREVIEW_INTERVAL} tac {STDOUT_FILEPATH}", "STDOUT")
# Save the current stdout for later restoration
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
    # Open a file in write mode to redirect stdout
    with open(STDOUT_FILEPATH, 'w+') as f:
        sys.stdout = f  # Redirect stdout to the file object
        sys.stderr = f
        viewer.view()
finally:
    # Restore original stdout after writing is done
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    server.reset()