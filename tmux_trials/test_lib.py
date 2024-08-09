from lib import TmuxServer, TmuxSession, TmuxEnvironment
import threading
import time
import traceback
import sys
import os
from io import TextIOWrapper

SERVER_NAME = "test_server"
SESSION_NAME = "test_session"
# SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 ubuntu:22.04"
SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 openinterpreter /bin/bash"
PREVIEW_FILEPATH = "/tmp/test_session_preview.html"
PREVIEW_INTERVAL = 2
FLUSH_INTERVAL = 1
SEND_KEY_INTERVAL = 1

STDOUT_FILEPATH = "/tmp/test_tmux_stdout.log"

if os.path.exists(STDOUT_FILEPATH):
    print("[*] Removing old log file:", STDOUT_FILEPATH)
    os.remove(STDOUT_FILEPATH)


def flush_filehandle_periodically(f: TextIOWrapper):
    while True:
        f.flush()
        time.sleep(FLUSH_INTERVAL)


def start_daemon_thread(target, *args, **kwargs):
    t = threading.Thread(target=target, args=args, kwargs=kwargs)
    t.daemon = True
    t.start()


def write_session_preview_with_cursor_periodically(session: TmuxSession):
    while True:
        try:
            # preview = session.preview(show_cursor=True)
            # preview = session.preview_html(show_cursor=False)
            # preview = session.preview_html(show_cursor=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True)
            preview = session.preview_html(show_cursor=True,wrap_html=True, daek_mode=True)
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


def test_key_inputs(env: TmuxEnvironment):
    while True:
        print("[*] Sending test keys...")
        env.send_key("date")
        env.send_key("Enter")
        time.sleep(SEND_KEY_INTERVAL)


server = TmuxServer(SERVER_NAME)
env = server.create_env(SESSION_NAME, SESSION_COMMAND)

start_daemon_thread(write_session_preview_with_cursor_periodically, env.session)
viewer = env.session.create_viewer()
viewer.add_cmd_pane(
    f"watch -n {PREVIEW_INTERVAL} cat {PREVIEW_FILEPATH}", "PREVIEW_WITH_CURSOR"
)
viewer.add_cmd_pane(f"watch -n {PREVIEW_INTERVAL} tac {STDOUT_FILEPATH}", "STDOUT")
# Save the current stdout for later restoration
original_stdout = sys.stdout
original_stderr = sys.stderr

# start_daemon_thread(test_key_inputs, env)

try:
    # Open a file in write mode to redirect stdout
    with open(STDOUT_FILEPATH, "a+") as f:
        sys.stdout = f  # Redirect stdout to the file object
        sys.stderr = f
        start_daemon_thread(flush_filehandle_periodically, f)
        viewer.view()
finally:
    # Restore original stdout after writing is done
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    server.reset()
