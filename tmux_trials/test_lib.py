from lib import TmuxServer, TmuxSession, TmuxEnvironment
import threading
import time
import traceback
import sys
import os
from io import TextIOWrapper
import string
import random
import black



SERVER_NAME = "test_server"
SESSION_NAME = "test_session"
# SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 ubuntu:22.04"
SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 openinterpreter /bin/bash"
PREVIEW_FILEPATH = "/tmp/test_session_preview.html"
STATS_FILEPATH = "/tmp/test_session_stats.log"
PREVIEW_INTERVAL = 2
FLUSH_INTERVAL = 1
SEND_KEY_INTERVAL = 1

STDOUT_FILEPATH = "/tmp/test_tmux_stdout.log"

if os.path.exists(STDOUT_FILEPATH):
    print("[*] Removing old log file:", STDOUT_FILEPATH)
    os.remove(STDOUT_FILEPATH)

def format_python_object(obj):
    ret = black.format_str(repr(obj), mode=black.FileMode())
    return ret

def flush_filehandle_periodically(f: TextIOWrapper):
    while True:
        f.flush()
        time.sleep(FLUSH_INTERVAL)


def start_daemon_thread(target, *args, **kwargs):
    t = threading.Thread(target=target, args=args, kwargs=kwargs)
    t.daemon = True
    t.start()


def write_env_stats_periodically(env: TmuxEnvironment):
    while True:
        try:
            content=format_python_object(env.info)
            #content = ujson.dumps(env.info,indent=4)
        except:
            content = "Failed to log stats for TmuxEnvironment\n"
            content += traceback.format_exc()
        finally:
            time.sleep(PREVIEW_INTERVAL)
        with open(STATS_FILEPATH, "w+") as f:
            f.write(content)


def write_session_preview_with_cursor_periodically(session: TmuxSession):
    while True:
        try:
            # preview = session.preview(show_cursor=True)
            # preview = session.preview(show_cursor=True, block_style=True)
            # preview = session.preview_html(show_cursor=False)
            # preview = session.preview_html(show_cursor=True)
            preview = session.preview_html(show_cursor=True, block_style=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True, block_style=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True, grayscale=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True, grayscale=True, cursor_char="_")
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True, grayscale=True, block_style=True)
            # preview = session.preview_html(show_cursor=True,wrap_html=True, dark_mode=True, grayscale=True, block_style=True, block_css_style="newDiv.style.borderLeft='1.5px solid red';")
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


def generate_random_keys():
    ret = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    return ret


def test_key_inputs(env: TmuxEnvironment):
    while True:
        print("[*] Sending test keys...")
        # env.send_key("date")
        random_keys = generate_random_keys()
        env.send_key(random_keys)
        env.send_key("Enter")
        time.sleep(SEND_KEY_INTERVAL)


server = TmuxServer(SERVER_NAME)
env = server.create_env(SESSION_NAME, SESSION_COMMAND)

start_daemon_thread(write_session_preview_with_cursor_periodically, env.session)
start_daemon_thread(write_env_stats_periodically, env)
viewer = env.session.create_viewer(default_layout="tiled")
viewer.add_cmd_pane(
    f"watch -n {PREVIEW_INTERVAL} cat {PREVIEW_FILEPATH}", "PREVIEW_WITH_CURSOR"
)
viewer.add_cmd_pane(f"watch -n {PREVIEW_INTERVAL} cat {STATS_FILEPATH}", "ENV_STATS")
viewer.add_cmd_pane(f"watch -n {PREVIEW_INTERVAL} tac {STDOUT_FILEPATH}", "STDOUT")
# Save the current stdout for later restoration
original_stdout = sys.stdout
original_stderr = sys.stderr

# start_daemon_thread(test_key_inputs, env)

exception_log = None
try:
    # Open a file in write mode to redirect stdout
    with open(STDOUT_FILEPATH, "a+") as f:
        sys.stdout = f  # Redirect stdout to the file object
        sys.stderr = f
        start_daemon_thread(flush_filehandle_periodically, f)
        viewer.view()
except:
    exception_log = traceback.format_exc()
finally:
    # Restore original stdout after writing is done
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    print("Printing output from:", STDOUT_FILEPATH)
    os.system("cat "+STDOUT_FILEPATH)
    if exception_log:
        print(exception_log)
        print("Exception found while running viewer.")
    else:
        print("No exception during running viewer.")
    print("Cleaning up by resetting server")
    server.reset()
