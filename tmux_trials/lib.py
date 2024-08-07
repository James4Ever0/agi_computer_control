from typing import Optional, Iterable
from wcwidth import wcswidth
import subprocess
import os
import traceback
import shutil
import parse
import json
from tempfile import NamedTemporaryFile

ENV='env'
TMUX = "tmux"
TMUXP = "tmuxp"
TMUX_WIDTH = 80
TMUX_HEIGHT = 24
ENCODING = "utf-8"

NEWLINE = "\n"
NEWLINE_BYTES = NEWLINE.encode(ENCODING)

CMDLIST_EXECUTE_TIMEOUT = 10

CURSOR = "<<<CURSOR>>>"
CURSOR_BYTES = CURSOR.encode(ENCODING)

REQUIRED_BINARIES = (ENV, TMUX, TMUXP)


def decode_bytes(_bytes: bytes, errors='ignore'):
    ret = _bytes.decode(ENCODING, errors=errors)
    return ret


def insert_cursor(_bytes: bytes, x: int):
    ret = b""
    try:
        line = decode_bytes(_bytes, errors='replace')
        char_index = 0
        for index, it in enumerate(line):
            if char_index >= x:
                ret = line[:index] + CURSOR + line[index:]
                ret = ret.encode(ENCODING)
                break
            char_width = wcswidth(it)
            char_index += char_width
        if ret == b"":
            ret = _bytes + CURSOR_BYTES
    except UnicodeDecodeError:
        print("[-] Failed to decode line while inserting cursor:", _bytes)
        print("[*] Falling back to bytes insert mode")
        ret = _bytes[:x] + CURSOR_BYTES + _bytes[x:]
    return ret


def assert_binary_existance(binaries: Iterable[str]):
    for it in binaries:
        assert shutil.which(it) != None, f"Binary '{it}' not found in path"


assert_binary_existance(REQUIRED_BINARIES)


def json_pretty_print(obj):
    ret = json.dumps(obj, indent=4, sort_keys=True, ensure_ascii=False)
    print(ret)
    return ret


def warn_nonzero_exitcode(exitcode: int):
    ret = exitcode == 0
    if not ret:
        print("[-] Process failed with exit code:", exitcode)
    return ret


class TmuxServer:
    def __init__(self, name: Optional[str] = None, reset: bool = True):
        if name is None:
            print("[*] Falling back to default server name")
            name = "default"
        self.name = name
        self.prefix = f"{TMUX} -L {self.name}"
        self.prefix_list = self.prefix.split(" ")
        print(f"[*] Tmux server using name '{self.name}'")

        if reset:
            self.reset()

    def reset(self):
        exit_code = self.tmux_execute_command("kill-server")
        ret = exit_code == 0
        if ret:
            print(f"[+] Server '{self.name}' reset complete")
        else:
            print(f"[+] Server '{self.name}' reset failed with exit code", exit_code)
        return ret

    def create_session(self, name: str, command: str):
        try:
            ret = TmuxSession(name, self, command)
            print(f"[+] Tmux session '{name}' created")
            return ret
        except TmuxSessionCreationFailure:
            traceback.print_exc()
            print(f"[-] Failed to create tmux session named '{name}'")

    def set_session_option(self, name: str, key: str, value: str):
        self.tmux_execute_command(f"set-option -t {name} {key} {value}")

    def kill_session(self, name: str):
        self.tmux_execute_command(f"kill-session -t {name}")

    def create_env(self, name: str, command: str):
        session = self.create_session(name, command)
        if session:
            ret = TmuxEnv(session)
            print("[+] Tmux env created")
            return ret
        else: 
            raise TmuxEnvCreationFailure("[-] Failed to create tmux env")

    def tmux_prepare_command(self, suffix: str):
        ret = f"{self.prefix} {suffix}"
        return ret

    def tmux_prepare_attach_command(self, name: str, view_only: bool):
        suffix = f"attach -t '{name}'"
        if view_only:
            suffix = f"{suffix} -r"
        ret = self.tmux_prepare_command(suffix)
        ret = f"{ENV} TMUX= {ret}"
        return ret

    def tmux_prepare_command_list(self, suffix_list: list[str]):
        ret = self.prefix_list + suffix_list
        return ret

    def tmux_get_command_output_bytes(self, suffix: str):
        cmd = self.tmux_prepare_command(suffix)
        ret = subprocess.check_output(cmd)
        return ret

    def tmux_execute_command(self, suffix: str):
        cmd = self.tmux_prepare_command(suffix)
        ret = self.execute_command(cmd)
        return ret

    def execute_command(self, cmd: str):
        print("[*] Executing command:", cmd)
        exitcode = os.system(cmd)
        ret = warn_nonzero_exitcode(exitcode)
        return ret

    def tmux_execute_command_list(self, suffix_list: list[str]):
        cmd_list = self.tmux_prepare_command_list(suffix_list)
        ret = self.execute_command_list(cmd_list)
        if not ret:
            print("[-] Failed to exit command list in tmux")
        return ret

    def execute_command_list(
        self, cmd_list: list[str], timeout: Optional[float] = CMDLIST_EXECUTE_TIMEOUT
    ):
        print("[*] Executing command list:", *cmd_list)
        exitcode = subprocess.Popen(cmd_list).wait(timeout=timeout)
        ret = warn_nonzero_exitcode(exitcode)
        return ret

    def apply_manifest(self, manifest: dict, attach=False):
        session_name = manifest["session_name"]
        print("[*] Tmuxp session name:", session_name)
        print("[*] Applying manifest:")
        with NamedTemporaryFile("w+", suffix=".json") as f:
            manifest_filepath = f.name
            content = json_pretty_print(manifest)
            f.write(content)
            f.flush()
            self.kill_session(session_name)
            try:
                self.tmuxp_load_from_filepath(manifest_filepath, attach)
            finally:
                print("[*] Removing session:", session_name)
                self.kill_session(session_name)

    def tmuxp_load_from_filepath(self, filepath: str, attach: bool):
        cmd_list = [TMUXP, "load", "-L", self.name, '-y', filepath]
        kwargs = {}
        if not attach:
            cmd_list.append("-d")
        else:
            kwargs["timeout"] = None
        ret = self.execute_command_list(cmd_list, **kwargs)
        if not ret:
            print("[-] Tmuxp manifest load failed")
        return ret


class TmuxSession:
    def __init__(
        self,
        name: str,
        server: TmuxServer,
        command: str,
        width=TMUX_WIDTH,
        height=TMUX_HEIGHT,
        isolate=True,
        kill_existing=True,
    ):
        self.name = name
        self.server = server
        if kill_existing:
            print(f"[*] Killing session '{name}' before creation")
            server.kill_session(name)
        success = server.tmux_execute_command(
            f"new-session -d -s {name} -x {width} -y {height} {command}"
        )
        if not success:
            raise TmuxSessionCreationFailure(
                f"Tmux session creation command failed"
            )
        if isolate:
            print(f"[*] Performing isolation for tmux session '{name}'")
            self.isolate()

    def isolate(self):
        self.set_option("prefix", "None")
        self.set_option("prefix2", "None")
        self.set_option("status", "off")
        self.set_option("aggressive-resize", "off")
        self.set_option("window-size", "manual")

    def preview_bytes(self):
        ret = self.server.tmux_get_command_output_bytes(
            f"capture-pane -t {self.name} -p"
        )
        return ret

    def preview(self, show_cursor=False):
        content_bytes = self.preview_bytes()
        if show_cursor:
            print("[*] Requested showing cursor")
            info = self.get_info()
            if info is None:
                print("[-] Failed to fetch tmux session info")
                return None
            else:
                x, y = info["cursor_x"], info["cursor_y"]
                print("[*] Inserting cursor at: %d, %d" % (x, y))
                content_byte_lines = content_bytes.splitlines()
                cursor_line_bytes = content_byte_lines[y - 1]
                content_byte_lines[y - 1] = insert_cursor(cursor_line_bytes, x - 1)
                content_bytes = NEWLINE_BYTES.join(content_byte_lines)
        ret = decode_bytes(content_bytes)
        return ret

    def kill(self):
        self.server.kill_session(self.name)
        del self

    def set_option(self, key: str, value: str):
        self.server.set_session_option(self.name, key, value)

    def get_info(self):
        list_session_format_template = "[#{session_name}] socket: #{socket_path} size: #{window_width}x#{window_height} cursor at: x=#{cursor_x},y=#{cursor_y} cursor flag: #{cursor_flag} cursor character: #{cursor_character}"
        output_bytes = self.server.tmux_get_command_output_bytes(
            f"list-sessions -F '{list_session_format_template}'"
        )
        numeric_properties = [
            "window_width",
            "window_height",
            "cursor_x",
            "cursor_y",
            "cursor_flag",
        ]
        parse_format = list_session_format_template.replace("#{", "{")
        for it in numeric_properties:
            parse_format = parse_format.replace(it, it + ":d")
        output = decode_bytes(output_bytes, errors='strict')
        data = parse.parse(parse_format, output)
        if data is not None:
            print("[+] Fetched info for session:", self.name)
            ret = dict(data)  # type: ignore
            json_pretty_print(ret)
            return ret
        else:
            print("[-] No info for session:", self.name)

    def create_viewer(self):
        ret = TmuxSessionViewer(self)
        return ret

    def view(self):
        viewer = self.create_viewer()
        viewer.view()


class TmuxEnv:
    def __init__(self, session: TmuxSession):
        self.session = session
        self.server = session.server

    def send_keys(self, keys: list[str]):
        command_list = ["send-keys", "-t", self.session.name, *keys]
        self.server.tmux_execute_command_list(command_list)

    def get_info(self):
        ret = self.session.get_info()
        return ret

    def kill(self):
        self.session.kill()
        del self


class TmuxSessionViewer:
    def __init__(self, session: TmuxSession):
        self.session = session
        self.server = session.server
        self.name = session.name + "_viewer"
        self.panes = []

    @property
    def manifest(self):
        ret = {
            "session_name": self.name,
            "windows": [
                    dict(name="viewer_window", layout="even-vertical", panes=self.panes)
                ],
            }
        return ret

    def add_viewer_pane(self, name: str, view_only=True):
        cmd = self.server.tmux_prepare_attach_command(self.session.name, view_only)
        self.add_cmd_pane(cmd, name)

    def add_cmd_pane(self, cmd: str, name: str):
        self.panes.append(dict(shell_command=cmd, name=name))

    def view(self, view_only=False):
        self.add_viewer_pane("EDIT" if not view_only else "VIEW_ONLY", view_only)
        self.server.apply_manifest(self.manifest, attach=True)


class TmuxSessionCreationFailure(Exception): ...
class TmuxEnvCreationFailure(Exception): ...
