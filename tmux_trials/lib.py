from typing import Optional, Iterable
from wcwidth import wcswidth
import subprocess
import os
import traceback
import shutil
import parse
import json
from tempfile import NamedTemporaryFile
import uuid
import difflib
import html
from typing import Union
from bs4 import BeautifulSoup, Tag

ENV = "env"
TMUX = "tmux"
TMUXP = "tmuxp"
AHA = "aha"
TMUX_WIDTH = 80
TMUX_HEIGHT = 24
ENCODING = "utf-8"

NEWLINE = "\n"
NEWLINE_BYTES = NEWLINE.encode(ENCODING)

CMDLIST_EXECUTE_TIMEOUT = 10

CURSOR = "<<<CURSOR>>>"
BODY = "<body>"

REQUIRED_BINARIES = (ENV, TMUX, TMUXP, AHA)

CURSOR_HTML = "<cursor>"

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
Content = Union[str, bytes]


def uuid_generator():
    ret = str(uuid.uuid4())
    ret = ret.replace("-", "")
    return ret


def remove_html_tags(html: str):
    ret = HTML_TAG_REGEX.replace(html, "")
    return ret


def html_escape(content: Content):
    content = ensure_str(content)
    ret = html.escape(content)
    return ret


def ensure_str(content: Content):
    if isinstance(content, bytes):
        content = decode_bytes(content)
    return content


def html_unescape(content: Content):
    content = ensure_str(content)
    ret = html.unescape(content)
    return ret


def line_with_cursor_to_html(
    line_with_cursor: Content, cursor: str, cursor_html=CURSOR_HTML
):
    escaped_line = html_escape(line_with_cursor)
    ret = escaped_line.replace(cursor, cursor_html)
    return ret


def line_merger(source_line: str, line_with_cursor: str, line_with_span: str):
    differ = difflib.Differ()
    cursor_diff = list(differ.compare(source_line, line_with_cursor))
    cursor_insert_index = cursor_diff.index("+ <")
    cursor_end_index = cursor_diff.index("+ >")
    cursor = line_with_cursor[cursor_insert_index : cursor_end_index + 1]
    span_diff = list(differ.compare(source_line, line_with_span))
    cursor_inserted = False
    original_char_index = 0
    ret = ""
    for it in span_diff:
        if original_char_index == cursor_insert_index:
            cursor_inserted = True
            ret += cursor
        if it.startswith(" "):
            original_char_index += 1
        ret += it[-1]
    if cursor_inserted == False:
        ret += cursor
    return ret


def ansi_to_html(ansi: bytes):
    with NamedTemporaryFile("wb") as f:
        f.write(ansi)
        f.flush()
        html = subprocess.check_output([AHA, "-f", f.name])
        return html


def html_to_soup(html: Content):
    soup = BeautifulSoup(html, "html.parser")
    return soup


def retrieve_pre_lines_from_html(html: Content):
    soup = html_to_soup(html)
    pre_elem = soup.find("pre")
    assert isinstance(pre_elem, Tag)
    ret = str(pre_elem)[5:-6]
    ret = ret.splitlines()[1:]
    return ret


def wrap_to_html_pre_elem(html: Content, pre_inner_html: str):
    soup = html_to_soup(html)
    soup.find("pre").extract()  # type: ignore
    ret = str(soup)
    ret = ret.replace(BODY, BODY + f"<pre>{pre_inner_html}</pre>", count=1)
    return ret


def decode_bytes(_bytes: bytes, errors="ignore"):
    ret = _bytes.decode(ENCODING, errors=errors)
    return ret


def insert_cursor(_bytes: bytes, x: int, cursor=CURSOR):
    ret = b""
    cursor_bytes = cursor.encode(ENCODING)
    try:
        line = decode_bytes(_bytes, errors="replace")
        char_index = 0
        for index, it in enumerate(line):
            if char_index >= x:
                ret = line[:index] + cursor + line[index:]
                ret = ret.encode(ENCODING)
                break
            char_width = wcswidth(it)
            char_index += char_width
        if ret == b"":
            ret = _bytes + cursor_bytes
    except UnicodeDecodeError:
        print("[-] Failed to decode line while inserting cursor:", _bytes)
        print("[*] Falling back to bytes insert mode")
        ret = _bytes[:x] + cursor_bytes + _bytes[x:]
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

    def kill(self):
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
            ret = TmuxEnvironment(session)
            print("[+] Tmux env created")
            return ret
        else:
            raise TmuxEnvironmentCreationFailure("[-] Failed to create tmux env")

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

    def tmux_get_command_output_bytes(self, suffix_list: list[str]):
        cmdlist = self.tmux_prepare_command_list(suffix_list)
        ret = subprocess.check_output(cmdlist)
        return ret

    def tmux_execute_command(self, suffix: str):
        cmd = self.tmux_prepare_command(suffix)
        ret = self.execute_command(cmd)
        return ret

    @staticmethod
    def execute_command(cmd: str):
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

    @staticmethod
    def execute_command_list(
        cmd_list: list[str], timeout: Optional[float] = CMDLIST_EXECUTE_TIMEOUT
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
        cmd_list = [TMUXP, "load", "-L", self.name, "-y", filepath]
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
            raise TmuxSessionCreationFailure(f"Tmux session creation command failed")
        if isolate:
            print(f"[*] Performing isolation for tmux session '{name}'")
            self.isolate()

    def isolate(self):
        self.set_option("prefix", "None")
        self.set_option("prefix2", "None")
        self.set_option("status", "off")
        self.set_option("aggressive-resize", "off")
        self.set_option("window-size", "manual")

    def preview_bytes(self, flags: list[str] = ["-p"]):
        ret = self.server.tmux_get_command_output_bytes(
            ["capture-pane", "-t", self.name, *flags]
        )
        return ret

    def preview_html_bytes(self):
        ret = self.preview_bytes(flags=["-p", "-e"])
        ret = ansi_to_html(ret)
        return ret

    def preview_html(self, show_cursor=False, wrap_html=False):
        html_bytes = self.preview_html_bytes()
        pre_lines = retrieve_pre_lines_from_html(html_bytes)
        if show_cursor:
            has_cursor, (x, y) = self.get_cursor_coordinates()
            if has_cursor:
                cursor_line_html = pre_lines[y]
                cursor_line_html_without_tags = remove_html_tags(cursor_line_html)
                cursor_line_bytes_without_tags = html_unescape(
                    cursor_line_html_without_tags
                )
                uuid_cursor = uuid_generator()
                cursor_line_bytes_with_uuid_cursor = insert_cursor(
                    cursor_line_bytes_without_tags, x, uuid_cursor
                )
                cursor_line_html_with_cursor = line_with_cursor_to_html(
                    cursor_line_bytes_with_uuid_cursor, uuid_cursor
                )
                pre_lines[y] = line_merger(
                    cursor_line_html_without_tags,
                    cursor_line_html_with_cursor,
                    cursor_line_html,
                )
        ret = NEWLINE.join(pre_lines)
        if wrap_html:
            ret = wrap_to_html_pre_elem(html_bytes, ret)
        return ret

    def get_cursor_coordinates(self):
        print("[*] Requesting cursor coordinates")
        has_cursor = False
        coordinates = None
        info = self.get_info()
        if info is None:
            print("[-] Failed to fetch corsor coordinates")
        else:
            x, y = info["cursor_x"], info["cursor_y"]
            print("[*] Cursor at: %d, %d" % (x, y))
            coordinates = (x, y)
            has_cursor = True
        return has_cursor, coordinates

    def preview(self, show_cursor=False):
        content_bytes = self.preview_bytes()
        if show_cursor:
            has_cursor, (x, y) = self.get_cursor_coordinates()
            if has_cursor:
                content_byte_lines = content_bytes.splitlines()
                cursor_line_bytes = content_byte_lines[y]
                content_byte_lines[y] = insert_cursor(cursor_line_bytes, x)
                content_bytes = NEWLINE_BYTES.join(content_byte_lines)
        ret = decode_bytes(content_bytes)
        return ret

    def kill(self):
        self.server.kill_session(self.name)
        del self

    def set_option(self, key: str, value: str):
        self.server.set_session_option(self.name, key, value)

    def get_info(self):
        list_session_format_template = "[#{session_name}] socket: #{socket_path} size: #{window_width}x#{window_height} cursor at: x=#{cursor_x},y=#{cursor_y} cursor flag: #{cursor_flag} cursor character: #{cursor_character} insert flag: #{insert_flag}, keypad cursor flag: #{keypad_cursor_flag}, keypad flag: #{keypad_flag}"
        session_filter = "#{==:#{session_name}," + self.name + "}"
        output_bytes = self.server.tmux_get_command_output_bytes(
            ["list-sessions", "-F", list_session_format_template, "-f", session_filter]
        )
        # print("[*] Output bytes:")
        # print(output_bytes)
        numeric_properties = [
            "window_width",
            "window_height",
            "cursor_x",
            "cursor_y",
            "cursor_flag",
            "insert_flag",
            "keypad_cursor_flag",
            "keypad_flag",
        ]
        # nonword_properties = ['cursor_character', 'socket_path']
        parse_format = list_session_format_template.replace(
            "#{", "{"
        )  # .replace("}",":w}")
        # for it in nonword_properties:
        #     parse_format = parse_format.replace("{"+it+":w}","{"+it+"}")
        output = decode_bytes(output_bytes, errors="strict")
        output = output[:-1]  # strip trailing newline
        # print("[*] Parse format:")
        # print(parse_format)
        data = parse.parse(parse_format, output)
        if isinstance(data, parse.Result):
            print("[+] Fetched info for session:", self.name)
            ret = data.named
            # print(ret)
            for it in numeric_properties:
                ret[it] = int(ret[it])  # type: ignore
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


class TmuxEnvironment:
    def __init__(self, session: TmuxSession):
        self.session = session
        self.server = session.server

    def send_key_list(self, key_list: list[str]):
        command_list = ["send-keys", "-t", self.session.name, *key_list]
        self.server.tmux_execute_command_list(command_list)

    def send_key(self, key: str):
        self.send_key_list([key])

    def get_info(self):
        ret = self.session.get_info()
        return ret

    def kill(self):
        self.session.kill()
        del self


class TmuxSessionViewer:
    def __init__(self, session: TmuxSession, layout="even-horizontal"):
        self.session = session
        self.server = session.server
        self.name = session.name + "_viewer"
        self.window_name = "viewer_window"
        self.layout = layout
        self.panes = []

    @property
    def manifest(self):
        ret = {
            "session_name": self.name,
            "windows": [
                dict(name=self.window_name, layout=self.layout, panes=self.panes)
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


class TmuxEnvironmentCreationFailure(Exception): ...
