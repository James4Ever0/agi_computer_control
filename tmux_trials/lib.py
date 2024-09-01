from typing import Optional, Iterable, TypedDict, Callable
from wcwidth import wcswidth
import subprocess
import os
import traceback
import shutil
import parse
import json
import time
from tempfile import NamedTemporaryFile
import uuid
from playwright.sync_api import sync_playwright
import pexpect
import threading
import numpy as np
from pyventus import AsyncIOEventEmitter
from strenum import StrEnum

import html
import re
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

TMUX_IDLE_SECONDS_THRESHOLD = 5
TMUX_IO_SPEED_CALCULATION_SCALE = (
    1,
    3,
    5,
    7,
    10,
)  # must be have a number equal or greater than TMUX_IDLE_SECONDS_THRESHOLD

TMUX_EVENT_WAIT_TIMEOUT = 20

CURSOR = "<<<CURSOR>>>"
CURSOR_END = "<<<CURSOR_END>>>"
HEAD = "<head>"
BODY_END = "</body>"

BLOCK_CSS_STYLE = "newDiv.style.backgroundColor = 'red';"

REQUIRED_BINARIES = (ENV, TMUX, TMUXP, AHA)

CURSOR_HTML = "<cursor>"
CURSOR_END_HTML = "</cursor>"
CURSOR_CHAR = "|"

HTML_TAG_REGEX = re.compile(r"<[^>]+?>")
Text = Union[str, bytes]
TERMINAL_VIEWPORT = {"width": 645, "height": 350}


def html_to_png(html: str, viewport=TERMINAL_VIEWPORT):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=viewport)  # type: ignore
        page = context.new_page()
        page.set_content(html)
        ret = page.screenshot()
        browser.close()
    return ret


def uuid_generator():
    ret = str(uuid.uuid4())
    ret = ret.replace("-", "")
    return ret


def remove_html_tags(html: str):
    ret = HTML_TAG_REGEX.sub("", html)
    return ret


def html_escape(content: Text):
    content = ensure_str(content)
    ret = html.escape(content)
    return ret


def ensure_bytes(content: Text) -> bytes:
    if isinstance(content, str):
        content = content.encode(ENCODING)
    return content


def ensure_str(content: Text) -> str:
    if isinstance(content, bytes):
        content = decode_bytes(content)
    return content  # type: ignore


def html_unescape(content: Text):
    content = ensure_str(content)
    ret = html.unescape(content)
    return ret


def line_with_cursor_to_html(
    line_with_cursor: Text,
    cursor: str,
    cursor_html=CURSOR_HTML,
    cursor_end_html=CURSOR_END_HTML,
):
    escaped_line = html_escape(line_with_cursor)
    ret = escaped_line.replace(cursor, cursor_html, 1).replace(
        cursor, cursor_end_html, 1
    )
    return ret


def diff_gen(it: str, prefix: str):
    ret = [f"{prefix} {char}" for char in it]
    return ret


def split_html(input_text: str) -> list[str]:
    # Find all matches of the regex pattern
    matches = HTML_TAG_REGEX.finditer(input_text)

    # Initialize the start index for slicing
    start = 0
    output = []

    for match in matches:
        # Get the start and end indices of the match
        match_start = match.start()
        match_end = match.end()

        # Split the input text at the current match
        output.append(input_text[start:match_start])
        output.append(input_text[match_start:match_end])

        # Update the start index for the next iteration
        start = match_end

    # Add the remaining text after the last match
    output.append(input_text[start:])

    # Filter empty text
    output = [it for it in output if it]

    return output


def tag_diff(source_with_tag: str) -> list[str]:
    # items = HTML_TAG_REGEX.split(source_with_tag)
    items = split_html(source_with_tag)
    print("[*] Items:", items)
    ret = []
    for it in items:
        if it.startswith("<"):
            ret.extend(diff_gen(it, "+"))
        else:
            ret.extend(diff_gen(it, " "))
    return ret


def get_original_char_index_in_diff_by_original_index(
    difflist: list[str], origin_index: int
):
    counter = 0
    ret = 0
    for index, it in enumerate(difflist):
        ret = index
        if counter == origin_index:
            break
        if it.startswith(" "):
            counter += 1
    if ret < origin_index:
        ret = -1
    return ret


def line_merger(line_with_cursor: str, line_with_span: str):
    cursor_diff = tag_diff(line_with_cursor)
    print("[*] Cursor diff:", cursor_diff)
    cursor_insert_index = cursor_diff.index("+ <")
    cursor_end_index = len(cursor_diff) - list(reversed(cursor_diff)).index("+ >") - 1
    cursor = line_with_cursor[cursor_insert_index : cursor_end_index + 1]
    cursor_range_diff = cursor_diff[cursor_insert_index : cursor_end_index + 1]
    is_block_cursor = cursor_range_diff.count("+ <") == 2
    cursor_ingested_chars = [it for it in cursor_range_diff if it.startswith(" ")]
    block_cursor_ingested_char_count = len(cursor_ingested_chars)
    # span_diff = list(differ.compare(source_line, line_with_span))
    span_diff = tag_diff(line_with_span)
    print("[*] Span diff:", span_diff)
    if is_block_cursor:
        if block_cursor_ingested_char_count == 1:
            print("[*] Origin index:", cursor_insert_index)
            span_diff_ingested_char_index = (
                get_original_char_index_in_diff_by_original_index(
                    span_diff, cursor_insert_index
                )
            )
            span_diff_chars = [it[2] for it in span_diff]
            print("[*] Span diff ingested char index:", span_diff_ingested_char_index)
            print("[*] Span diff char count:", len(span_diff_chars))
            if span_diff_ingested_char_index == -1:
                print("[*] Appending cursor to the end.")
                span_diff_chars.append(cursor)
            else:
                print(
                    "[*] Replacing cursor at span diff char #"
                    + str(span_diff_ingested_char_index)
                )
                try:
                    span_diff_chars[span_diff_ingested_char_index] = cursor
                except IndexError:
                    span_diff_chars.insert(span_diff_ingested_char_index, cursor)
            ret = "".join(span_diff_chars)
        elif block_cursor_ingested_char_count == 0:
            raise Exception("Error merging block cursor with zero ingested char count.")
        else:
            raise Exception(
                "Error merging block cursor with abnormal ingested char count:",
                block_cursor_ingested_char_count,
            )
    else:
        cursor_inserted = False
        original_char_index = 0
        ret = ""
        for it in span_diff:
            if original_char_index == cursor_insert_index:
                if not cursor_inserted:
                    cursor_inserted = True
                    ret += cursor
            if it.startswith(" "):
                original_char_index += 1
            ret += it[-1]
        if cursor_inserted == False:
            ret += cursor
    return ret


def ansi_to_html(ansi: bytes, dark_mode: bool):
    with NamedTemporaryFile("wb") as f:
        f.write(ansi)
        f.flush()
        cmdlist = [AHA, "-f", f.name]
        if dark_mode:
            cmdlist.append("--black")
        html = subprocess.check_output(cmdlist)
        return html


def html_to_soup(html: Text):
    soup = BeautifulSoup(html, "html.parser")
    return soup


def retrieve_pre_lines_from_html(html: Text):
    soup = html_to_soup(html)
    pre_elem = soup.find("pre")
    assert isinstance(pre_elem, Tag)
    ret = str(pre_elem)[5:-6]
    ret = ret.splitlines()[1:]
    return ret


def render_html_cursor(
    html: str,
    block_style: bool,
    block_css_style: str,
    cursor_char: str,
    cursor_html=CURSOR_HTML,
    cursor_end_html=CURSOR_END_HTML,
):
    if block_style:
        # force the cursor area has transparent background, and has higher z-index than all
        ret = html.replace(
            cursor_html,
            '<span id="cursor" style="background: none !important; position:relative; z-index: 2 !important;">',
        )
        ret = ret.replace(cursor_end_html, "</span>")
        # use javascript to place a red div with the same size of the cursor area, in between the cursor span and pre elem
        ret = ret.replace(
            HEAD,
            HEAD
            + """<script>
        // Wait for the page to load
        window.addEventListener('load', function() {
        // Get element by ID 'cursor'
        var cursorElement = document.getElementById('cursor');
        
        if(cursorElement) {
            // Get position, width and height of the element
            var rect = cursorElement.getBoundingClientRect();
            
            // Create a new div with red background
            var newDiv = document.createElement('div');
            newDiv.style.position = 'absolute';
            newDiv.style.left = rect.left + 'px';
            newDiv.style.top = rect.top + 'px';
            newDiv.style.width = rect.width + 'px';
            newDiv.style.height = rect.height + 'px';
            
            """
            + block_css_style
            + """
            newDiv.style.zIndex=1;

            // Add the newly created div directly into body
            document.body.appendChild(newDiv);
        }
        });
        </script>""".lstrip(),
        )
    else:
        ret = html.replace(
            cursor_html,
            f'<span style="filter: none !important; color: red !important; font-weight: bold !important;" id="cursor">{cursor_char}</span>',
        )
    return ret


def wrap_to_html_pre_elem(
    html: Text,
    pre_inner_html: str,
    grayscale: bool,
    block_style: bool,
    block_css_style: str,
    cursor_char: str,
    cursor_render: bool = True,
):
    soup = html_to_soup(html)
    soup.find("pre").extract()  # type: ignore
    ret = str(soup)
    if grayscale:
        ret = ret.replace(
            HEAD, HEAD + "<style> span {filter: grayscale(100%);} </style>"
        )
    ret = ret.replace(BODY_END, f"<pre>{pre_inner_html}</pre>" + BODY_END)
    if cursor_render:
        ret = render_html_cursor(
            ret,
            block_style=block_style,
            block_css_style=block_css_style,
            cursor_char=cursor_char,
        )
    return ret


def decode_bytes(_bytes: bytes, errors="ignore"):
    ret = _bytes.decode(ENCODING, errors=errors)
    return ret


def insert_cursor(
    content: Text, x: int, block_style: bool, cursor=CURSOR, cursor_end=CURSOR_END
):
    _bytes = ensure_bytes(content)
    ret = b""
    cursor_bytes = cursor.encode(ENCODING)
    cursor_end_bytes = cursor_end.encode(ENCODING)
    try:
        line = decode_bytes(_bytes, errors="replace")
        char_index = 0
        for index, it in enumerate(line):
            if char_index >= x:
                if len(line) < index:
                    line += " " * (len(line) - index)
                if block_style:
                    if len(line) < index + 1:
                        line += " "
                    ret = (
                        line[:index]
                        + cursor
                        + line[index]
                        + cursor_end
                        + line[index + 1 :]
                    )
                else:
                    ret = line[:index] + cursor + line[index:]
                ret = ret.encode(ENCODING)
                break
            char_width = wcswidth(it)
            char_index += char_width
        if ret == b"":
            if block_style:
                ret = _bytes + cursor_bytes + b" " + cursor_end_bytes
            else:
                ret = _bytes + cursor_bytes
    except UnicodeDecodeError:
        print("[-] Failed to decode line while inserting cursor:", _bytes)
        print("[*] Falling back to bytes insert mode")
        if block_style:
            if len(_bytes) < x + 1:
                _bytes += b" " * (x + 1 - len(_bytes))
            ret = (
                _bytes[:x]
                + cursor_bytes
                + bytes([_bytes[x]])
                + cursor_end_bytes
                + _bytes[x + 1 :]
            )
        else:
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
        ret = self.tmux_execute_command("kill-server")
        if ret:
            print(f"[+] Server '{self.name}' reset complete")
        else:
            print(f"[-] Server '{self.name}' reset failed")
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

    def resize_session_size(self, name: str, x: int, y: int):
        self.tmux_execute_command(f"resize-window -t {name} -x {x} -y {y}")

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
        self.width = width
        self.height = height
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

    def prepare_attach_command(self, view_only: bool):
        ret = self.server.tmux_prepare_attach_command(self.name, view_only)
        return ret

    def isolate(self):
        self.set_option("prefix", "None")
        self.set_option("prefix2", "None")
        self.set_option("status", "off")
        self.set_option("aggressive-resize", "off")
        self.set_option("window-size", "manual")

    def preview_png(
        self,
        show_cursor=False,
        filename: Optional[str] = None,
        dark_mode=False,
        grayscale=False,
        block_style=False,
        cursor_char=CURSOR_CHAR,
        block_css_style=BLOCK_CSS_STYLE,
    ):
        html = self.preview_html(
            show_cursor=show_cursor,
            wrap_html=True,
            dark_mode=dark_mode,
            grayscale=grayscale,
            block_style=block_style,
            cursor_char=cursor_char,
            block_css_style=block_css_style,
        )
        png_bytes = html_to_png(html)
        if filename:
            with open(filename, "wb") as f:
                f.write(png_bytes)
        return png_bytes

    def preview_bytes(self, flags: list[str] = ["-p"]):
        ret = self.server.tmux_get_command_output_bytes(
            ["capture-pane", "-t", self.name, *flags]
        )
        return ret

    def preview_html_bytes(self, dark_mode: bool):
        ret = self.preview_bytes(flags=["-p", "-e"])
        ret = ansi_to_html(ret, dark_mode)
        return ret

    def preview_html(
        self,
        show_cursor=False,
        wrap_html=False,
        dark_mode=False,
        grayscale=False,
        block_style=False,
        cursor_char=CURSOR_CHAR,
        block_css_style=BLOCK_CSS_STYLE,
    ):
        has_cursor, (x, y) = self.get_cursor_coordinates()
        html_bytes = self.preview_html_bytes(dark_mode)
        pre_lines = retrieve_pre_lines_from_html(html_bytes)
        if show_cursor:
            if has_cursor:
                cursor_line_html = pre_lines[y]
                print("[*] Cursor line html:", cursor_line_html)
                cursor_line_html_without_tags = remove_html_tags(cursor_line_html)
                print(
                    "[*] Cursor line html without tags:", cursor_line_html_without_tags
                )
                cursor_line = html_unescape(cursor_line_html_without_tags)
                print("[*] Cursor line:", cursor_line)
                uuid_cursor = uuid_generator()
                cursor_line_bytes_with_uuid_cursor = insert_cursor(
                    cursor_line,
                    x,
                    cursor=uuid_cursor,
                    cursor_end=uuid_cursor,
                    block_style=block_style,
                )
                print("[*] Inserting cursor:", cursor_line_bytes_with_uuid_cursor)
                cursor_line_html_with_cursor = line_with_cursor_to_html(
                    cursor_line_bytes_with_uuid_cursor, uuid_cursor
                )
                print("[*] Replacing cursor:", cursor_line_html_with_cursor)
                merged_line = line_merger(
                    # cursor_line_html_without_tags,
                    cursor_line_html_with_cursor,
                    cursor_line_html,
                )
                print("[*] Merged line:", merged_line)
                pre_lines[y] = merged_line
        ret = NEWLINE.join(pre_lines)
        if wrap_html:
            ret = wrap_to_html_pre_elem(
                html_bytes,
                ret,
                cursor_char=cursor_char,
                block_css_style=block_css_style,
                grayscale=grayscale,
                block_style=block_style,
            )
        return ret

    def get_cursor_coordinates(self):
        print("[*] Requesting cursor coordinates")
        has_cursor = False
        coordinates = (-1, -1)
        info = self.get_info()
        if info is None:
            print("[-] Failed to fetch corsor coordinates")
        else:
            x, y = info["cursor_x"], info["cursor_y"]
            print("[*] Cursor at: %d, %d" % (x, y))
            coordinates = (x, y)
            has_cursor = True
        return has_cursor, coordinates

    def preview(self, show_cursor=False, block_style=False):
        has_cursor, (x, y) = self.get_cursor_coordinates()
        content_bytes = self.preview_bytes()
        if show_cursor:
            if has_cursor:
                content_byte_lines = content_bytes.splitlines()
                cursor_line_bytes = content_byte_lines[y]
                content_byte_lines[y] = insert_cursor(
                    cursor_line_bytes, x, block_style=block_style
                )
                content_bytes = NEWLINE_BYTES.join(content_byte_lines)
        ret = decode_bytes(content_bytes)
        return ret

    def kill(self):
        self.server.kill_session(self.name)
        del self

    def resize(self, x: int, y: int):
        self.server.resize_session_size(self.name, x, y)

    def resize_to_default(self):
        self.resize(self.width, self.height)

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

    def create_viewer(self, *args, **kwargs):
        ret = TmuxSessionViewer(self, *args, **kwargs)
        return ret

    def view(self):
        viewer = self.create_viewer()
        viewer.view()


class ProcessEvent(StrEnum):
    active = "ProcessActive"
    idle = "ProcessIdle"
    output = "ProcessOutput"
    exit = "ProcessExit"


class EventManager:
    def __init__(self):
        self.emitter = AsyncIOEventEmitter()
        self.event_linker = self.emitter._event_linker

    def on(self, event_name: str, callback: Callable):
        self.event_linker.on(event_name)(callback)

    def emit_classic(self, event, *args, **kwargs):
        self.emitter.emit(event, *args, **kwargs)

    def emit(self, event, *args, **kwargs):
        args = (event, *args)
        self.emit_classic(event, *args, **kwargs)


class ProcessEventWaitTimeout(Exception): ...


class ProcessEventWatcher:
    def __init__(
        self,
        command: str,
        speed_intervals: list[int],
        idle_threshold: int,
        watch_interval=1,
    ):
        self.watch_interval = watch_interval
        self.command = command
        self.process = pexpect.spawn(command, timeout=None)
        self.speed_intervals = list(speed_intervals)
        self.speed_intervals.sort()
        self.idle_threshold = self.calculate_idle_threshold(idle_threshold)
        self.event_manager = EventManager()
        self.add_activity_event_listeners()
        self.stats = {
            "count": 0,
            "bytes": b"",
            "io_speed": {f"{it}s": 0 for it in speed_intervals},
        }
        self.datapoints = []
        self.process_idle = False
        self.maxpoints = max(speed_intervals) + 1

    def get_process_info(self):
        info_items = []
        for k, v in self.info.items():
            info_items.append(f"{k.title()}: {v}")
        info = " ".join(info_items)
        return info

    @property
    def info(self):
        ret = dict(
            pid=self.process.pid,
            command=self.command,
            idle=self.process_idle,
            stats=self.stats,
        )
        return ret

    def activity_callback(self, name):
        print("[*]", self.get_process_info(), f"Event: {name}")

    def add_activity_event_listeners(self):
        self.on_idle(self.activity_callback)
        self.on_active(self.activity_callback)

    def calculate_idle_threshold(self, idle_threshold: int):
        for it in self.speed_intervals:
            if it >= idle_threshold:
                print(f"[*] Calculated idle threshold: {it} sec(s)")
                return it
        raise Exception(
            f"Unable to find a suitable idle threshold {idle_threshold} within speed intervals {self.speed_intervals}"
        )

    def update_stats(self):
        count = 0
        while self.process.isalive():
            one_byte = self.process.read(1)
            count += 1
            self.stats["count"] = count
            self.stats["bytes"] += one_byte

    def calculate_nth_average_speed(self, nth: int):
        slice_length = min(len(self.datapoints), nth)
        diff = np.diff(self.datapoints)
        ret = sum(diff[:slice_length]) / nth
        ret = float(-ret)
        return ret

    def wait_for_process_state(
        self, idle: bool, timeout: float, confirmation_threshold=1, loop_interval=1
    ):
        if idle:
            inverse_state = False
        else:
            inverse_state = True
        elapsed_time = 0
        confirmation_count = 0
        while True:
            if self.process_idle == idle:
                confirmation_count += 1
            else:
                confirmation_count = 0
            if confirmation_count >= confirmation_threshold:
                break
            if elaspsed_time >= timeout:
                raise ProcessStateWaitTimeout(
                    f"[-] Failed to wait for process state (idle: {idle}) within {timeout} sec(s) timeout limit"
                )
            time.sleep(loop_interval)
            elapsed_time += loop_interval

    def wait_for_idle_state(self, timeout: float):
        self.wait_for_process_state(True, timeout)

    def wait_for_active_state(self, timeout: float):
        self.wait_for_process_state(False, timeout)

    def on_idle(self, callback: Callable):
        self.event_manager.on(ProcessEvent.idle, callback)

    def on_active(self, callback: Callable):
        self.event_manager.on(ProcessEvent.active, callback)

    def on_output(self, callback: Callable):
        self.event_manager.on(ProcessEvent.output, callback)

    def on_exit(self, callback: Callable):
        self.event_manager.on(ProcessEvent.exit, callback)

    def handle_process_output(self):
        read_bytes = self.stats.get("bytes")
        self.stats["bytes"] = b""  # clear the clutter
        self.event_manager.emit(ProcessEvent.output, data=read_bytes)

    def update_datapoints(self):
        count = self.stats.get("count")

        self.datapoints.insert(0, count)
        if len(self.datapoints) > self.maxpoints:
            self.datapoints = self.datapoints[: self.maxpoints]

    def update_speed(self):
        for it in self.speed_intervals:
            speed = self.calculate_nth_average_speed(it)
            self.stats["io_speed"][f"{it}s"] = speed

    def watch_once(self):
        self.handle_process_output()
        self.update_datapoints()
        self.update_speed()
        self.update_process_idle_state()

    def update_process_idle_state(self):
        idle_threshold_speed = self.stats["io_speed"][f"{self.idle_threshold}s"]

        if idle_threshold_speed > 0:
            if self.process_idle:
                self.process_idle = False
                self.emit(ProcessEvent.active)
        else:
            if not self.process_idle:
                self.process_idle = True
                self.emit(ProcessEvent.idle)

    def watch(self):
        self.start_daemon_thread(self.update_stats)
        while self.process.isalive():
            time.sleep(self.watch_interval)
            self.watch_once()

        self.handle_process_exit()

    def handle_process_exit(self):
        print("[*] Process exited with status code:", self.process.status)
        self.event_manager.emit(ProcessEvent.exit, status=self.process.status)

    def emit(self, event: ProcessEvent, *args, **kwargs):
        self.event_manager.emit(event, *args, **kwargs)

    def watch_in_background(self):
        self.start_daemon_thread(self.watch)

    @staticmethod
    def start_daemon_thread(target: Callable, *args, **kwargs):
        t = threading.Thread(target=target, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()


class TmuxEventWatcher(ProcessEventWatcher):
    def __init__(
        self,
        session: TmuxSession,
        speed_intervals=TMUX_IO_SPEED_CALCULATION_SCALE,
        idle_threshold=TMUX_IDLE_SECONDS_THRESHOLD,
    ):
        command = session.prepare_attach_command(view_only=True)
        super().__init__(
            command=command,
            speed_intervals=speed_intervals,
            idle_threshold=idle_threshold,
        )


class TmuxEnvironment:
    def __init__(self, session: TmuxSession, wait_timeout=TMUX_EVENT_WAIT_TIMEOUT):
        self.session = session
        self.server = session.server
        self.wait_timeout = wait_timeout
        self.watcher = TmuxEventWatcher(session)
        self.watcher.watch_in_background()

    @property
    def stats(self):
        return self.watcher.stats

    @property
    def info(self):
        return self.watcher.info

    def wait_for_idle_state(self):
        self.watcher.wait_for_idle_state(self.wait_timeout)

    def wait_for_active_state(self):
        self.watcher.wait_for_active_state(self.wait_timeout)

    def on_active(self, callback: Callable):
        self.watcher.on(ProcessEvent.active, callback)

    def on_idle(self, callback: Callable):
        self.watcher.on(ProcessEvent.idle, callback)

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


class TmuxWindow(TypedDict):
    layout: str
    panes: list[dict]


class TmuxSessionViewer:
    def __init__(
        self,
        session: TmuxSession,
        default_layout="even-horizontal",
        default_window_name="viewer_window",
    ):
        self.session = session
        self.server = session.server
        self.name = session.name + "_viewer"
        self.default_window_name = default_window_name
        self.windows: dict[str, TmuxWindow] = {}
        self.default_layout = default_layout
        self.add_new_window(self.default_window_name)

    def add_new_window(self, window_name: str, layout: Optional[str] = None):
        self.windows[window_name] = dict(  # type:ignore
            layout=self.default_layout, panes=[]
        )
        if layout is not None:
            self.modify_window_layout(window_name, layout)

    def modify_window_layout(self, window_name: str, layout: str):
        self.windows[window_name]["layout"] = layout

    def get_or_create_window(self, window_name: Optional[str]):
        if window_name is None:
            window_name = self.default_window_name
        if window_name not in self.windows:
            self.add_new_window(window_name)
        return self.windows[window_name]

    @property
    def manifest(self):
        ret = {
            "session_name": self.name,
            "windows": [dict(name=k, **v) for k, v in self.windows.items()],
        }
        return ret

    def add_viewer_pane(
        self, pane_name: str, window_name: Optional[str] = None, view_only=True
    ):
        cmd = self.session.prepare_attach_command(view_only)
        self.add_cmd_pane(cmd, pane_name, window_name=window_name)

    def add_cmd_pane(self, cmd: str, pane_name: str, window_name: Optional[str] = None):
        window = self.get_or_create_window(window_name)
        window["panes"].append(dict(shell_command=cmd, name=pane_name))

    def view(self, view_only=False):
        self.session.resize_to_default()
        pane_name = "VIEW_EDIT" if not view_only else "VIEW_ONLY"
        self.add_viewer_pane(pane_name, view_only=view_only)
        self.server.apply_manifest(self.manifest, attach=True)


class TmuxSessionCreationFailure(Exception): ...


class TmuxEnvironmentCreationFailure(Exception): ...
