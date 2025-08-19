import subprocess

# TODO: use libvnc bindings
# TODO: use tigervnc bindings

CONTAINER_NAME = "gui_recorder_novnc"

# The user has to be root or in the docker group

class X11VNCDockerExecutor:
    def __init__(self, container_name: str = CONTAINER_NAME):
        self.container_name = container_name

    def execute_docker_command_detached(self, command_args: list[str]):
        subprocess.call(["docker", "exec", "-d", self.container_name] + command_args)

    def execute_python_oneliner(self, oneliner_script: str):
        self.execute_docker_command_detached(["python3", "-c", oneliner_script])

    def execute_pynput_oneliner(self, pynput_oneliner: str):
        self.execute_python_oneliner(f"import pynput; {pynput_oneliner}")

    # handle the cases of keyboard and mouse input events
    # the input format shall be exactly the same as recorded event stream, so we can replay the events

    # mouse events:
    # - `{"event": "mouse_move", "x": <x: int>, "y": <y: int>, "timestamp": <timestamp: float>}`
    def execute_mouse_move(self, x: int, y: int):
        self.execute_pynput_oneliner(f"pynput.mouse.Controller().position = ({x}, {y})")

    # - `{"event": "mouse_click", "x": <x: int>, "y": <y: int>, "button": <button: str>, "pressed": <pressed: bool>, "timestamp": <timestamp: float>}`

    @staticmethod
    def get_mouse_button_fullname(
        button: str,
    ):  # convert the mouse button name to the full name of the button class
        if button.startswith("Button."):
            button_fullname = f"pynput.mouse.{button}"
        elif button.startswith("pynput.mouse."):
            button_fullname = button
        elif button.startswith("mouse.Button."):
            button_fullname = f"pynput.{button}"
        elif button in ["left", "right", "middle"]:
            button_fullname = f"pynput.mouse.Button.{button}"
        else:
            raise ValueError(f"Unknown pynput mouse button: {button}")
        return button_fullname

    def execute_mouse_click(self, x: int, y: int, button: str, pressed: bool):
        button_fullname = self.get_mouse_button_fullname(button)
        if pressed:
            self.execute_pynput_oneliner(
                f"pynput.mouse.Controller().press(%s)" % button_fullname
            )
        else:
            self.execute_pynput_oneliner(
                f"pynput.mouse.Controller().release(%s)" % button_fullname
            )

    # - `{"event": "mouse_scroll", "x": <x: int>, "y": <y: int>, "dx": <dx: int>, "dy": <dy: int>,  "timestamp": <timestamp: float>}`
    def execute_mouse_scroll(self, x: int, y: int, dx: int, dy: int):
        self.execute_mouse_move(x, y)
        self.execute_pynput_oneliner(f"pynput.mouse.Controller().scroll({dx}, {dy})")

    @staticmethod
    def get_keyboard_key_fullname(key: str):
        if key.startswith("Key."):
            key_fullname = f"pynput.keyboard.{key}"
        elif key.startswith("keyboard.Key."):
            key_fullname = f"pynput.{key}"
        elif key.startswith("'"):
            key_fullname = key
        elif key.startswith('"'):
            key_fullname = key
        else:
            raise ValueError(f"Unknown pynput keyboard key: {key}")

        return key_fullname

    # keyboard events:
    # - `{"event": "key_press", "key": <key: str>, "timestamp": <timestamp: float>}`
    def execute_key_press(self, key: str):
        key_fullname = self.get_keyboard_key_fullname(key)
        self.execute_pynput_oneliner(
            f"pynput.keyboard.Controller().press({key_fullname})"
        )

    # - `{"event": "key_release", "key": <key: str>, "timestamp": <timestamp: float>}`
    def execute_key_release(self, key: str):
        key_fullname = self.get_keyboard_key_fullname(key)
        self.execute_pynput_oneliner(
            f"pynput.keyboard.Controller().release({key_fullname})"
        )

    def take_screenshot(self):
        # use docker python one-liner to take screenshot at /tmp/x11vnc_executor_screenshot.png, and retrieve the file to given destination using docker cp
        container_screenshot_tmp_path = "/tmp/x11vnc_executor_screenshot.png"
        screenshot_python_onliner = (
            "import mss; sct = mss.mss(); sct.shot(output='%s')"
            % container_screenshot_tmp_path
        )
        self.execute_python_oneliner(screenshot_python_onliner)
        self.execute_docker_command_detached(
            [
                "docker",
                "cp",
                f"{self.container_name}:{container_screenshot_tmp_path}",
                ".",
            ]
        )
