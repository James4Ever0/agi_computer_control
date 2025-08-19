import pynput

# mouse events:
# - `{"event": "mouse_move", "x": <x: int>, "y": <y: int>, "timestamp": <timestamp: float>}`
pynput.mouse.Controller().position = (100, 100)
# - `{"event": "mouse_click", "x": <x: int>, "y": <y: int>, "button": <button: str>, "pressed": <pressed: bool>, "timestamp": <timestamp: float>}`
pynput.mouse.Controller().press(pynput.mouse.Button.left)
pynput.mouse.Controller().release(pynput.mouse.Button.left)
# - `{"event": "mouse_scroll", "x": <x: int>, "y": <y: int>, "dx": <dx: int>, "dy": <dy: int>,  "timestamp": <timestamp: float>}`
pynput.mouse.Controller().move(0, 0)
pynput.mouse.Controller().scroll(0, 1)

# keyboard events:
# - `{"event": "key_press", "key": <key: str>, "timestamp": <timestamp: float>}`
pynput.keyboard.Controller().press(pynput.keyboard.Key.enter)
# - `{"event": "key_release", "key": <key: str>, "timestamp": <timestamp: float>}`
pynput.keyboard.Controller().release(pynput.keyboard.Key.enter)