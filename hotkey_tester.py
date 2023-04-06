import pyautogui
import pynput

# you simply deprecate all hotkeys.

keyboard_controller = pynput.keyboard.Controller()

keyboard_controller.press(pynput.keyboard.Key.ctrl)
pyautogui.typewrite('1')
keyboard_controller.release(pynput.keyboard.Key.ctrl)
