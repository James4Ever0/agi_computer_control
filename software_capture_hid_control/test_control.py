# TODO: more control methods (non-hardware) under way
# vnc/rdp (rdpy (py2), rdpy3, python3-aardwolf, rdesktop (rdp)) (docker-vnc image: dorowu/ubuntu-desktop-lxde-vnc:focal ; docker rdp image: scottyhardy/docker-remote-desktop)
# ssh (terminal interface)
# spice (remmina, remote-viewer (RHEL))
# xvfb (with pyautogui?) (use vglrun (GPU)) (what alternatives to xvfb are for macOS and Windows?)
# -----------[use remote control methods as self control methods]-----------
# self control (pyautogui, pynput, (win)tty, tmux, subprocess, ttyd with electron/xvfb based browser client)

# qtpy: pyqt4/5/6 abstraction layer
# https://github.com/spyder-ide/qtpy

# docker-wine image (in case running windows app on linux): scottyhardy/docker-wine

# MineRL GPU rendering: https://minerl.readthedocs.io/en/latest/notes/performance-tips.html

# rdpy3: https://github.com/massimiliano-dalcero/rdpy
# ref: https://github.com/citronneur/rdpy/issues/91

# shall you look over our previous project lazero/metalazero

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum
from enum import auto

# TODO: test this under py3.9/3.10

class ControlMethod(StrEnum):
    xvfb = auto()
# breakpoint()

controlMethod = ControlMethod.xvfb

if controlMethod == ControlMethod.xvfb:
    import pyautogui
    from pyvirtualdisplay import Display
    import easyprocess
    import time
    # virtual_display = ":3"
    with Display(backend='xvfb') as disp:
        print("NEW DISPLAY AT", disp.display) # 0, INT
        # with Display(backend='xvfb') as disp2:
        #     print("NEW DISPLAY AT", disp2.display) # 2
        easyprocess.EasyProcess('gnome-terminal')
        pyautogui.write("echo hello world")
        pyautogui.screenshot("terminal.png")