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
    # do not use pyautogui.

    # instead use:
    # [xdotool](https://github.com/jordansissel/xdotool)
    # [python-libxdo](https://pypi.org/project/python-libxdo/)
    # [xdotool python wrapper](https://github.com/Tlaloc-Es/xdotool)
    # [python-xdoctl](https://pypi.org/project/python-xdoctl/)
    # [pyxdotool](https://github.com/cphyc/pyxdotool)
    # -------------[AND NOW FOR SOMETHING COMPLETELY DIFFERENT]---------------
    # [bezmouse](https://github.com/vincentbavitz/bezmouse) might help you evade bot checks, but it is up to you to **compress** user mouse coordinates. maybe just average out tracks per action frame? you name it.
    # also compress key events?
    # another story please...

    # from pyvirtualdisplay import Display
    from pyvirtualdisplay.smartdisplay import SmartDisplay
    import easyprocess
    import time
    import os
    import subprocess
    os.system("rm *.png")
    # nah, not working...
    # from pynput.keyboard import Controller
    # keyboard = Controller()
    # virtual_display = ":3"
    with SmartDisplay(backend='xvfb') as disp:
        # with Display(backend='xvfb') as disp:
        # with Display(visible=False) as disp:
        print("NEW DISPLAY AT", disp.display)  # 0, INT
        print("ENV DISPLAY?", os.environ["DISPLAY"])  # :0
        # with Display(backend='xvfb') as disp2:
        #     print("NEW DISPLAY AT", disp2.display) # 2
        # working! do not use gnome-terminal.
        proc = easyprocess.EasyProcess(["alacritty"])
        # proc = easyprocess.EasyProcess(['gnome-terminal', f"--display={disp.display}"])
        # proc = easyprocess.EasyProcess(['gnome-terminal', f"--display={disp.display}"])
        proc.start()
        time.sleep(3)
        # from Xlib.display import Display
        # Display(os.environ['DISPLAY']).get_input_focus()
        # not working.
        # pyautogui.write("echo hello world\n")
        # works.
        p = subprocess.Popen(
            "xdotool type --file -".split(), stdin=subprocess.PIPE)
        stdout_data = p.communicate(input=b'echo hello world\n')[0]
        # p.wait()
        # keyboard.type("echo hello world\n")
        pyautogui.screenshot("terminal2.png")
        img = disp.grab()  # shit! not working.
        if img:
            img.save("terminal.png")
        else:
            print("no image yet.")
        proc.stop()
