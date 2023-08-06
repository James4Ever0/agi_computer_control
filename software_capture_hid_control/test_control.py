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
    import easyprocess  # no support for stdin!
    # import time
    import os
    import subprocess

    def type_string(string: str):
        input_bytes = string.encode()
        p = subprocess.Popen(
            "xdotool type --file -".split(), stdin=subprocess.PIPE)
        stdout_data = p.communicate(input=input_bytes)[0]
        return stdout_data

    os.system("rm *.png")
    # keyboard = Controller()
    # virtual_display = ":3"
    # backend = 'xvnc'
    backend = 'xephyr' # like visible xvfb, useful for live streaming (no need for ffmpeg hacks with xvfb)
    # backend = 'xvfb'
    # with Display(backend=backend) as disp:
    # proc_cmd = ["xterm"]
    proc_cmd = ["alacritty"]
    with SmartDisplay(backend=backend, size=(1920,1080), extra_args=['-fullscreen',  '-softCursor']) as disp: 
    # with SmartDisplay(backend=backend, size=(1920, 1080)) as disp: 
    # with SmartDisplay(backend=backend, size=(1920, 1080), extra_args=['-fullscreen']) as disp: # for unit testing purpose. maybe we should log events on that display.
    # with SmartDisplay(backend=backend, extra_args=['-title', 'xephyr_test']) as disp: # get window location by title first, then limit all events to that window.
    # with SmartDisplay(backend='xvfb') as disp:
        # with Display(backend='xvfb') as disp:
        # with Display(visible=False) as disp:

        # works, but remember to import/reload this module when in virtual display context.
        import pyautogui
        print("NEW DISPLAY AT", disp.display)  # 0, INT
        print("ENV DISPLAY?", os.environ["DISPLAY"])  # :0

        # pynput controller not working.
        # from pynput.keyboard import Controller
        # from pynput.keyboard import Listener
        # keyboardListener = Listener()
        # keyboardController = Controller()
        # with Display(backend='xvfb') as disp2:
        #     print("NEW DISPLAY AT", disp2.display) # 2
        # working! do not use gnome-terminal.

        # proc = easyprocess.EasyProcess(["alacritty"])
        # proc = easyprocess.EasyProcess(['gnome-terminal', f"--display={disp.display}"])
        # proc = easyprocess.EasyProcess(['gnome-terminal', f"--display={disp.display}"])
        # no need for starting/stopping
        import mss
        with easyprocess.EasyProcess(proc_cmd) as proc:
            # proc.start()
            # proc.start().sleep(3)
            # proc.sleep(5)
            proc.sleep(3)
            # time.sleep(3)
            # from Xlib.display import Display
            # Display(os.environ['DISPLAY']).get_input_focus()
            # not working.
            pyautogui.write("echo hello world pyautogui\n")
            # works.

            # need this to "awake" the terminal when fullscreen.
            os.system("xdotool mousemove 0 0")
            os.system("xdotool click 1")

            type_string('echo hello world\n')
            # p.wait()
            # keyboardController.type("echo hello world pynput\n")
            pyautogui.screenshot("terminal2.png")  # full shot
            # img = disp.grab()  # # partial shot, only on changes
            # maybe we can use this as some sort of "attention" mechanism?
            img = disp.grab(autocrop=False)  # full shot again.
            if img:
                img.save("terminal.png")
            else:
                print("no image yet.")
            type_string("just some words.")
            # .save not working
            # mss.mss().save(output="terminal4.png")
            # mon_shot = mss.mss().save(mon=1, output="terminal4.png")
            if backend == "xvfb":
                mon_shot = mss.mss().shot(output="terminal4.png")
            # print(mon_shot)
            # nope. no attention/diff mechanism.
            disp.grab().save("terminal3.png")
            # proc.stop()
