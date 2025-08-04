This is a web GUI/terminal recorder. It can record the GUI/terminal input/output and save it to a file.

Has some buttons for controlling recording, a textarea for specifying output filename and descriptions, and an iframe for novnc/ttyd interactions.

The Web server launches a Docker container (worker) running ttyd/novnc and recording the terminal/GUI input/output.

Terminal workers require asciinema.

GUI workers require mss, pynput.