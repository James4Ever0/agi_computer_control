rm /tmp/terminal.cast

# let's just terminate the ttyd process. see if we get the recording.
# it works. the cast is saved.
ttyd -p 8080 --once asciinema rec -c 'bash' -t 'Terminal Recorder' -y /tmp/terminal.cast --overwrite

# start the asciinema process, get the PID of asciinema launcher
# ttyd -p 8080 --once bash poc_asciinema_launcher.sh

