#!/bin/bash

PID_FILE=/tmp/poc_asciinema_launcher.pid

# get the self pid and write to pid file
self_pid=$$
echo $self_pid > $PID_FILE

echo "Self PID: $self_pid"
echo "PID File: $PID_FILE"

asciinema rec -c 'bash' -t 'Terminal Recorder' -y /tmp/terminal.cast