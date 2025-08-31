rm -f /tmp/ts_pipe && mkfifo /tmp/ts_pipe
while true; do echo $(date +%s) > /tmp/ts_pipe; done