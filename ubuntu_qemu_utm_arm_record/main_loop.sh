while true; do ps aux | grep python3 | awk '{print $2}' | xargs -iabc kill -s TERM abc ; python3 main_recorder.py; done;
# use sudo!
# sudo -u root bash