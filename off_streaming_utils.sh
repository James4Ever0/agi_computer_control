tmux kill-session -t streaming_utils
ps aux | grep bash | grep -v grep | grep kali_prepare_dirs_loop.sh | awk '{print $2}' | xargs -iabc kill -s TERM abc
tmux kill-session -t kali_prepare_two_webdav_dirs