# create some room for tmux.
# use tmux automation tool, reading yaml config and create multiple windows.
tmux kill-session -t kali_prepare_two_webdav_dirs
# tmuxp load kali_prepare_two_webdav_dirs.yaml
gnome-terminal -- tmuxp load kali_prepare_two_webdav_dirs.yaml