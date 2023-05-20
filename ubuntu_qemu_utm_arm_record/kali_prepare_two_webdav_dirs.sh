# create some room for tmux.
# use tmux automation tool, reading yaml config and create multiple windows.
tmux kill-session -t kali_prepare_two_webdav_dirs
# tmuxp load kali_prepare_two_webdav_dirs.yaml

# use absolute path.
nohup xvfb-run --server-num=12 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority2 gnome-terminal --display=12 -- tmuxp load "/media/root/Toshiba XG3/works/agi_computer_control/ubuntu_qemu_utm_arm_record/kali_prepare_two_webdav_dirs.yaml" & 
# nohup xvfb-run --server-num=12 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority2 gnome-terminal --display=12 -- tmuxp load "/media/root/Toshiba XG3/works/agi_computer_control/ubuntu_qemu_utm_arm_record/kali_prepare_two_webdav_dirs.yaml" & 

# gnome-terminal -- tmuxp load "/media/root/Toshiba XG3/works/agi_computer_control/ubuntu_qemu_utm_arm_record/kali_prepare_two_webdav_dirs.yaml"