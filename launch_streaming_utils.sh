tmux kill-session -t streaming_utils

gnome-terminal -- bash ubuntu_qemu_utm_arm_record/kali_prepare_dirs_loop.sh
gnome-terminal -- tmuxp load streaming_utils.yaml