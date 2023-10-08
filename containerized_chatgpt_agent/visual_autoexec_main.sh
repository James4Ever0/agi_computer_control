xvfb-run -n 99 -f ~/.Xauthority xfce4-session &
sleep 2 && env DISPLAY=:99 python3 visual_server_on_ubuntu.py &
sleep 5 && python3 visual_autoexec_example.py