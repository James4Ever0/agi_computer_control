SESSION_NAME=agent_terminal

# tmux send-keys -t $SESSION_NAME "are you ok"
tmux send-keys -t $SESSION_NAME "Enter"
# tmux send-keys -t agent_terminal "apt install -y vim" Enter

# ctrl: c
# shift: s
# meta: m

# keys avaliable:
# https://github.com/tmux/tmux/blob/109d2bda1a7b06948e35b7e08c159e71ebc988fb/regress/input-keys.sh
# https://github.com/tmux/tmux/blob/109d2bda1a7b06948e35b7e08c159e71ebc988fb/key-string.c

# TODO: display terminal response statistics and distinguish between idle and active time intervals