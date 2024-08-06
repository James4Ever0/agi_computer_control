SESSION_NAME=agent_terminal
TERM_WIDTH=80
TERM_HEIGHT=24
TERMINAL_COMMAND="docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 ubuntu:22.04"


bash kill_tmux_session_isolated.sh

tmux new-session -d -s $SESSION_NAME -x $TERM_WIDTH -y $TERM_HEIGHT $TERMINAL_COMMAND

tmux set-option -t $SESSION_NAME prefix None
tmux set-option -t $SESSION_NAME prefix2 None
tmux set-option -t $SESSION_NAME status off
tmux set-option -t $SESSION_NAME aggressive-resize off
tmux set-option -t $SESSION_NAME window-size manual # this is effective
