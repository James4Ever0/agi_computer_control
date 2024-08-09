SESSION_NAME=agent_terminal

tmux capture-pane -t $SESSION_NAME -pe | aha # capture all color sequence, but without knowing cursor location
# tmux capture-pane -t $SESSION_NAME -p