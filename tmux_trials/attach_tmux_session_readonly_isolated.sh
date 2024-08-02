SESSION_NAME=agent_terminal

# tmux attach -t "$SESSION_NAME" \; switch-client -r

tmux attach -r -t "$SESSION_NAME"
