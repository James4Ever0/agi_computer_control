export TMUX=""
SESSION_NAME="naive_interactive_session"
env INIT_CLI="tmux new-session -s $SESSION_NAME python3 naive_interactive.py" TMUX="" conda run -n cybergod --no-capture-output python3 naive_actor.py
tmux kill-session -t $SESSION_NAME