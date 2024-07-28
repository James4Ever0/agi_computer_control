tmux new -s beat_server -d "conda run -n cybergod --no-capture-output python beat_server.py"
tmux new -s llm_fallback_server -d "bash run_llm_fallback_server.sh"