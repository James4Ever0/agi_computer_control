# by format or 3rd party library
# tmux list-commands # to get all commands
# tmux list-commands | grep '\-F format'

SESSION_NAME=agent_terminal

tmux list-sessions -F '[#{session_name}] bell: #{window_bell_flag} slient: #{window_silence_flag} active: #{window_activity_flag} socket: #{socket_path} size: #{window_width}x#{window_height} cursor at: x=#{cursor_x},y=#{cursor_y}' | grep $SESSION_NAME
# to run everything with different server, use `-L` or `-S` before command
# tmux -L <server_name> new
# tmux -S <server_file_path> new
