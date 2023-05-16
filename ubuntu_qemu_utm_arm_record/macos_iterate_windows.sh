python3 -c "from screenshot import get_window_id; print(get_window_id.get_window_info(options=get_window_id.build_option_bitmask()))" | grep Name | less

# or use obs, then use obs-websocket to take screeenshots.
