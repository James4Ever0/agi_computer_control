# we need three processes/threads

# one for screenshot, using mss, taking one and save to jpeg every one second

# one for keyboard events, using pynput

# one for mouse events, using pynput

# the entire thing will be saved under a specific directory

# file handles will be closed within the "finally" code block