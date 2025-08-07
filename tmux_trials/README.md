This directory contains some experimental tmux utilities.

It contains some APIs for taking screenshot of live tmux session, insert command into tmux, and tracking the tmux i/o activity status (idle/active)

It collects a list of special keys in Tmux at funckeys_final.txt

The screenshot can be in bytes (directly from tmux), html (parsed by aya ansi to html converter), and png (rendered from html).

It cannot record the action (input) from the user.