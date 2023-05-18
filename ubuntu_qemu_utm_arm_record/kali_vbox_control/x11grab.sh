env XAUTHORITY=/tmp/.Xauthority ffmpeg -f x11grab -i :10 -f image2pipe pipe:1 | ffplay -f x11grab -i - -autoexit
# env XAUTHORITY=/tmp/.Xauthority ffmpeg -f x11grab -i :10 -f image2pipe pipe:1
# env XAUTHORITY=/tmp/.Xauthority ffplay -f x11grab -i :10 2>&1