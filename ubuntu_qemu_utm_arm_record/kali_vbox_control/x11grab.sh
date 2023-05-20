cd .
while true; do
    env XAUTHORITY=/tmp/.Xauthority ffmpeg -f x11grab -r 10 -i :10 -f image2pipe pipe:1 | ffplay -f image2pipe -i - -autoexit
    sleep 1
done;
# still with some lag! need improvement.

# env XAUTHORITY=/tmp/.Xauthority ffmpeg -f x11grab -i :10 -f image2pipe pipe:1
# env XAUTHORITY=/tmp/.Xauthority ffplay -f x11grab -i :10 2>&1