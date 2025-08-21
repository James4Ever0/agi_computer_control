echo "Empty GIF file count:"
find ./gif_video/ -type f -size 0 | wc -l
echo "Removing empty files"
find ./gif_video/ -type f -size 0 | xargs -Iabc rm abc
echo "Done"
