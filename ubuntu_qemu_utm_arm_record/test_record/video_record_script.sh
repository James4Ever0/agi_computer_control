/usr/bin/python3 pyscreenshot_output.py | ffmpeg -y -f rawvideo -pix_fmt bgr0 -s 1920x1080 -i - -r 30 -c:v libx264 ./test_record/video_record.mp4
