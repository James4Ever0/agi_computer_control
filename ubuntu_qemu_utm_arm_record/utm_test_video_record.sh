pix_fmt=bgr0
video_size=1280x737
frame_rate=30
output_path=output.mp4

python3 pyscreenshot_output.py | ffmpeg -y -f rawvideo -pix_fmt $pix_fmt -s $video_size -i - -r $frame_rate -c:v libx264 $output_path
