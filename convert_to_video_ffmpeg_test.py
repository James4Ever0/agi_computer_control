# ffmpeg has better compression ratio. opencv sucks.

import os
import json

with open("screenshot_and_actions.json", "r") as f:
    data = json.loads(f.read())
    screenshot_and_actions = data["screenshot_and_actions"]
    image_size = screenshot_and_actions[0]["screenshot"]["imageSize"]  # [x, y]
    lines = [
        e["screenshot"]["imagePath"]
        for e in screenshot_and_actions
        # f"""file '{e["screenshot"]["imagePath"]}'""" for e in screenshot_and_actions
    ]

input_file_list_path = "input_filenames.txt"
# output_video_path = "ffmpeg_output_test_hwaccel_hevc.mp4"
output_video_path = "ffmpeg_output_test_hwaccel.mp4"
# output_video_path = "ffmpeg_output_test.mp4"

video_size = f"{image_size[0]}*{image_size[1]}"

with open(input_file_list_path, "w+") as f:
    f.write("\n".join(lines))
    f.write("\n")

pix_fmt = "bgr0"
framerate = 1 / 0.0300241
# threads = 16
# threads = 100
# preset = 'ultrafast'

# i think the bottleneck is not the encoder but the IO.
# put screenshots to ramdisk?

# command = f'cat {input_file_list_path} | xargs -I abc cat abc | ffmpeg -y -f rawvideo -pix_fmt {pix_fmt} -s {video_size} -i - -r {framerate} -c:v hevc_videotoolbox  {output_video_path}' # hardware accelerated

# command = f'cat {input_file_list_path} | xargs -I abc cat abc | ffmpeg -y -f rawvideo -pix_fmt {pix_fmt} -s {video_size} -i - -r {framerate} -c:v hevc_videotoolbox -threads {threads} {output_video_path}' # hardware accelerated

command = f"cat {input_file_list_path} | xargs -I abc cat abc | ffmpeg -y -r {framerate} -f rawvideo -pix_fmt {pix_fmt} -s {video_size} -i - -c:v h264_videotoolbox {output_video_path}"  # hardware accelerated, do not use "-r" option.

# command = f'cat {input_file_list_path} | xargs -I abc cat abc | ffmpeg -y -f rawvideo -pix_fmt {pix_fmt} -s {video_size} -i - -r {framerate} -c:v h264_videotoolbox -threads {threads} -preset {preset} {output_video_path}' # hardware accelerated

# command = f'cat {input_file_list_path} | xargs -I abc cat abc | ffmpeg -y -f rawvideo -pix_fmt {pix_fmt} -s {video_size} -i - -r {framerate} -c:v libx264 {output_video_path}' # quality first, otherwise it will be very blurry.

# command = f'ffmpeg -y -f concat -vcodec rawvideo -pix_fmt {pix_fmt} -s {video_size} -i {input_file_list_path} -r {framerate} -c:v libx264 {output_video_path}'

### SUCCESSFUL COMMAND:
################################################################
### cat screenshots/* | ffplay -f rawvideo -pixel_format bgr0 -video_size 2560x1600 -framerate 30 -i -
### cat input_filenames.txt | xargs -I abc cat abc | ffplay -f rawvideo -pixel_format bgr0 -video_size 2560x1600 -framerate 30 -i -
################################################################


# try use xargs instead. mind the difference of linux and macos.

print(command)
os.system(command)
