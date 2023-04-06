# opencv or ffmpeg. your choice.

import json
import os

with open("screenshot_and_actions.json", "r") as f:
    data = json.loads(f.read())
    screenshot_and_actions = data["screenshot_and_actions"]
    image_size = screenshot_and_actions[0]["screenshot"]["imageSize"]  # [x, y]
    lines = [
        e["screenshot"]["imagePath"]
        for e in screenshot_and_actions
        # f"""file '{e["screenshot"]["imagePath"]}'""" for e in screenshot_and_actions
    ]

output_video_path = "ffmpeg_output_test_opencv.mp4"

framerate = 1 / 0.0300241

# 1580 frames
# fps: 33.30
# duration: 47.44

import cv2
import numpy as np

# Define the codec and create VideoWriter object
if os.path.exists(output_video_path):
    os.remove(output_video_path)
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
out = cv2.VideoWriter(
    output_video_path, fourcc, framerate, (image_size[0], image_size[1])
)  # framerate cannot be decimal?

# Iterate through each image path and write it to the video output
import progressbar
from PIL import Image

# i think this is better.

for path in progressbar.progressbar(lines):
    # Read the image from raw pixel format in BGRX order
    # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    with open(path, "rb") as f:
        image_bytes = f.read()
        pil_img = Image.frombytes("RGB", image_size, image_bytes, "raw", "BGRX")
        cv2_img = np.array(pil_img)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("IMG", cv2_img)
    # cv2.waitKey(0)

    # Write the image to the video output
    out.write(cv2_img)
    del cv2_img
    del pil_img

# Release the video writer and close the file
out.release()
