import os
import yaml

script_path = "script.yaml"
dir_path = "video"

with open(script_path,'r') as f:
    data = yaml.load(f, Loader=yaml.BaseLoader)

# os.system

output_path= "output.json"
import json
import rich
rich.print(data)
for d in data:
    vlink = d.get('video', None)
    if vlink:
        video_output = os.path.join()
        os.system(f'ffmpeg -i "{vlink}" {video_output}')