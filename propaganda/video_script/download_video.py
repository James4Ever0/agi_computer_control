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
output_data = {}
index = 0
for d in data:
    vlink = d.get('video', None)
    if vlink:
        index+=1
        vpath = f"{index}.mp4"
        video_output = os.path.join(dir_path, vpath)
        os.system(f'ffmpeg -y -i "{vlink}" {video_output}')
        output_data.update({vlink:video_output})

with open(output_path,'w+') as f:
    f.write(json.dumps(output_data, indent=4, ensure_ascii=False))