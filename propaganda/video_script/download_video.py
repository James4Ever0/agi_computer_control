import os
import yaml

script_path = "script.yaml"
dir_path = "video"

with open(script_path,'r') as f:
    data = yaml.load(f)

os.system