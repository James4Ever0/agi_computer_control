import pyttsx3
# import os

dir_path = "audio"
import yaml

script_path = "script.yaml"
with open(script_path,'r') as f:
    data = yaml.load(f, Loader=yaml.BaseLoader)

for index, elem in enumerate(data):
    test = elem['text']
    output_path = f"{dir_path}/{index}.wav"
    pyttsx3