import pyttsx3
import os

dir_path = "audio"


script_path = "script.yaml"
with open(script_path,'r') as f:
    data = yaml.load(f, Loader=yaml.BaseLoader)
