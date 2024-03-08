import torch
import torch_directml
import contextlib
import copy
from functools import reduce

@contextlib.contextmanager
def null_inference_mode(*args, **kwargs):
    try:
        yield
    finally:
        pass
    
torch.inference_mode = null_inference_mode

old_cat = copy.copy(torch.cat)

def smart_cat(arr, *args, **kwargs):
    new_arr = []
    for it in arr:
        shape = it.shape
        size = reduce(lambda x,y: x*y, shape)
        if size >0:
            new_arr.append(it)
    ret = old_cat(new_arr, *args, **kwargs)
    return ret

torch.cat = smart_cat

dev = torch_directml.device()
model = torch.hub.load("../yolov5", "yolov5m", source='local')
# get torch cache path?
# model = torch.hub.load("ultralytics/yolov5", "yolov5m")

filepath = "./zidane.jpg"

# two issues:
# 1. directml doesn't work with inference mode (yet), you can nullify it.
# 2. torch.cat is not working properly, because we are passing zero size arrays into it. however, the cpu executor supports it.
# with torch.inference_mode(mode=False):
with torch.no_grad():
    # for _ in range(1000):
    print(model(filepath))
model.to(dev)
# with torch.inference_mode(mode=False):
with torch.no_grad():
    # for _ in range(1000):
    # amd yes!
    print(model(filepath))
