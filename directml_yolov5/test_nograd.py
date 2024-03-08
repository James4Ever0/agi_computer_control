import torch
import torch_directml

dev = torch_directml.device()
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# two issues:
# 1. directml doesn't work with torch.inference_mode. you can nullify it.
# 2. torch.cat is not working properly, because we are passing zero size arrays into it. however, the cpu executor supports it.
with torch.no_grad():
    print(model("C:/Users/z98hu/Desktop/zidane.jpg"))
model.to(dev)
with torch.no_grad():
    print(model("C:/Users/z98hu/Desktop/zidane.jpg"))
