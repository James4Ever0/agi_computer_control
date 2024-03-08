import torch
import torch_directml

dev = torch_directml.device()

a = torch.ones((20, 20))
a.to(dev)
print(torch.cat([a, a]))
print(torch.cat([a, a], dim=1))
print(torch.cat([a, a], 1))
print(torch.cat((a, a), 1))
print(torch.cat((a, a, a), 1))
print(torch.cat((a, a, a, a), 1))
print(torch.cat((a, a, a, a, a), 1))
