import torch

# you may witness the difference.

def get_optim_lrs(optim):
    lr_list = []
    for pg in optim.param_groups:
        lr = pg['lr']
        lr_list.append(lr)

def set_optim_lrs(optim, lr_list):
    for index,pg in enumerate(optim.param_groups):
        pg['lr'] = lr_list[index]

# eliminate complex setup.
lr = 0.001
model = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.Linear(10, 1))
optim = torch.optim.SGD(model.parameters(), lr=lr)

# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
# call `scheduler.step()` to schedule next learning rate

# class MyScheduler(torch.optim.lr_scheduler.LRScheduler):
#     ...

lr_list = get_optim_lrs(optim)
print(lr_list)

# you don't need the scheduler. or you might need the scheduler that can recurse with the model output.