import torch

# how to create the map?
# model.named_parameter_list().__next__()
import psutil
process = psutil.Process()

def get_ram_usage():
    
    memory_usage = process.memory_info().rss / 1024**3  # in GB

    print("Memory Usage:", memory_usage, "GB")


class MyModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MyModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = self.fc1.forward(x)
        ret = self.fc2.forward(x1)
        return ret


input_size = 2000
output_size = 2500
hidden_size = 3000
model = MyModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

optimizer_list = {}
parameter_list = {}

# will get faster over size, but the gradient may not descent as fast
# memory usage is nearly the same

freeze = True

# freeze = False

if freeze:
    model.eval()

get_ram_usage() # 0.22

lr = 0.001

criterion = torch.nn.MSELoss()
# param_limit = 1
# param_limit = 2
param_limit = None

if not freeze:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    for index, it in enumerate(model.named_parameters()):
        if param_limit is not None:
            if index >= param_limit: break
        name, param = it
        # print(name)
        parameter_list[name] = param
        optimizer_list[name] = torch.optim.Adam([param], lr=lr)

get_ram_usage() # 0.22

# fc1.weight
# fc1.bias
# fc2.weight
# fc2.bias

# create fake learning data
batch_size = 10
x = torch.randn(batch_size, input_size)
target = torch.randn(batch_size, output_size)

parameter_names = list(parameter_list.keys())
parameter_index_count = len(parameter_names)

# this is good
randomize_param_selection = True

# this is bad
# randomize_param_selection = False
import random

for epoch in range(100):
    if freeze:
        if randomize_param_selection:
            selected_parameter_index = random.randint(0, parameter_index_count - 1)
        else:
            selected_parameter_index = epoch % parameter_index_count
        selected_parameter_name = parameter_names[selected_parameter_index]

        # this part is not necessary. i doubt that.
        # does not save memory

        for pname, param in parameter_list.items():
            if pname != selected_parameter_name:
                param.requires_grad = False
                param.grad = None
            else:
                param.requires_grad = True
        # breakpoint()

        optimizer = optimizer_list[selected_parameter_name]

    # Forward pass
    output = model(x)

    # Compute the loss
    loss = criterion(output, target)

    # Zero the gradients
    # optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print the loss every 10 epochs
    if epoch % 10 == 9:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# how to get memory usage?


# Get the memory usage
# no nothing shown of cpu
# memory_usage = torch.cuda.memory_allocated(device='cpu')  # in Bytes
# # memory_usage = torch.cuda.memory_allocated(device=device) / 1024**3  # in GB

# print("Memory Usage:", memory_usage, "Bytes")

############ CPU Usage ############

print("Freeze?", freeze)
get_ram_usage()
# I think maybe this is intended to be used in online training. cause it significantly reduces overfitting.

# Freeze? False Memory Usage: 0.17539215087890625 GB
# Epoch 100, Loss: 1.3618760931422003e-05

# Freeze? True Memory Usage: 0.17582321166992188 GB
# Epoch 100, Loss: 0.021172840148210526

################################################################

# Epoch 100, Loss: 5.7390594482421875
# Freeze? True Memory Usage: 0.3774299621582031 GB

# Epoch 100, Loss: 0.0014482313999906182
# Freeze? False Memory Usage: 0.37708282470703125 GB

# Epoch 100, Loss: 0.00897219032049179
# Freeze? True Memory Usage: 0.32117462158203125 GB

# Epoch 100, Loss: 0.13553307950496674
# Freeze? True Memory Usage: 0.3498115539550781 GB

# less memory used?