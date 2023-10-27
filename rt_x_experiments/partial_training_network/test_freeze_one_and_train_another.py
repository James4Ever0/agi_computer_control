import torch

# how to create the map?
# model.named_parameter_list().__next__()


class MyModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MyModel, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = self.fc1.forward(x)
        ret = self.fc2.forward(x1)
        return ret


input_size = 200
output_size = 250
hidden_size = 300
model = MyModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

optimizer_list = {}
parameter_list = {}
lr = 0.001

criterion = torch.nn.MSELoss()

for it in model.named_parameters():
    name, param = it
    # print(name)
    parameter_list[name] = param
    optimizer_list[name] = torch.optim.Adam([param], lr=lr)

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

randomize_param_selection = True
# randomize_param_selection = False
import random

for epoch in range(100):
    if randomize_param_selection:
        selected_parameter_index = random.randint(0, parameter_index_count - 1)
    else:
        selected_parameter_index = epoch % parameter_index_count
    selected_parameter_name = parameter_names[selected_parameter_index]
    for pname, param in parameter_list.items():
        if pname != selected_parameter_name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = optimizer_list[selected_parameter_name]

    # Forward pass
    output = model(x)

    # Compute the loss
    loss = criterion(output, target)

    # Zero the gradients
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

import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024**3  # in GB

print("Memory Usage:", memory_usage, "GB")