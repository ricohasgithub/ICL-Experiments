import torch.nn as nn
import torch

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
# Add across the second dimension to get 1
input_sum = torch.sum(input, dim=1)
target_sum = torch.sum(target, dim=1)
print(input_sum)
print(target_sum)

# create a 3 dimensional tensor
input = torch.randn(1, 3, 5)
input = input.view(-1, 5)

output = loss(input, target)
print(output)
# output.backward()
