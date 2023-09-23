import torch
a = torch.tensor(5)
b = torch.tensor(3)
a.add_(b)
print(a)