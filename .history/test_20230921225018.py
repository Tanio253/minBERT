import torch
a = torch.tensor(5)
b = torch.tensor(3)
c = torch.tensor(2)
a.add_(b, c)
print(a)