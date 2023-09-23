import torch
a = torch.tensor(5)
b = torch.tensor(3)
c = torch.tensor(2)
d = torch.tensor(8)
a.addcdiv_(b,c,d)
print(a)