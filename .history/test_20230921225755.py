import torch
a = torch.tensor(5)
b = torch.tensor(6)
c = torch.tensor(2)
d = torch.tensor(4)
a.addcdiv_(b,c,d)
print(a)