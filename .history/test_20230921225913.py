import torch
a = torch.tensor(5, dtype = float)
b = torch.tensor(6, dtype = float)
c = torch.tensor(2, dtype = float)
d = torch.tensor(4, dtype = float)
a.addcdiv_(b,c,d)
print(a)