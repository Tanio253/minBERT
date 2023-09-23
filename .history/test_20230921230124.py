import torch
a = torch.tensor(5, dtype = torch.float64)
b = torch.tensor(6, dtype = torch.float64)
c = torch.tensor(2, dtype = torch.float64)
d = torch.tensor(4, dtype = torch.float64)
e = torch.tensor(3, dtype = torch.float64)
a.addcdiv_(b,c,d,e)
print(a)