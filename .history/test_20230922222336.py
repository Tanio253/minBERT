import torch
import copy
# a = torch.tensor(5, dtype = torch.float64)
# b = torch.tensor(6, dtype = torch.float64)
# c = torch.tensor(2, dtype = torch.float64)
# d = torch.tensor(4, dtype = torch.float64)
# e = torch.tensor(3, dtype = torch.float64)
# a.addcdiv_(b,c,d,e)
# print(a)
# a = dict(b= 4,c = 5)
# b = a['b']
# b = 1
# print(a['b'])
# a = torch.tensor(5)
# b = a
# b = torch.tensor(3)
# print(a)
a = torch.rand((3,2))
b = copy.copy(a)
b = torch.zeros(3,2)
print(a)