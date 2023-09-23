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
# b +=1
# print(a['b'])
# a = torch.tensor(5)
# b = a
# b = torch.tensor(3)
# print(a)
# a = torch.ones((3,2))
# b = a[0,0]
# b.add_(1)
# print(a)
# m_t = torch.zeros((3,2))
# v_t = torch.zeros((3,2))
# beta1 = 0.9
# grad = torch.rand((3,2))
# m_t = beta1*m_t + (1.0-beta1)*grad
# v_t.mul_(beta1).add_(1.0 - beta1, grad)
# print(m_t)
# print(v_t)
filepath = './data/sst/ids-sst-train.csv'
with open(filepath) as fp:
    for record in fp:
        print(record)
        id , sent, sentiment = record.strip().split('\t')
        print(id)
        print(sent)
        print(sentiment)
