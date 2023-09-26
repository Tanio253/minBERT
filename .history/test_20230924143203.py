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
filepath = './data/sst_ds/ids-sst-train.csv'
def load_data(filepath, flag = 'train'):
    sents = []
    sent_ids = []
    labels = []
    num_labels = set()
    if flag == 'test':
        with open(filepath) as fp:
            for record in fp:
                record = record.strip().split('\t')
                if len(record)==2:
                    continue
                else:
                    _, id, sent = record
                sent_ids.append(id)
                sents.append(sent)
    else:
        with open(filepath) as fp:
            for record in fp:
                record = record.strip().split('\t')
                if len(record)==3:
                    continue
                else: 
                    _, id, sent, sentiment = record
                sent_ids.append(id)
                sents.append(sent)
                print(sentiment)
                labels.append(int(sentiment))
                num_labels.add(sentiment)
    return (sents, sent_ids, labels), len(num_labels)
(sents, sent_ids, labels), num_labels = load_data(filepath)
print(sents)
print(sent_ids)
print(num_labels)