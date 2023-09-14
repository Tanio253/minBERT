import torch
import torch.nn as nn
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention,self).__init__(config)
        self.num_heads = config.num_heads
        self.head_size = self.hidden_size//self.num_heads
        self.total_head_size = self.num_heads*self.head_size
        self.query = nn.Linear(config.hidden_size, config.total_head_size)
        self.key = nn.Linear(config.hidden_size, config.total_head_size)
        self.value = nn.Linear(config.hidden_size, config.total_head_size)
    def transform(self, x, f):
        bs, sen_len = x[:2]
        proj = f(x)
        proj = proj.view(bs,sen_len,self.num_heads,self.head_size)
        proj = proj.permute(0,2,3,1)
        return proj
    def attention(self, xq, xk, xv):
        
    def forward(self, x):
        xq = transform(x, self.query)
        xk = transform(x, self.key)
        xv = transform(x, self.value)
        out = attention(xq ,xk, xv)
        return out