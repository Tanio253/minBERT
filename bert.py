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
        self.proj = nn.Linear(config.total_head_size, config.fhsize)
        self.softmax = nn.Softmax(dim = -1)
        self.do = nn.Dropout( p = config.dropout_rate)
    def transform(self, x, f):
        bs, sen_len = x[:2]
        proj = f(x)
        proj = proj.view(bs,sen_len,self.num_heads,self.head_size)
        proj = proj.permute(0,2,1,3)
        return proj
    def attention(self, xq, xk, xv):
        assert xq.shape = _, self.num_heads, _, self.head_size
        xk = xk.permute(0,1,3,2)
        assert xk.shape = _, self.num_heads, self.head_size, _
        assert xv.shape = _, self.num_heads, _, self.head_size
        b, _, l, _ = xq.shape
        attention_score = self.softmax(xq.bmm(xk)/self.head_size).mm(xv) # b, h, l, e/h
        attention_score = attention_score.permute(0,2,1,3).view(b,l,-1) #b, l, e
        attention_score = self.proj(attention_score) #b, l, fhsize
    def forward(self, x):
        xq = transform(x, self.query)
        xk = transform(x, self.key)
        xv = transform(x, self.value)
        out = attention(xq ,xk, xv)
        return out
class BertLayer(nn.Module):
    def __init__(self, config):
        super.__init__(config)
        self.MultiheadAttention = BertAttention(config)
        self.norm = nn.LayerNorm(config.fhsize)
        self.ff = nn.Sequential(nn.Linear(config.fhsize, config.ffpro1, bias = True),
                                nn.ReLU(),
                                nn.Linear(config.ffpro1, config.ffpro2, bias = True),
                               )
    def forward(self, x):
        x = x + self.MultiheadAttention(x)
        x = self.norm(x)
        x = x+ self.ff(x)
        x = self.norm(x)
        return x
class BertModel(nn.Module):
    def __init__(config):
        super.__init__(config)
        self.embedding = EmbeddingLayer
        self.num_layers = config.num_layers
    def forward(self, x, config):
        x = self.embedding(x)
        for i in range(self.num_layers):
            bertlayer = BertLayer(config)
            x = bertlayer(x)
        last_hidden_state = x[:,1:,:]
        cls