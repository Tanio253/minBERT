import torch
import torch.nn as nn
import math
class BertAttention(nn.Module):
    def __init__(self, config, masked_attention = None):
        super(BertAttention,self).__init__(config)
        assert config.hidden_size%config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_size = self.hidden_size//self.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim = -1)
        self.do = nn.Dropout( p = config.dropout_rate)
        self.masked_attention = masked_attention
    def transform(self, x, f):
        bs, sen_len = x[:2]
        proj = f(x)
        proj = proj.view(bs,sen_len,self.num_heads,self.head_size)
        proj = proj.permute(0,2,1,3)
        return proj
    def attention(self, xq, xk, xv):
        # B: batch size
        # L: source sequence length
        # T: target sequence length
        # E: embedding vector
        # H: number of heads
        # D: head size 
        assert xq.shape = _, self.num_heads, _, self.head_size
        xk = xk.permute(0,1,3,2)
        assert xk.shape = _, self.num_heads, self.head_size, _
        assert xv.shape = _, self.num_heads, _, self.head_size
        B, _, L, _ = xq.shape
        attention_score = xq.bmm(xk)/math.sqrt(self.head_size) # (B, H, L, T)
        if self.masked_attention is not None:
            #masked_attention: (B,1,L,T)
            masked_attention = 1.0 - masked_attention
            masked_attention *= -1e5
        attention_score += masked_attention
        attention_score = self.softmax(attention_score)
        attention_score = self.dropout(attention_score) # B, H, L, D
        attention_score = attention_score.permute(0,2,1,3).view(B,L,-1) #B, L, E
        return attention
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
        self.attention_dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.ff = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size, bias = True),
                                nn.GELU(),
                                nn.Linear(config.intermediate_size, config.hidden_size, bias = True),
                               )
    def forward(self, x):
        f = self.MultiheadAttention(x)
        f = self.dropout(self.attention_dense_layer(f))
        out = self.norm(x+f)
        f =self.dropout(self.ff(out))
        out = self.norm(out+f)
        return out
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
        cls_dim = x[:,0,:]
        return last_hidden_state, cls_dim