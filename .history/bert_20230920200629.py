import torch
import torch.nn as nn
import math
from based_bert import BertPreTrainedModel
from utils import *
import sys
import torch.nn.functional as F
activate_func = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
}
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size%config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size//config.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim = -1)
        self.do = nn.Dropout( p = config.dropout_rate)
    def transform(self, x, f):
        bs, sen_len = x.size()[:2]
        proj = f(x)
        proj = proj.view(bs,sen_len,self.num_heads,self.head_size)
        proj = proj.permute(0,2,1,3)
        return proj
    def attention(self, xq, xk, xv, masked_attention):
        # B: batch size
        # S: sequence length
        # E: embedding vector
        # H: number of heads
        # D: head size 
        # assert  _, self.num_heads, _ , self.head_size = xq.shape
        xk_transposed = xk.permute(0,1,3,2)
        # assert xk.shape = _, self.num_heads, self.head_size, _
        # assert xv.shape = _, self.num_heads, _, self.head_size
        B, _, L, _ = xq.shape
        attention_score = xq.matmul(xk_transposed)/math.sqrt(self.head_size) # (B, H, S, S)
        if masked_attention is not None:
            #masked_attention: (B,1,1,S)
            masked_attention = masked_attention[:,None,None,:]
            masked_attention = (1.0 - masked_attention)*-10000
        attention_score += masked_attention
        attention_score = self.softmax(attention_score)
        attention_score = self.do(attention_score).matmul(xv) # B, H, L, D
        attention_score = attention_score.permute(0,2,1,3).reshape(B,L,-1) #B, S, E
        return attention_score
    def forward(self, x, masked_attention):
        xq = self.transform(x, self.query)
        xk = self.transform(x, self.key)
        xv = self.transform(x, self.value)
        out = self.attention(xq ,xk, xv, masked_attention)
        return out
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MultiheadAttention = BertAttention(config)
        self.attention_dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.ff = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size, bias = True),
                                activate_func.get(config.activation_func, None)(),
                                nn.Linear(config.intermediate_size, config.hidden_size, bias = True),
                               )
        self.do = nn.Dropout(p = config.dropout_rate)
    def forward(self, x, masked_attention):
        f = self.MultiheadAttention(x, masked_attention)
        f = self.do(self.attention_dense_layer(f))
        out = self.norm(x+f)
        f =self.do(self.ff(out))
        out = self.norm(out+f)
        return out
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_layers = config.num_layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pos_ids = torch.arange(config.max_position_embeddings)
        self.register_buffer('position_ids', self.pos_ids)
        self.norm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.do = nn.Dropout(config.dropout_rate)
        self.bert_layers = nn.ModuleList([BertLayer(config,) for _ in range(config.num_hidden_layers)])
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
        self.init_weights()
    def embed(self, input_ids):
        #token embedding
        seq_len = input_ids.size(1)
        token_embed = self.token_embedding(input_ids)
        #position embedding
        pos_ids = self.pos_ids[:seq_len].unsqueeze(0)
        pos_embed = self.pos_embedding(pos_ids)
        #segment embedding
        segment_id = torch.zeros(input_ids.size(), dtype = torch.long )
        segment_embed = self.segment_embedding(segment_id)
        out = token_embed+ pos_embed+ segment_embed
        out = self.do(self.norm(out))
        return out
        
    def forward(self, x, masked_attention = None):
        # x: (B, S, E)
        # mask: (B,S)
        x = self.embed(x)
        for layer in self.bert_layers:
            x = layer(x, masked_attention)
        last_hidden_state = x
        cls_token = x[:,0,:]
        cls_token = self.pooler_af(self.pooler_dense(cls_token))
        out = {'last_hidden_state': last_hidden_state, 'pooler_output':cls_token}
        return out