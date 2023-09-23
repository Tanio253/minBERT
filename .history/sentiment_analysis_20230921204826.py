from bert import BertModel
import torch
import torch.nn as nn
class BertSentimentClassifier(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.do = nn.Dropout(config.dropout_rate)
        self.sentiment_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # if we use pretrain we dont need to keep track of gradient
        for params in self.bert.parameters:
            if config.option == 'pretrain':
                params.requires_grad = False 
            if config.option == 'finetune':
                params.requires_grad = True
    def forward(self, input_ids, masked_attention):
        _, bert_encode = self.bert(input_ids, masked_attention)
        bert_encode = self.do(bert_encode)
        logits = self.sentiment_proj(bert_encode)
        return logits