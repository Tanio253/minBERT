from bert import BertModel
import torch
import torch.nn as nn
import argparse
from types import SimpleNamespace
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
def load_data(filepath, flag = 'train'):
    if flag == 'test':
        with open(filepath) as fp:
            for record in fp:
                sent = record['sentence'].strip()
                
    with open(filepath) as fp:
        for record in fp:
            sent = record['sentence']
            sent
def train(args):
def test(args):
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'enter the config for your bert training')
    parser.add_argument('--option', type = str, help = 'choose pretrain mode or finetune mode', default= 'finetune')
    parser.add_argument('--num_epochs', type = int, default= 10)
    parser.add_argument('--lr', type = float, default= 1e-5)
    parser.add_argumnet('--bs', type = int, default = 64)
    args = parser.parse_args()
    #training sst dataset ---------------------------------
    config = SimpleNamespace(
        filepath = 'sst.pt',
        option = args.option,
        num_epochs = args.num_epochs,
        lr = args.lr,
        bs = args.bs,
        train_set = './data/sst_ds/ids-sst-train.csv',
        dev_set = './data/sst_ds/ids-sst-dev.csv',
        test_set = './data/sst_ds/ids-sst-test-student.csv',
        dev_out = './prediction/'+args.option+'-sst-dev.txt',
        test_out = './prediction/'+args.option+'-sst-test.txt',
    )
    train(config)
    test(config)
    #training cfimdb dataset -------------------------------------
    config = SimpleNamespace(
        filepath = 'cfimdb.pt',
        option = args.option,
        num_epochs = args.num_epochs,
        lr = args.lr,
        bs = args.bs,
        train_set = './data/cfimdb/ids-cfimdb-train.csv',
        dev_set = './data/cfimdb/ids-cfimdb/dev.csv',
        test_set = './data/vfimdb/ids-cfimdb-test-student.csv',
        dev_out = './prediction/'+ args.option+ '-cfimdb_dev.txt',
        test_out = './prediction/'+ args.option + '-cfimdb-test.txt',
    )
    train(config)
    test(config)

