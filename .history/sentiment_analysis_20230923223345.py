from bert import BertModel
import tokenizer
import torch
import torch.nn as nn
import argparse
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
import tqdm
from optimizer import AdamW
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
    def forward(self, input_ids, attention_mask = None):
        _, bert_encode = self.bert(input_ids, attention_mask)
        bert_encode = self.do(bert_encode)
        logits = self.sentiment_proj(bert_encode)
        return logits
    
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
def load_data(filepath, flag = 'train'):
    sents = []
    sent_ids = []
    labels = []
    num_labels = {}
    if flag == 'test':
        with open(filepath) as fp:
            for record in fp:
                record = record.strip().split('\t')
                if len(record)==2:
                    id, sent = record
                else:
                    _, id, sent = record
                sent_ids.append(id)
                sents.append(sent)
    else:
        with open(filepath) as fp:
            for record in fp:
                record = record.strip().split('t')
            if len(record)==3:
                id, sent, sentiment = record
            else: 
                _, id, sent, sentiment = record
            sent_ids.append(id)
            sents.append(sent)
            labels.append(int(sentiment))
            num_labels.add(sentiment)
    return (sents, sent_ids, labels), num_labels
def collate_fn(data):
    sents, sent_ids , labels = data
    encoder = tokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_mask = encoder(sents, padding = True, truncation = True)
    return sents, input_ids, attention_mask, sent_ids, labels
def train(args, mode = 'train'):
    model = BertSentimentClassifier(args)
    if mode == 'train':
        data, num_labels = load_data(args.train_set, flag = 'train')
        model.train()
    else:
        data, num_labels = load_data(args.dev_set, flag = 'train')
        model.eval()
    args['num_labels'] = num_labels
    ds = SentimentDataset(data)
    dl = DataLoader(ds,args.bs,collate_fn=collate_fn)   
    for epoch in range(args.num_epochs):
        pbar = tqdm(dl, desc = f'Epoch: {epoch}' )
        total_loss = 0
        for batch in pbar:
            _, input_ids, attention_mask, _, labels = batch
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            if mode == 'train':
                optimizer = AdamW(model.parameters(), lr = args.lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
            total_loss+=loss
        avg_loss = total_loss/len(dl)
        print('Epoch {}: Average Loss is: {:.4f}'.format(epoch+1, avg_loss))
    #evaluate 
def test(args):
    data, num_labels = load_data(args.test_set, flag = 'test')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'enter the config for your bert training')
    parser.add_argument('--option', type = str, help = 'choose pretrain mode or finetune mode', default= 'finetune')
    parser.add_argument('--num_epochs', type = int, default= 10)
    parser.add_argument('--lr', type = float, default= 1e-5)
    parser.add_argument('--bs', type = int, default = 64)
    parser.add_argument('--dropout-rate', type = float, default = 0.3)
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

