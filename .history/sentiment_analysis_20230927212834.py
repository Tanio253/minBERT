from bert import BertModel
from tokenizer import BertTokenizer
import torch
from torch.cuda import device_count
import torch.nn as nn
import argparse, numpy as np, random
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from optimizer import AdamW
from functools import partial
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
TQDM_DISABLE = False
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class BertSentimentClassifier(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.do = nn.Dropout(config.dropout_rate)
        self.sentiment_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # if we use pretrain we dont need to keep track of gradient
        for params in self.bert.parameters():
            if config.option == 'pretrain':
                params.requires_grad = False 
            if config.option == 'finetune':
                params.requires_grad = True
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        bert_encode = self.do(out['pooler_output'])
        logits = self.sentiment_proj(bert_encode)
        return logits
    
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
def load_data(filepath, flag):
    sents = []
    sent_ids = []
    labels = []
    num_labels = set()
    if flag == 'test':
        with open(filepath) as fp:
            for record in fp:
                record = record.strip().split('\t')
                if record[0]=='0':
                    continue
                else:
                  _, _, id, sent = record
                sent_ids.append(id)
                sents.append(sent)
        labels = torch.zeros(len(sent_ids))
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
                labels.append(int(sentiment))
                num_labels.add(sentiment)
    batched_data = tuple(zip(sents, sent_ids, labels))
    return batched_data, len(num_labels)
def custom_collate(data, device):
    sents = [d[0] for d in data]
    sent_ids = [d[1] for d in data]
    labels = [d[2] for d in data]
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_tokenizer = bert_tokenizer(sents, return_tensors= 'pt', padding = True, truncation = True  )
    input_ids, attention_mask = bert_tokenizer['input_ids'], bert_tokenizer['attention_mask']
    input_ids = input_ids.to(device)
    labels = torch.tensor(labels, device = device)
    attention_mask = attention_mask.to(device)
    return  input_ids, attention_mask, labels, sents, sent_ids
def evaluation(model, dl):
    model.eval()
    y_pred = []
    y_true = []
    sents = []
    sent_ids = []
    pbar = tqdm(dl, desc = f'evaluation')
    with torch.no_grad():
      for batch in pbar:
          input_ids, attention_mask, labels, sent, sent_id = batch
          logits = model(input_ids, attention_mask)
          logits = logits.cpu().numpy() #why detach here
          logits = np.argmax(logits, axis = 1).flatten()
          labels = labels.cpu().flatten()
          y_pred.extend(logits)
          y_true.extend(labels)
          sents.extend(sent)
          sent_ids.extend(sent_id)
      f1_s = f1_score(y_true,y_pred, average= 'macro')
      acc = accuracy_score(y_true, y_pred)
      return acc, f1_s, y_pred, sents, sent_ids
def test_evaluation(model, dl):
    model.eval()
    y_pred = []
    sents = []
    sents_id = []
    pbar = tqdm(dl, desc = f'test time')
    with torch.no_grad():
      for batch in pbar:
          input_ids, attention_mask, _,  sent, sent_id = batch
          logits = model(input_ids, attention_mask)
          logits = logits.cpu().numpy()
          logits = np.argmax(logits, axis = 1).flatten()
          y_pred.extend(logits)
          sents.extend(sent)
          sents_id.extend(sent_id)
    return y_pred, sents, sents_id
def save_model(model, optimizer, args, config):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'config': config,
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, args.filepath)
    print(f"save the model to {args.filepath}")

def train(args):
    device = 'cuda' if args.use_gpu else 'cpu'
    best_dev_acc = 0
    dev_accs = []
    train_data, num_labels = load_data(args.train_set, flag = 'train')
    config = SimpleNamespace(
        hidden_size = args.hidden_size,
        dropout_rate = args.dropout_rate,
        num_labels = num_labels,
        option = args.option,
    )
    collate_fn = partial(custom_collate, device = device)
    model = BertSentimentClassifier(config).to(device)
    train_data = SentimentDataset(train_data)
    train_dl = DataLoader(train_data, args.bs,shuffle = True, collate_fn=collate_fn) 
    dev_data, _ = load_data(args.dev_set, flag = 'train')
    dev_data = SentimentDataset(dev_data)
    dev_dl = DataLoader(dev_data,args.bs,shuffle = False, collate_fn= collate_fn )
    optimizer = AdamW(model.parameters(), lr = args.lr)
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_dl, desc = f'Epoch: {epoch+1}' )
        total_loss = 0
        for batch in pbar:
            model.train()
            input_ids, attention_mask, labels, *_ = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            #loss = nn.CrossEntropyLoss()(logits, labels)/args.bs
            loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / args.bs
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})
            total_loss+=loss.item()
        train_acc, train_f1, *_ = evaluation(model, train_dl)
        dev_acc , dev_f1, *_ = evaluation(model, dev_dl)
        total_loss/=len(train_dl)
        if dev_acc>best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config)
        dev_accs.append(dev_acc)
        print(f'Epoch {epoch+1} training loss: {total_loss: .4f} training acc: {train_acc: .4f} training f1 score: {train_f1} dev acc: {dev_acc: .4f} dev f1 scores: {dev_f1}')
    avg_acc = torch.mean(torch.tensor(dev_accs, dtype = torch.float64)).item()
    print(f'Average accuracy is: {avg_acc: .4f} Best accuracy is: {best_dev_acc: .4f}')
def test(args):
  with torch.no_grad():
    device = 'cuda' if args.use_gpu else 'cpu'
    test_data, num_labels = load_data(args.test_set, flag = 'test')
    test_ds = SentimentDataset(test_data)
    collate_fn = partial(custom_collate, device = device)
    test_dl = DataLoader(test_ds, args.bs, collate_fn= collate_fn)
    dev_data, num_labels = load_data(args.dev_set, flag = 'train')
    dev_ds = SentimentDataset(dev_data)
    dev_dl = DataLoader(dev_ds, args.bs, collate_fn= collate_fn)
    save_info = torch.load(args.filepath)
    model = BertSentimentClassifier(save_info['config'])
    model.load_state_dict(save_info['model'])
    model = model.to(device)
    test_pred, test_sents, test_sent_ids = test_evaluation(model, test_dl)
    acc, f1, dev_pred, dev_sents, dev_sent_ids = evaluation(model, dev_dl)
    with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {acc :.4f}")
            f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
            for i, s, p in zip(dev_sent_ids, dev_sents, dev_pred ):
                f.write(f"{i} , {s}, {p} \n")

    with open(args.test_out, "w+") as f:
        f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
        for i, s, p  in zip(test_sent_ids, test_sents, test_pred ):
            f.write(f"{i} , {s}, {p} \n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'enter the config for your bert training')
    parser.add_argument('--seed', type = int, default= 11711)
    parser.add_argument('--option', type = str, help = 'choose pretrain mode or finetune mode', choices = ('pretrain','finetune'), default= 'finetune')
    parser.add_argument('--num_epochs', type = int, default= 10)
    parser.add_argument('--lr', type = float, default= 1e-5)
    parser.add_argument('--bs', type = int, default = 8)
    parser.add_argument('--dropout_rate', type = float, default = 0.3)
    parser.add_argument('--use_gpu', action = 'store_true')
    args = parser.parse_args()
    #training sst dataset ---------------------------------
    seed_everything(args.seed)
    config = SimpleNamespace(
        filepath = 'sst.pt',
        option = args.option,
        num_epochs = args.num_epochs,
        lr = args.lr,
        bs = args.bs,
        dropout_rate = args.dropout_rate,
        hidden_size = 768,
        train_set = './data/sst_ds/ids-sst-train.csv',
        dev_set = './data/sst_ds/ids-sst-dev.csv',
        test_set = './data/sst_ds/ids-sst-test-student.csv',
        dev_out = './prediction/'+args.option+'-sst-dev.csv',
        test_out = './prediction/'+args.option+'-sst-test.csv',
        use_gpu = args.use_gpu,
    )
    train(config)
    print('Evaluating on SST dataset')
    test(config)
    #training cfimdb dataset -------------------------------------
    config = SimpleNamespace(
        filepath = 'cfimdb.pt',
        option = args.option,
        num_epochs = args.num_epochs,
        lr = args.lr,
        bs = 8,
        dropout_rate = args.dropout_rate,
        hidden_size = 768,
        train_set = './data/cfimdb/ids-cfimdb-train.csv',
        dev_set = './data/cfimdb/ids-cfimdb-dev.csv',
        test_set = './data/cfimdb/ids-cfimdb-test-student.csv',
        dev_out = './prediction/'+ args.option+ '-cfimdb-dev.csv',
        test_out = './prediction/'+ args.option + '-cfimdb-test.csv',
        use_gpu = args.use_gpu,
    )
    train(config)
    print('Evaluating on Cfimdb dataset')
    test(config)

