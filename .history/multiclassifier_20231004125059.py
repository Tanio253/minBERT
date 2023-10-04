import torch
import tqdm
import torch.nn as nn
from dataset_handling import load_data, MultitaskDataset
from bert import BertModel
from optimizer import AdamW
import argparse, random, numpy as np
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from evaluation import MultiEvaluation, MultiEvaluationTest
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(model, optimizer, config):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, config.filepath)
    print(f"save the model to {config.filepath}")

class MultiBert(nn.Module):
    def __init__(self, config):
        super.__init__(self)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.do = nn.Dropout(p = config.dropout_rate)
        self.config = config
        if config.option == 'pretrain':
            for p in self.bert.parameters():
                p.requires_grad = False
        if config.option == 'finetune':
            for p in self.bert.parameters():
                p.requires_grad = True
    def forward(self, input_ids, attention_mask ):
        bert_output = self.bert(input_ids, attention_mask)
        bert_output = self.do(bert_output['pooler_output'])
        return bert_output
    def predict_paraphrase(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        proj = nn.Linear(self.config.hidden_size*2, 1)
        input_ids = torch.concat((input_ids1,input_ids2), dim = -1)
        attention_mask = torch.concat((attention_mask1, attention_mask2), dim = -1)
        logits = self.forward(input_ids, attention_mask)
        logits = proj(logits)
        logits = nn.Sigmoid(logits).round().float()
        return logits
    def predict_similarity(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        proj = nn.Linear(self.config.hidden_size*2, 1)
        input_ids = torch.concat((input_ids1,input_ids2), dim = -1)
        attention_mask = torch.concat((attention_mask1, attention_mask2), dim = -1)
        logits = self.forward(input_ids, attention_mask)
        logits = proj(logits)
        return logits
    def predict_sentiment(self, input_ids, attention_mask):
        proj = nn.Linear(self.config.hidden_size, 5)
        logits = self.forward(input_ids= input_ids, attention_mask= attention_mask)
        logits = proj(logits)
        return logits
    

def train(config):
    sst_dev_acc = []
    quora_dev_acc = []
    sts_dev_acc = []
    best_sst_dev_acc = []
    best_quora_dev_acc = []
    best_sts_dev_acc = []
    sst_ds = load_data(config.sst_training, flag = 1)
    sst_ds = MultitaskDataset(sst_ds)
    sst_dl = DataLoader(sst_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    sst_dev_ds = load_data(config.sst_dev, flag = 1)
    sst_dev_ds = MultitaskDataset(sst_dev_ds)
    sst_dev_dl = DataLoader(sst_dev_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    quora_ds = load_data(config.quora_traning, flag = 2)
    quora_ds = MultitaskDataset(quora_ds)
    quora_dl = DataLoader(quora_ds, config.bs, shuffle = True, collate_fn = collate_fn)
    quora_dev_ds = load_data(config.quora_dev, flag = 2)
    quora_dev_ds = MultitaskDataset(quora_dev_ds)
    quora_dev_dl = DataLoader(quora_dev_ds, config.bs, shuffle = True, collate_fn = collate_fn)
    sts_ds = load_data(config.sts_training, flag = 2)
    sts_ds = MultitaskDataset(sts_ds)
    sts_dl = DataLoader(sts_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    sts_dev_ds = load_data(config.sts_dev, flag = 2)
    sts_dev_ds = MultitaskDataset(sts_dev_ds)
    sts_dev_dl = DataLoader(sts_dev_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    model = MultiBert(config)
    optimizer = AdamW(model.parameters(), config.lr)
    for e in range(config.num_epochs):
        for batch in tqdm(sst_dl, desc = f'Epoch: {e+1}'):
            model.train()
            input_ids, attention_mask, labels, *_ = batch
            optimizer.zero_grad()
            logits = model.predict_sentiment(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
        for batch in tqdm(quora_dl, desc = f'Epoch: {e+1}'):
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels, *_ = batch
            optimizer.zero_grad()
            logits = model.predict_paraphrase(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = nn.BCELoss()(logits, labels)
            loss.backward()
            optimizer.step()
        for batch in tqdm(sts_dl, desc = f'Epoch: {e+1}'):
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels, *_ = batch
            optimizer.zero_grad()
            logits = model.predict_similarity(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = nn.MSELoss(logits, labels)
            loss.backward()
            optimizer.step()
        multitask_accuracy = MultiEvaluation(model, sst_dl, quora_dl, sts_dl)
        multitask_dev_accuracy = MultiEvaluationTest(model, sst_dev_dl, quora_dev_dl, sts_dev_dl)
        if multitask_dev_accuracy['sst_accuracy']>best_sst_dev_acc:
            best_sst_dev_acc = multitask_dev_accuracy['sst_accuracy']
        if multitask_dev_accuracy['quora_accuracy']>best_quora_dev_acc:
            best_quora_dev_acc = multitask_dev_accuracy['quora_accuracy']
        if multitask_dev_accuracy['sts_accuracy']>best_sts_dev_acc:
            best_sts_dev_acc = multitask_dev_accuracy['sts_accuracy']
        
        save_model(model, optimizer, args, config)



def test_multitask(model, test_set):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'add config')
    parser.add_argument('--seed', type = int, default= 11711)
    parser.add_argument('--option', type = str, help = 'choose pretrain mode or finetune mode', choices = ('pretrain','finetune'), default= 'finetune')
    parser.add_argument('--num_epochs', type = int, default= 10)
    parser.add_argument('--lr', type = float, default= 1e-5)
    parser.add_argument('--bs', type = int, default = 8)
    parser.add_argument('--dropout_rate', type = float, default = 0.3)
    parser.add_argument('--use_gpu', action = 'store_true')
    args = parser.parse_args()
    seed_everything(args.seed)
    config = SimpleNamespace(
        filepath = 'sst.pt',
        option = args.option,
        num_epochs = args.num_epochs,
        lr = args.lr,
        bs = args.bs,
        dropout_rate = args.dropout_rate,
        hidden_size = 768,
        sst_training = './data/sst_ds/ids-sst-train.csv',
        sst_dev = './data/sst_ds/ids-sst-dev.csv',
        sst_test = './data/sst_ds/ids-sst-test-student.csv',
        sst_dev_out = './prediction/'+args.option+'-sst-dev.csv',
        sst_test_out = './prediction/'+args.option+'-sst-test.csv',
        use_gpu = args.use_gpu,
    )
    train(config)
    