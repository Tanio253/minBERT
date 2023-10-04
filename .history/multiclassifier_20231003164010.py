import torch
import tqdm
import torch.nn as nn
from dataset_handling import load_data, MultitaskDataset
from bert import BertModel
import argparse, random, numpy as np
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class MultiBert(nn.Module):
    def __init__(self, config, num_labels):
        super.__init__(self)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.do = nn.Dropout(p = config.dropout_rate)
        self.proj = nn.Linear(config.hidden_state, num_labels)
        if config.option == 'pretrain':
            for p in self.bert.parameters():
                p.requires_grad = False
        if config.option == 'finetune':
            for p in self.bert.parameters():
                p.requires_grad = True
    def forward(self, input_ids, attention_mask ):
        bert_output = self.bert(input_ids, attention_mask)
        bert_output = self.do(bert_output['pooler_output'])
        logits = self.proj(bert_output)
        return logits 
    def predict_paraphrase(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        input_ids = torch.concat((input_ids1,input_ids2), dim = -1)
        attention_mask = torch.concat((attention_mask1, attention_mask2), dim = -1)
        logits = self.forward(input_ids, attention_mask)
        return logits
    def predict_similarity(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        input_ids = torch.concat((input_ids1,input_ids2), dim = -1)
        attention_mask = torch.concat((attention_mask1, attention_mask2), dim = -1)
        logits = self.forward(input_ids, attention_mask)
        return logits
    def predict_sentiment(self, input_ids, attention_mask):
        logits = self.forward(input_ids= input_ids, attention_mask= attention_mask)
        return logits
    def train(self, config):
        sst_ds = load_data(config.sst_data, flag = 1)
        quora_ds = load_data(config.quora_data, flag = 2)
        sts_ds = load_data(config.sts_data, flag = 2)
        sst_ds = MultitaskDataset(sst_ds)
        sst_dl = DataLoader()


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

    )
    MTbert = MultiBert(config)
    MTbert.train()