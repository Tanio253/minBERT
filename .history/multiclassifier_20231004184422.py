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
from functools import partial
from tokenizer import BertTokenizer
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
        super().__init__()
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
    

def custom_collate(data, device, sentpair: bool):
    if sentpair is not None:
        sents = [d[1] for d in data]
        sent_ids = [d[0] for d in data]
        labels = [d[2] for d in data]
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_tokenizer = bert_tokenizer(sents, return_tensors= 'pt', padding = True, truncation = True  )
        input_ids, attention_mask = bert_tokenizer['input_ids'], bert_tokenizer['attention_mask']
        input_ids = input_ids.to(device)
        labels = torch.tensor(labels, device = device)
        attention_mask = attention_mask.to(device)
    else:
        sents1 = [d[1] for d in data]
        sents2 = [d[2] for d in data]
        sent_ids = [d[0] for d in data]
        labels = [d[3] for d in data]
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_tokenizer = bert_tokenizer(sents1, return_tensors= 'pt', padding = True, truncation = True  )
        input_ids1, attention_mask1 = bert_tokenizer['input_ids'], bert_tokenizer['attention_mask']
        bert_tokenizer = bert_tokenizer(sents2, return_tensors= 'pt', padding = True, truncation = True  )
        input_ids2, attention_mask2 = bert_tokenizer['input_ids'], bert_tokenizer['attention_mask']
        input_ids1 = input_ids1.to(device)
        attention_mask1 = attention_mask1.to(device)
        input_ids2 = input_ids2.to(device)
        attention_mask2 = attention_mask2.to(device)
        labels = torch.tensor(labels, device = device)
        input_ids = tuple(input_ids1, input_ids2)
        attention_mask = tuple(attention_mask1, attention_mask2)
        sents = tuple(sents1, sents2)
    
    return  input_ids, attention_mask, labels, sents, sent_ids

def train(config):
    device = 'cuda' if config.use_gpu else 'cpu'
    sentpair_collate_fn = partial(custom_collate, device = device, sentpair = 1)
    collate_fn = partial(custom_collate, device = device, sentpair = 0)
    best_multitask_accuracy = 0
    best_sst_dev_acc = []
    best_quora_dev_acc = []
    best_sts_dev_acc = []
    sst_ds = load_data(config.sst_training, flag = 1)
    sst_ds = MultitaskDataset(sst_ds)
    sst_dl = DataLoader(sst_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    sst_dev_ds = load_data(config.sst_dev, flag = 1)
    sst_dev_ds = MultitaskDataset(sst_dev_ds)
    sst_dev_dl = DataLoader(sst_dev_ds, config.bs, shuffle = True, collate_fn= collate_fn)
    quora_ds = load_data(config.quora_training, flag = 2)
    quora_ds = MultitaskDataset(quora_ds)
    quora_dl = DataLoader(quora_ds, config.bs, shuffle = True, collate_fn = sentpair_collate_fn)
    quora_dev_ds = load_data(config.quora_dev, flag = 2)
    quora_dev_ds = MultitaskDataset(quora_dev_ds)
    quora_dev_dl = DataLoader(quora_dev_ds, config.bs, shuffle = True, collate_fn = sentpair_collate_fn)
    sts_ds = load_data(config.sts_training, flag = 2)
    sts_ds = MultitaskDataset(sts_ds)
    sts_dl = DataLoader(sts_ds, config.bs, shuffle = True, collate_fn= sentpair_collate_fn)
    sts_dev_ds = load_data(config.sts_dev, flag = 2)
    sts_dev_ds = MultitaskDataset(sts_dev_ds)
    sts_dev_dl = DataLoader(sts_dev_ds, config.bs, shuffle = True, collate_fn= sentpair_collate_fn)
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
        if multitask_dev_accuracy['sts_pearson']>best_sts_dev_acc: # (-1,1)
            best_sts_dev_acc = multitask_dev_accuracy['sts_pearson']
        overall_accuracy = multitask_dev_accuracy['sst_accuracy']+ multitask_dev_accuracy['quora_accuracy'] + (multitask_dev_accuracy['sts_pearson']+1)/2.0
        if  overall_accuracy > best_multitask_accuracy:
            best_multitask_accuracy = overall_accuracy
            save_model(model, optimizer, config)
        sst_accuracy = multitask_accuracy['sst_accuracy']
        sst_dev_accuracy = multitask_dev_accuracy['sst_accuracy']
        quora_accuracy = multitask_accuracy['quora_accuracy']
        quora_dev_accuracy = multitask_dev_accuracy['quora_accuracy']
        sts_accuracy = multitask_accuracy['sts_accuracy']
        sts_dev_accuracy = multitask_dev_accuracy['sts_accuracy']
        print(f'Epoch {e+1} sst training: {sst_accuracy: .4f} sst dev: {sst_dev_accuracy: .4f} \
              quora training: {quora_accuracy: .4f} quora dev: {quora_dev_accuracy: .4f} \
              sts training: {sts_accuracy: .4f} sts dev: {sts_dev_accuracy: .4f} \
              overall accuracy: {overall_accuracy: .4f}')
    print(f'Best overall accuracy: {best_multitask_accuracy: .4f}')


def test(config):
    with torch.no_grad():
        device = 'cuda' if config.use_gpu else 'cpu'
        collate_fn = partial(custom_collate, device = device, sentpair = 0)
        sentpair_collate_fn = partial(custom_collate, device = device, sentpair = 1)
        sst_test = load_data(config.sst_test, flag = 1)
        sst_test = MultitaskDataset(sst_test)
        sst_test = DataLoader(sst_test, config.bs, collate_fn= collate_fn)
        sst_dev = load_data(config.sst_dev, flag = 1)
        sst_dev = MultitaskDataset(sst_dev)
        sst_dev = DataLoader(sst_dev, config.bs, collate_fn= collate_fn)
        test_result = MultiEvaluationTest(model, sst_test)
        eval_result = MultiEvaluation(model, sst_dev)
        quora_test = load_data(config.quora_test, flag = 1)
        quora_test = MultitaskDataset(quora_test)
        quora_test = DataLoader(quora_test, config.bs, collate_fn= sentpair_collate_fn)
        quora_dev = load_data(config.quora_dev, flag = 1)
        quora_dev = MultitaskDataset(quora_dev)
        quora_dev = DataLoader(quora_dev, config.bs, collate_fn= sentpair_collate_fn)
        test_result = MultiEvaluationTest(model, quora_test)
        eval_result = MultiEvaluation(model, quora_dev)
        sts_test = load_data(config.sts_test, flag = 1)
        sts_test = MultitaskDataset(sts_test)
        sts_test = DataLoader(sts_test, config.bs, collate_fn= sentpair_collate_fn)
        sts_dev = load_data(config.sts_dev, flag = 1)
        sts_dev = MultitaskDataset(sts_dev)
        sts_dev = DataLoader(sts_dev, config.bs, collate_fn= sentpair_collate_fn)
        test_result = MultiEvaluationTest(model, sts_test)
        eval_result = MultiEvaluation(model, sts_dev)
        save_info = torch.load(config.filepath)
        model = MultiBert(save_info['config'])
        model.load_state_dict(save_info['model'])
        model = model.to(device)
        

        with open(config.sst_dev_out, "w+") as f:
                print(f"sst dev acc :: {eval_result['sst_accuracy'] :.4f}")
                f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
                for i, s, p in zip(eval_result['sst_sent_ids'], eval_result['sst_sents'], eval_result['sst_pred']):
                    f.write(f"{i} , {s}, {p} \n")

        with open(config.sst_test_out, "w+") as f:
            f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
            for i, s, p  in zip(test_result['sst_sent_ids'], test_result['sst_sents'], test_result['sst_pred'] ):
                f.write(f"{i} , {s}, {p} \n")
        
        with open(config.quora_dev_out, "w+") as f:
                print(f"quora dev acc :: {eval_result['quora_accuracy'] :.4f}")
                f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
                for i, s, p in zip(eval_result['quora_sent_ids'], eval_result['quora_sents'], eval_result['quora_pred']):
                    f.write(f"{i} , {s}, {p} \n")

        with open(config.quora_test_out, "w+") as f:
            f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
            for i, s, p  in zip(test_result['quora_sent_ids'], test_result['quora_sents'], test_result['quora_pred'] ):
                f.write(f"{i} , {s}, {p} \n")
        
        with open(config.sts_dev_out, "w+") as f:
                print(f"sts dev acc :: {eval_result['sts_accuracy'] :.4f}")
                f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
                for i, s, p in zip(eval_result['sts_sent_ids'], eval_result['sts_sents'], eval_result['sts_pred']):
                    f.write(f"{i} , {s}, {p} \n")

        with open(config.sts_test_out, "w+") as f:
            f.write(f"Id \t Sentence \t Predicted_Sentiment \n")
            for i, s, p  in zip(test_result['sts_sent_ids'], test_result['sts_sents'], test_result['sts_pred'] ):
                f.write(f"{i} , {s}, {p} \n")


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
        filepath = 'multitask.pt',
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
        quora_training = './data/quora/quora-train.csv',
        quora_dev = './data/quora/quora-dev.csv',
        quora_test = './data/quora/quora-test-student.csv',
        quora_dev_out = './prediction/'+args.option+'-quora-dev.csv',
        quora_test_out = './prediction/'+args.option+'-quora-test.csv',
        sts_training = './data/sts/sts-train.csv',
        sts_dev = './data/sts/sts-dev.csv',
        sts_test = './data/sts/sts-test-student.csv',
        sts_dev_out = './prediction/'+args.option+'-sts-dev.csv',
        sts_test_out = './prediction/'+args.option+'-sts-test.csv',
        use_gpu = args.use_gpu,
    )
    train(config)
    test(config)
    