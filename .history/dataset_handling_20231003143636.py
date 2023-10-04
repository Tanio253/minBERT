from csv import DictReader
from torch.utils.data import Dataset, dataloader
def load_data(filepath, flag = 2):
    data = []
    with open(filepath) as file:
        reader = DictReader(file, delimiter = '\t')
        for r in reader:
            if flag == 2:
                if len(r)==4 : continue
                data.append((r['id'], r['sentence1'], r['sentence2'], r['is_duplicate']))
            if flag == 1:
                if len(r)==3: continue
                data.append((r['id'], r['sentence'], r['sentiment']))
    return data
class MultitaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
print(load_data('./data/quora/quora-train.csv')[0])