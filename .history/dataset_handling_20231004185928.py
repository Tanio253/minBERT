from csv import reader
from torch.utils.data import Dataset
def load_data(filepath, flag = 2):
    data = []
    with open(filepath) as file:
        for r in reader(file, delimiter = '\t'):
            if flag == 2:
                data.append((r[1], r[2], r[3], r[4]))
            if flag == 1:
                data.append((r[1], r[2], r[3]))
    return data[1:]
class MultitaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
print(load_data('/home/tanio/cs224n/minBERT/data/sss/ids-sst-train.csv', flag = 1)[0])