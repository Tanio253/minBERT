from csv import reader
from torch.utils.data import Dataset
def load_data(filepath, flag = 2):
    data = []
    with open(filepath) as file:
        reader = reader(file, delimiter = '\t')
        for r in reader:
            if flag == 2:
                if len(r)==4 : continue
                data.append((r[0], r[1], r[2], r[3]))
            if flag == 1:
                if len(r)==3: continue
                data.append((r[0], r[1], r[2]))
    return data
class MultitaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
