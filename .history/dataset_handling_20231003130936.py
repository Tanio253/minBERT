import csv 
from torch.utils.data import Dataset, dataloader
def load_data(filepath):
    
class MultitaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    