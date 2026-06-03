"""
数据集构建
"""
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

config = Config()

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item][0]
        label = self.data[item][1]
        return text, label

if __name__ == "__main__":
    train_data = {}