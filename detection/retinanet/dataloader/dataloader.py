from torch.utils.data import Dataset, DataLoader
import torch
import json

class DatasetBase(Dataset):
    def __init__(self, dataroot, transforms=None, is_train=True, input_size=600):
        self.data_list = []
        with dataroot.open('r') as f:
            for line in f:
                self.data_list.append(json.loads(line.strip('\n')))
        self.is_train = is_train
        self.transforms = transforms
        self.input_size = input_size
