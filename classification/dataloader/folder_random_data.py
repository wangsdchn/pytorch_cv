# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from PIL import Image


class RandomDataset(Dataset):
    def __init__(self, dataset_length, input_size=224, transforms=None):
        self.length = dataset_length
        self.input_size = (input_size, input_size, 3)
        self.transforms = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        prob = torch.rand(1).item()
        if prob < 0.5:
            random_tensor = torch.randint(0, 130, self.input_size, dtype=torch.uint8)
            label = 0
        else:
            random_tensor = torch.randint(120, 200, self.input_size, dtype=torch.uint8)
            label = 1
        img_pil = Image.fromarray(random_tensor.numpy())
        random_tensor = self.transforms(img_pil)
        return random_tensor, label

