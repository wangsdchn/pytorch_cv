# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, data_dict):
        self.transforms = data_dict.transforms
        data_root = data_dict.data_root
        assert data_root.suffix == '.txt'
        self.image_list = []
        with data_root.open('r') as f:
            self.image_list = f.readlines()
        self.length = len(self.image_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image_path, label = self.image_list[item].strip('\n').split('\t')
        image_pil = Image.open(image_path).convert('RGB')
        image_tensor = self.transforms(image_pil)
        data = {'images': image_tensor, 'labels': int(label), 'images_path': image_path}
        return data
