# -*- coding: utf-8 -*-
import torch
from .dataloader.folder_random_data import RandomDataset


class TrainBase:
    def __init__(self):
        pass

    def init_model(self):
        pass

    def load_dataset(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save_chackpoint(self):
        pass
