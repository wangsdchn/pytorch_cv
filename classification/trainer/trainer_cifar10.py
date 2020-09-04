from .trainer_base import BaseTrainer
from pathlib import Path


class Cifar10Trainer(BaseTrainer):
    def _init_add_args(self):
        self.args.train_data_path = Path('/home/work/dataset/public/')
        self.args.test_data_path = Path('/home/work/dataset/public/')
        self.args.image_size = 32
        self.args.num_classes = 10
        self.args.train_batch_size = 512
        self.args.test_batch_size = 128
        self.args.dataset_name = 'Cifar10'
