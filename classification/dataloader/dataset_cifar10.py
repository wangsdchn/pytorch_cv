from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


class Cifar10(Dataset):
    def __init__(self, data_dict):
        self.dataset = CIFAR10(root=str(data_dict.data_root), train=data_dict.is_train, transform=data_dict.transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        data = {'images': image, 'labels': label, 'images_path': ''}
        return data
