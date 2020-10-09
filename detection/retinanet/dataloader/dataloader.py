from torch.utils.data import Dataset, DataLoader
import torch
import json
import cv2
from PIL import Image
from detection.retinanet.utils.encoder import DataEncoder
from detection.retinanet.utils.augument import resize


class DatasetBase(Dataset):
    def __init__(self, dataroot, transforms=None, is_train=True, input_size=600):
        data_list = []
        with open(str(dataroot), 'r') as f:
            for line in f:
                data_list.append(json.loads(line.strip('\n')))
        self.is_train = is_train
        self.transforms = transforms
        self.input_size = input_size
        self.encoder = DataEncoder(input_size=input_size)
        self.images, self.bboxes, self.labels = [], [], []
        for data in data_list:
            self.images.append(data['image_name'])
            bboxes, labels = [], []
            if 'objects' in data:
                for obj in data['objects']:
                    bboxes.append(obj['bbox'])
                    labels.append(obj['label'])
            self.bboxes.append(torch.tensor(bboxes).float())
            self.labels.append(torch.tensor(bboxes).long())

    def __len__(self):
        return len(self.images)

    def _augument(self, image, bboxes):
        image, bboxes = resize(image, bboxes, size=self.input_size)
        return image, bboxes

    def __getitem__(self, item):
        # image = Image.open(self.images[item]).convert('RGB')
        image = self.images[item]
        bboxes = self.bboxes[item]
        labels = self.labels[item]
        # image, bboxes = self._augument(image, bboxes)
        if self.transforms:
            image = self.transforms(image)
        data = {
            'images': image,
            'bboxes': bboxes,
            'labels': labels
        }
        return data

    def collate_fn(self, batch):
        images = torch.stack(batch['images'], dim=0)
        bboxes, labels = batch['bboxes'], batch['labels']
        cls_targets, loc_targets = [], []
        for i in range(images.size(0)):
            loc_target, cls_target = self.encoder.encode(bboxes, labels)
            cls_targets.append(cls_target)
            loc_targets.append(loc_target)
        data = {
            'images': images,
            'loc_targets': torch.stack(loc_targets, dim=0),
            'cls_target': torch.stack(cls_targets, dim=0)
        }
        return data


def loader_test():
    val_json = '../dataset/coco_val.json'
    dataset = DatasetBase(val_json)
    for data in dataset:
        print(data)


if __name__ == '__main__':
    loader_test()
