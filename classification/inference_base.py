# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
from pathlib import Path
from easydict import EasyDict

# custom
from classification import backbones


def is_image_file(imgname):
    IMG_EXTS = ['.jpg', '.jpeg', '.png']
    is_image_flag = True if os.path.splitext(imgname.lower())[1] in IMG_EXTS else False
    return is_image_flag


class ImageFolder(Dataset):
    def __init__(self, image_path, transform):
        self.transform = transform
        if image_path.suffix == '.txt':
            with image_path.open('r') as f:
                self.images_list = f.readlines()
        else:
            self.images_list = [image for image in image_path.rglob('*') if is_image_file(str(image))]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        image_path = self.images_list[item].strip('\n')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path


def parse_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        nkey = key
        if key.startswith('module.'):
            nkey = key[7:]
        new_state_dict[nkey] = value
    return new_state_dict


class Infer:
    def __init__(self, arch, image_size=224, num_classes=2, checkpoint=None, arch_type='default'):
        self.use_cuda = torch.cuda.is_available()
        self.model = self._init_model(arch, num_classes=num_classes, checkpoint=checkpoint, arch_type=arch_type)
        if self.use_cuda:
            self.model = self.model.cuda()
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])

    def _init_model(self, arch, num_classes=2, checkpoint=None, arch_type='default'):
        if arch_type == 'custom_define':
            model = backbones.__dict__[arch](num_classes=num_classes)
        else:
            model = models.__dict__[arch](pretrained=False, num_classes=num_classes)
        if checkpoint is not None:
            print("Loading checkpoint from {}".format(checkpoint))
            new_state_dict = parse_state_dict(torch.load(checkpoint)['state_dict'])
            model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def test_batch(self, image_path, batch_size, workers):
        test_dataset = ImageFolder(image_path, self.transform_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                                     pin_memory=True)
        all_result = []
        for batch_idx, (images, images_path) in enumerate(test_dataloader):
            if self.use_cuda:
                images = images.cuda()
            outputs = self.model(images)
            probs, preds = outputs.softmax(dim=1).max(1)
            probs, preds = probs.view(-1), preds.view(-1)
            for idx in range(images.size(0)):
                result = '{}\t{}\t{}\n'.format(images_path[idx], preds[idx].item(), probs[idx].item())
                all_result.append(result)
        with open('result.txt', 'w') as f:
            f.writelines(all_result)

    def test_image(self, image):
        image_tensor = self.transform_test(image)
        image_tensor = image_tensor.unsqueeze(0)
        if self.use_cuda:
            image_tensor = image_tensor.cuda()
        output = self.model(image_tensor)
        prob, pred = output.softmax(dim=1).max(1)
        if self.use_cuda:
            prob, pred = prob.cpu().item(), pred.cpu().item()
        else:
            prob, pred = prob.item(), pred.item()
        print(pred, prob)


def run():
    args = EasyDict({
        'arch': 'resnet18',
        'arch_type': 'custom_define',
        'num_classes': 2,
        'image_size': 224,
        'gpu_ids': '0',
        'image_path': Path('dataset/test.txt'),
        'batch_size': 4,
        'num_workers': 0,
        'checkpoint': Path('./checkpoints/model.pth')
    })
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    inference = Infer(arch=args.arch, image_size=args.image_size, num_classes=args.num_classes, checkpoint=args.checkpoint)
    if is_image_file(args.image_path):
        image = Image.open(args.image_path).convert('RGB')
        inference.test_image(image)
    else:
        inference.test_batch(args.image_path, args.batch_size, args.num_workers)


if __name__ == '__main__':
    run()
