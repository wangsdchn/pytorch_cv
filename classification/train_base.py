# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from easydict import EasyDict
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import sys
import shutil
from tqdm import tqdm
sys.path.extend('..')

# custom
from contribs.optimizers_def import create_optimizer
from contribs.loss import create_criterion
from contribs.metric import AverageMeter, accuracy
import classification.dataloader as dataloader


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_ids)
        self.acc, self.best_acc = 0.0, 0.0
        self.model = self.init_model()

        # dataset
        self.train_dataloader, self.test_dataloader = self.load_dataset()

        # loss
        self.criterion = create_criterion(self.args)

        # cuda
        self.use_cuda = torch.cuda.is_available()

        # optimizer
        self.optimizer = create_optimizer(self.model, self.args)
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).cuda()
            self.optimizer = nn.DataParallel(self.optimizer).cuda()

        # logger
        self.writer = SummaryWriter(self.args.logdir)
        self.iters = 0

    def init_model(self):
        model = models.__dict__[self.args.arch](pretrained=False)
        num_in_feature = model.fc.in_features
        model.fc = nn.Linear(num_in_feature, self.args.num_classes)
        return model

    def load_dataset(self):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_train = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.args.image_size + 30),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        data_init_train = EasyDict({
            'data_root': self.args.train_data_path,
            'transforms': transform_train,
        })
        data_init_test = EasyDict({
            'data_root': self.args.test_data_path,
            'transforms': transform_test,
        })
        self.dataset = dataloader.__dict__[self.args.dataset_name]
        train_dataset = self.dataset(data_init_train)
        test_dataset = self.dataset(data_init_test)
        if self.args.num_samples == -1:
            self.args.num_samples = len(train_dataset)
        random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.args.num_samples)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                      num_workers=self.args.num_workers, sampler=random_sampler, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size,
                                     num_workers=self.args.num_workers, shuffle=False, pin_memory=True)
        return train_dataloader, test_dataloader

    def train(self, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        pbar = tqdm(self.train_dataloader)
        for batch_idx, data in enumerate(pbar):
            self.iters += 1
            images, labels = data['images'], data['labels']
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prec1 = accuracy(outputs.data, labels.data, topk=(1,))
            loss = loss.view(-1)
            losses.update(loss.data[0], images.size(0))
            top1.update(prec1[0], images.size(0))

            if batch_idx % self.args.log_interval == 0:
                print('epoch:{}, iter:{}/{}, loss:{}, acc:{}'.format(epoch, batch_idx, self.iters, losses.avg, top1.avg))
                losses.reset(), top1.reset()

    def test(self, epoch):
        self.model.eval()
        top1 = AverageMeter()
        all_result = []
        for batch_idx, (images, labels, images_path) in enumerate(dataloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            prec1 = accuracy(outputs.data, labels.data, topk=(1,))
            top1.update(prec1[0], images.size(0))

            if self.args.is_save:
                probs, preds = outputs.softmax(dim=1).max(dim=1)
                probs, preds = probs.view(-1), preds.view(-1)
                for idx in range(images.size(0)):
                    result = '{}\t{}\t{}\t{}\n'.format(images_path[idx], labels[idx].item(), preds[idx].item(),
                                                       probs[idx].item())
                    all_result.append(result)
        if self.args.is_save:
            with open('result.txt', 'w') as f:
                f.writelines(all_result)
        self.acc = top1.avg
        print('Test epoch:{}, acc:{}'.format(epoch, top1.avg))

    def save_chackpoint(self, epoch):
        state = {
            'state_dict': self.model.state_dict(),
            'acc': self.acc,
            'epoch': epoch,
            'args': self.args
        }
        self.args.checkpoint_save_name.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(self.args.checkpoint_save_name))
        if self.acc > self.best_acc:
            self.best_acc = self.acc
            best_filepath = str(self.args.checkpoint_save_name).replace('.pth', '_epoch{}_acc{}.pth'.format(epoch, round(self.acc, 4)))
            shutil.copyfile(str(self.args.checkpoint_save_name), best_filepath)


def main():
    args = EasyDict({
        'arch': 'resnet18',
        'phase': 'train',
        'num_classes': 2,
        'image_size': 224,
        'gpu_ids': '0',
        'total_epochs': 10000,
        'num_samples': -1,
        'optimizer_type': 'adam',
        'loss_type': 'CrossEntropyLoss',
        'base_lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.99,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'log_interval': 10,

        'dataset_name': 'ImageFolder',
        'train_data_path': Path('dataset/train.txt'),
        'test_data_path': Path('dataset/test.txt'),
        'train_batch_size': 4,
        'test_batch_size': 1,
        'num_workers': 0,

        'logdir': 'train_log/',
        'is_save': True,
        'checkpoint_save_name': Path('./checkpoints/model.pth')
    })
    trainer = BaseTrainer(args)
    if args.phase == 'test':
        trainer.test(epoch=0)
        exit()
    for epoch in range(args.total_epochs):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.save_chackpoint(epoch)


if __name__ == '__main__':
    main()
