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
from contribs.utils.optimizers_def import create_optimizer
from contribs.utils.loss import create_criterion
from contribs.utils.metric import AverageMeter, accuracy
from contribs.utils.lr_scheduler_def import adjust_learning_rate
from contribs.utils.set_seeds import set_seeds
from contribs.utils.init_net import init_params
import classification.dataloader as dataloader
import classification.backbones as backbones


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_ids)
        self.acc, self.best_acc = 0.0, 0.0

        # cuda
        self.use_cuda = torch.cuda.is_available()

        # model relative define
        self.criterion = create_criterion(self.args)
        self.optimizer = create_optimizer(self.model, self.args)
        self.model = self.init_model()
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).cuda()
            self.optimizer = nn.DataParallel(self.optimizer).cuda()
            self.criterion = nn.DataParallel(self.criterion).cuda()

        # dataset
        self.init_transforms()
        self.init_dataset()

        # logger
        self.writer = SummaryWriter(self.args.logdir)
        self.iters = 0

    def init_model(self):
        if self.args.arch_type == 'custom_define':
            model = backbones.__dict__[self.args.arch](pretrained=False, num_classes=self.args.num_classes)
        else:
            model = models.__dict__[self.args.arch](pretrained=False, num_classes=self.args.num_classes)
            # num_in_feature = model.fc.in_features
            # model.fc = nn.Linear(num_in_feature, self.args.num_classes)
        if self.args.checkpoint_pretrained:
            checkpoint = torch.load(self.args.checkpoint_pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            if 'state_dict_optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['state_dict_optimizer'])
        else:
            init_params(model)
        return model

    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_train = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.args.image_size + 30),
            transforms.RandomCrop(self.args.image_size),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        self.data_init_train = EasyDict({
            'data_root': self.args.train_data_path,
            'transforms': transform_train,
        })
        self.data_init_test = EasyDict({
            'data_root': self.args.test_data_path,
            'transforms': transform_test,
        })

    def init_dataset(self):
        self.dataset = dataloader.__dict__[self.args.dataset_name]
        train_dataset = self.dataset(self.data_init_train)
        test_dataset = self.dataset(self.data_init_test)
        if self.args.num_samples == -1:
            self.args.num_samples = len(train_dataset)
        random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=self.args.num_samples)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                      num_workers=self.args.num_workers, sampler=random_sampler, pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size,
                                     num_workers=self.args.num_workers, shuffle=False, pin_memory=True)

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
            if self.args.optimizer_type == 'sgd':
                adjust_learning_rate(self.optimizer, self.args.base_lr, epoch, self.args.lr_decay_epoch)
            prec1 = accuracy(outputs.data, labels.data, topk=(1,))
            loss = loss.view(-1)
            losses.update(loss.data[0], images.size(0))
            top1.update(prec1[0], images.size(0))
            self.writer.add_scalar('train/loss', loss.data[0], self.iters)
            self.writer.add_scalar('train/acc', top1.val, self.iters)

            if batch_idx % self.args.log_interval == 0:
                print('epoch:{}, iter:{}/{}, loss:{}, acc:{}'.format(epoch, batch_idx, self.iters, losses.avg, top1.avg))
                losses.reset(), top1.reset()

    def test(self, epoch):
        self.model.eval()
        top1 = AverageMeter()
        all_result = []
        for batch_idx, data in enumerate(dataloader):
            images, labels, images_path = data['images'], data['labels'], data['images_path']
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            prec1 = accuracy(outputs.data, labels.data, topk=(1,))
            top1.update(prec1[0], images.size(0))
            self.writer.add_scalar('test/acc', top1.val, self.iters)

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
            'state_dict_optimizer': self.optimizer.state_dict(),
            'acc': self.acc,
            'epoch': epoch,
            'iter': self.iters,
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
        'arch_type': 'custom_define',
        'phase': 'train',
        'num_classes': 2,
        'image_size': 224,
        'gpu_ids': '0',
        'total_epochs': 10000,
        'num_samples': -1,
        'optimizer_type': 'adam',
        'loss_type': 'CrossEntropyLoss',
        'base_lr': 0.001,
        'lr_decay_epoch': 10,
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
        'checkpoint_pretrained': None,
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
    set_seeds(0)
    # main()
