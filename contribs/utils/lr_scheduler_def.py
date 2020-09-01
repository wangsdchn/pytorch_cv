import torch


def adjust_learning_rate(optimizer, base_lr, epoch, step):
    lr = base_lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
