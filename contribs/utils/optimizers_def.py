import torch.optim as optim


def create_optimizer(model, args):
    """
    为不同层设置不同学习率
    model.parameters() 为参数列表，可自由设置，根据名字过滤
    optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

    load_state_dict(state_dict) 恢复优化器状态
    """
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(args.beta1, args.beta2), eps=1e-08,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    return optimizer
