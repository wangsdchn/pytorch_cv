import torch.optim as optim


def create_optimizer(model, args):
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(args.beta1, args.beta2), eps=1e-08,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    return optimizer
