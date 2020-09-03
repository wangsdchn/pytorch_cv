# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True
