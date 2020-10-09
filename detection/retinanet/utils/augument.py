from PIL import Image
import torch


def resize(image, bboxes, size):
    w, h = image.size
    if isinstance(size, int):
        size = (size, size)
    scale_w = size[0] / w
    scale_h = size[1] / h
    image = image.resize(size, Image.BILINEAR)
    bboxes = bboxes * torch.tensor([scale_w, scale_h, scale_w, scale_h]).float()
    return image, bboxes