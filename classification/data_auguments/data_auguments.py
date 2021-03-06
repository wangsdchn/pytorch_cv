import torch
import cv2
from PIL import Image
import numpy as np
import random

__all__ = ['RandomDirection', 'RandomQuality']


class RandomDirection:
    def __init__(self, directions=None, prob=1.0):
        if directions is None:
            directions = [0, 90, 180, 270]
        self.directions = directions
        self.prob = prob

    def __call__(self, image_pil):
        if np.random.uniform() < self.prob:
            direction = random.choice(self.directions)
            image_pil = image_pil.rotate(direction)
        return image_pil


class ChannelsShuffle:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, image_pil):
        if np.random.uniform() < self.prob:
            BGR = cv2.split(np.asarray(image_pil))
            np.random.shuffle(BGR)
            image_pil = Image.fromarray(cv2.merge(BGR))
        return image_pil


class RandomQuality:
    def __init__(self, low_quality=80, high_quality=100):
        self.low_quality = low_quality
        self.high_quality = high_quality

    def __call__(self, image_pil):
        quality = random.randint(self.low_quality, self.high_quality)
        image_cv = np.asarray(image_pil)
        _, image_data = cv2.imencode('.jpg', image_cv[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        image_cv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_pil = Image.fromarray(image_cv[:, :, ::-1])
        return image_pil


if __name__ == '__main__':
    image = Image.open('test.jpg')
    randomop = RandomQuality(10, 20)
    image = randomop(image)
    image_cv = np.asarray(image)
    cv2.imshow('op', image_cv[:, :, ::-1])
    cv2.waitKey()
