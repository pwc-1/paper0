import numpy as np
from PIL import Image
from mindspore.dataset.vision import py_transforms

RANDOM_RESOLUTIONS = [512, 768, 1024, 1280, 1536]


class BatchToTensor(object):
    def __call__(self, imgs):

        return [py_transforms.ToTensor()(img) for img in imgs]

