import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, ConcatDatasetWithIndex

class Base(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex
		
class HousesTrain(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/content/houses/My_HouseGan"
        with open("/content/train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class HousesValidation(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/content/houses/_MyHouseGan/"
        with open("/content/validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys