from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, use_meanteacher=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.use_meanteacher = use_meanteacher

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        if self.use_meanteacher:
            img_2 = img.copy()

        if self.transform is not None:
            img = self.transform(img)
        
        if self.use_meanteacher:
            if self.transform is not None:
                img_2 = self.transform(img_2)
            return img, img_2, fname, pid, camid, index

        return img, fname, pid, camid, index
