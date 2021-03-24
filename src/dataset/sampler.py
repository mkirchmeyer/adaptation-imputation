from copy import deepcopy

import torchvision
import torch.utils.data
import random
import numpy as np

"""
Adapted from https://github.com/galatolofederico/pytorch-balanced-batch
"""


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, in_memory=False, is_criteo=False):
        self.dataset = {}
        self.dataset_backup = {}
        self.balanced_max = 0
        self.in_memory = in_memory
        self.is_criteo = is_criteo
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))

        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.dataset_backup = deepcopy(self.dataset)

    def __iter__(self):
        while len(self.dataset[self.keys[self.currentkey]]) > 0:
            yield self.dataset[self.keys[self.currentkey]].pop()
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        for label in self.dataset_backup:
            np.random.shuffle(self.dataset_backup[label])
        self.dataset = deepcopy(self.dataset_backup)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.SVHN:
            return dataset.labels[idx].item()
        elif str(dataset_type) == "<class 'src.dataset.dataset_criteo.CriteoDataset'>":
            return dataset.y[idx].item()
        elif self.in_memory:
            if not self.is_criteo:
                return dataset[idx][1].item()
            else:
                return dataset[idx][2].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset.train_labels[idx].item()

    def __len__(self):
        return self.balanced_max * len(self.keys)