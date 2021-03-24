import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

from src.dataset.sampler import BalancedBatchSampler


class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """

    def __init__(self, path, config, is_train, is_source, indexes):
        """
        Initialize file path and train/test mode.
        """
        data_source = pd.read_csv(os.path.join(path, "data/total_source_data.txt"))
        data_target = pd.read_csv(os.path.join(path, "data/total_target_data.txt"))

        # Prepare dataset for log normalization
        data_source.iloc[:, :13] = data_source.iloc[:, :13] + 1
        data_target.iloc[:, :13] = data_target.iloc[:, :13] + 1
        data_source.iloc[:, 1] = data_source.iloc[:, 1] + 2
        data_target.iloc[:, 1] = data_target.iloc[:, 1] + 2
        data_source.iloc[:, :13] = np.floor(np.log(data_source.iloc[:, :13]))
        data_target.iloc[:, :13] = np.floor(np.log(data_target.iloc[:, :13]))

        if not config.model.upper_bound:
            data_target.iloc[:, 0] = 0
            data_target.iloc[:, 4] = 0
            data_target.iloc[:, 5] = 0
            data_target.iloc[:, 6] = 0
            data_target.iloc[:, 10] = 0
            data_target.iloc[:, 11] = 0

        X_source = data_source.iloc[:, :-1].values
        y_source = data_source.iloc[:, -1].values
        X_target = data_target.iloc[:, :-1].values
        y_target = data_target.iloc[:, -1].values
        X_train_source, X_test_source, y_train_source, y_test_source = \
            train_test_split(X_source, y_source, test_size=0.2, random_state=12)
        X_train_target, X_test_target, y_train_target, y_test_target = \
            train_test_split(X_target, y_target, test_size=0.2, random_state=12)

        X1_train_source = X_train_source[:, :13]
        X1_test_source = X_test_source[:, :13]
        X1_train_target = X_train_target[:, :13]
        X1_test_target = X_test_target[:, :13]

        use_categorical = config.model.use_categorical

        if is_source:
            self.X1 = X1_train_source
            self.X2 = X_train_source[:, indexes] if not use_categorical else X_train_source[:, 13:]
            self.y = y_train_source
            self.X1_test = X1_test_source
            self.X2_test = X_test_source[:, indexes] if not use_categorical else X_test_source[:, 13:]
            self.y_test = y_test_source
        else:
            self.X1 = X1_train_target
            self.X2 = X_train_target[:, indexes] if not use_categorical else X_train_target[:, 13:]
            self.y = y_train_target
            self.X1_test = X1_test_target
            self.X2_test = X_test_target[:, indexes] if not use_categorical else X_test_target[:, 13:]
            self.y_test = y_test_target

        self.is_train = is_train

    def __getitem__(self, idx):
        dataI = self.X1[idx, :] if self.is_train else self.X1_test[idx, :]
        dataC = self.X2[idx, :] if self.is_train else self.X2_test[idx, :]
        target = self.y[idx] if self.is_train else self.y_test[idx]
        Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1).float()
        Xc = torch.from_numpy(dataC.astype(np.int32)).unsqueeze(-1).float()
        return Xi, Xc, target

    def __len__(self):
        if self.is_train:
            return len(self.X1)
        else:
            return len(self.X1_test)


def get_criteo(is_train, is_source, path, config, indexes=np.array([]), batch_size=32, in_memory=True,
               is_balanced=False, drop_last=True):
    criteo_dataset = CriteoDataset(path, config, is_train=is_train, is_source=is_source, indexes=indexes)

    if in_memory:
        criteo_loader = torch.utils.data.DataLoader(
            dataset=criteo_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False)

        dataI = torch.zeros((len(criteo_loader), 13, 1))
        dataC = torch.zeros((len(criteo_loader), len(indexes), 1))
        label = torch.zeros(len(criteo_loader))

        for i, (dataI_, dataC_, target) in enumerate(criteo_loader):
            dataI[i] = dataI_
            dataC[i] = dataC_
            label[i] = target

        full_data = torch.utils.data.TensorDataset(dataI, dataC, label.long())

        if is_balanced:
            criteo_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(full_data, in_memory=True, is_criteo=True),
                drop_last=drop_last)
        else:
            criteo_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)

    else:
        if is_balanced:
            criteo_loader = torch.utils.data.DataLoader(
                dataset=criteo_dataset,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(criteo_dataset, is_criteo=True),
                drop_last=drop_last)
        else:
            criteo_loader = torch.utils.data.DataLoader(
                dataset=criteo_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)

    return criteo_loader
