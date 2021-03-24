import numpy as np
import torch
from torch import nn

from src.dataset.dataset_criteo import get_criteo
from experiments.launcher.config import DatasetConfig
from src.dataset.sampler import BalancedBatchSampler
from torchvision import datasets
from src.dataset.dataset_mnistm import get_mnistm
from src.dataset.dataset_usps import get_usps
import torch.utils.data as data_utils

from torchvision.transforms import transforms

transform_usps = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])

transform_mnist32 = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])

transform_usps32 = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(32),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])])

transform_svhn = transforms.Compose([transforms.Resize(32),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_mnist32rgb = transforms.Compose([transforms.Resize(32),
                                           transforms.Grayscale(3),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_mnistrgb = transforms.Compose([transforms.Grayscale(3),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_mnistm = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_mnist(train, transform, path, image_size=28, batch_size=32, in_memory=True, num_channel=1, is_balanced=False,
              drop_last=True, download=True):
    """Get MNIST dataset loader."""
    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=f"{path}/data/",
                                   train=train,
                                   transform=transform,
                                   download=download)

    if in_memory:
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False)

        data = torch.zeros((len(mnist_data_loader), num_channel, image_size, image_size))
        label = torch.zeros(len(mnist_data_loader))

        for i, (data_, target) in enumerate(mnist_data_loader):
            # print(i, data_.shape)
            data[i] = data_
            label[i] = target

        full_data = torch.utils.data.TensorDataset(data, label.long())

        if is_balanced:
            mnist_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(full_data, in_memory=True),
                drop_last=drop_last)
        else:
            mnist_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)
    else:
        if is_balanced:
            mnist_data_loader = torch.utils.data.DataLoader(
                dataset=mnist_dataset,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(mnist_dataset),
                drop_last=drop_last)
        else:
            mnist_data_loader = torch.utils.data.DataLoader(
                dataset=mnist_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)

    return mnist_data_loader


def get_svhn(train, transform, path, image_size=28, batch_size=32, in_memory=True, num_channel=1, is_balanced=False,
             drop_last=True, download=True):
    """Get SVHN dataset loader."""
    # dataset and data loader
    if train:
        split = "train"
    else:
        split = "test"

    svhn_dataset = datasets.SVHN(root=f"{path}/data/", split=split, transform=transform, download=download)
    # svhn_dataset = SVHN(root=f"{path}/data/", split=split, transform=transform, download=True)

    if in_memory:
        svhn_data_loader = torch.utils.data.DataLoader(
            dataset=svhn_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False)

        data = torch.zeros((len(svhn_data_loader), num_channel, image_size, image_size))
        label = torch.zeros(len(svhn_data_loader))

        for i, (data_, target) in enumerate(svhn_data_loader):
            # print(i, data_.shape)
            data[i] = data_
            label[i] = target

        full_data = torch.utils.data.TensorDataset(data, label.long())

        if is_balanced:
            svhn_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(full_data, in_memory=True),
                drop_last=drop_last)
        else:
            svhn_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)

    else:
        if is_balanced:
            svhn_data_loader = torch.utils.data.DataLoader(
                dataset=svhn_dataset,
                batch_size=batch_size,
                sampler=BalancedBatchSampler(svhn_dataset),
                drop_last=drop_last)
        else:
            svhn_data_loader = torch.utils.data.DataLoader(
                dataset=svhn_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last)

    return svhn_data_loader


def create_dataset(config, path, in_memory=True, is_balanced=False):
    init_batch_size = config.training.init_batch_size

    if config.model.source == "SVHN" and config.model.target == "MNIST":
        dataset = DatasetConfig(channel=3, im_size=32)
        data_loader_train_s = get_svhn(train=True, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, is_balanced=is_balanced)
        data_loader_train_s_init = get_svhn(train=True, transform=transform_svhn, path=path,
                                            image_size=dataset.im_size,
                                            batch_size=init_batch_size, num_channel=dataset.channel,
                                            in_memory=in_memory, is_balanced=is_balanced)
        data_loader_test_s = get_svhn(train=False, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                      batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                      in_memory=in_memory, drop_last=False)
        data_loader_train_t = get_mnist(train=True, transform=transform_mnist32rgb, path=path,
                                        image_size=dataset.im_size,
                                        batch_size=config.training.batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory)
        data_loader_test_t = get_mnist(train=False, transform=transform_mnist32rgb, path=path,
                                       image_size=dataset.im_size,
                                       batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, drop_last=False)

    elif config.model.source == "MNIST" and config.model.target == "SVHN":
        dataset = DatasetConfig(channel=3, im_size=32)
        data_loader_train_s = get_mnist(train=True, transform=transform_mnist32rgb, path=path,
                                        image_size=dataset.im_size,
                                        batch_size=config.training.batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory)
        data_loader_test_s = get_mnist(train=False, transform=transform_mnist32rgb, path=path,
                                       image_size=dataset.im_size,
                                       batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, drop_last=False)
        data_loader_train_s_init = get_mnist(train=True, transform=transform_mnist32rgb, path=path,
                                             image_size=dataset.im_size,
                                             batch_size=init_batch_size, num_channel=dataset.channel,
                                             in_memory=in_memory, is_balanced=is_balanced)
        data_loader_train_t = get_svhn(train=True, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, is_balanced=is_balanced)
        data_loader_test_t = get_svhn(train=False, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                      batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                      in_memory=in_memory, drop_last=False)

    elif config.model.source == "USPS" and config.model.target == "MNIST":
        dataset = DatasetConfig(channel=1, im_size=32)
        data_loader_train_s = get_usps(train=True, transform=transform_usps32, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, is_balanced=is_balanced)
        data_loader_train_s_init = get_usps(train=True, transform=transform_usps32, path=path,
                                            image_size=dataset.im_size,
                                            batch_size=init_batch_size, num_channel=dataset.channel,
                                            in_memory=in_memory, is_balanced=is_balanced)
        data_loader_test_s = get_usps(train=False, transform=transform_usps32, path=path, image_size=dataset.im_size,
                                      batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                      in_memory=in_memory, drop_last=False)
        data_loader_train_t = get_mnist(train=True, transform=transform_mnist32, path=path, image_size=dataset.im_size,
                                        batch_size=config.training.batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory)
        data_loader_test_t = get_mnist(train=False, transform=transform_mnist32, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, drop_last=False)

    elif config.model.source == "MNIST" and config.model.target == "USPS":
        dataset = DatasetConfig(channel=1, im_size=32)
        data_loader_train_s = get_mnist(train=True, transform=transform_mnist32, path=path, image_size=dataset.im_size,
                                        batch_size=config.training.batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory, is_balanced=is_balanced)
        data_loader_train_s_init = get_mnist(train=True, transform=transform_mnist32, path=path,
                                             image_size=dataset.im_size,
                                             batch_size=init_batch_size, num_channel=dataset.channel,
                                             in_memory=in_memory, is_balanced=is_balanced)
        data_loader_test_s = get_mnist(train=False, transform=transform_mnist32, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, drop_last=False)
        data_loader_train_t = get_usps(train=True, transform=transform_usps32, path=path, image_size=dataset.im_size,
                                       batch_size=config.training.batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory)
        data_loader_test_t = get_usps(train=False, transform=transform_usps32, path=path, image_size=dataset.im_size,
                                      batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                      in_memory=in_memory, drop_last=False)

    elif config.model.source == "MNIST" and config.model.target == "MNISTM":
        dataset = DatasetConfig(channel=3, im_size=32)
        data_loader_train_s = get_mnist(train=True, transform=transform_mnist32rgb, path=path,
                                        image_size=dataset.im_size,
                                        batch_size=config.training.batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory, is_balanced=is_balanced)
        data_loader_train_s_init = get_mnist(train=True, transform=transform_mnist32rgb, path=path,
                                             image_size=dataset.im_size,
                                             batch_size=init_batch_size, num_channel=dataset.channel,
                                             in_memory=in_memory, is_balanced=is_balanced)
        data_loader_test_s = get_mnist(train=False, transform=transform_mnist32rgb, path=path,
                                       image_size=dataset.im_size,
                                       batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                       in_memory=in_memory, drop_last=False)
        data_loader_train_t = get_mnistm(train=True, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                         batch_size=config.training.batch_size, num_channel=dataset.channel,
                                         in_memory=in_memory)
        data_loader_test_t = get_mnistm(train=False, transform=transform_svhn, path=path, image_size=dataset.im_size,
                                        batch_size=config.training.test_batch_size, num_channel=dataset.channel,
                                        in_memory=in_memory, drop_last=False)

    else:
        raise Exception("Source and Target do not exist")

    return dataset, data_loader_train_s, data_loader_test_s, data_loader_train_t, data_loader_test_t, \
           data_loader_train_s_init


def create_dataset_criteo(config, path, in_memory=True, is_balanced=False, indexes=np.array([])):
    data_loader_train_s = get_criteo(is_train=True, is_source=True, path=path, config=config,
                                     batch_size=config.training.batch_size, drop_last=True, is_balanced=is_balanced,
                                     in_memory=in_memory, indexes=indexes)
    if config.training.init_batch_size != config.training.batch_size:
        data_loader_train_s_init = get_criteo(is_train=True, is_source=True, path=path, config=config,
                                              batch_size=config.training.init_batch_size, drop_last=True,
                                              is_balanced=is_balanced, in_memory=in_memory, indexes=indexes)
    else:
        data_loader_train_s_init = data_loader_train_s
    data_loader_test_s = get_criteo(is_train=False, is_source=True, path=path, config=config,
                                    batch_size=config.training.test_batch_size, drop_last=False,
                                    in_memory=in_memory, indexes=indexes)
    data_loader_train_t = get_criteo(is_train=True, is_source=False, path=path, config=config,
                                     batch_size=config.training.batch_size, drop_last=True,
                                     in_memory=in_memory, indexes=indexes)
    data_loader_test_t = get_criteo(is_train=False, is_source=False, path=path, config=config,
                                    batch_size=config.training.batch_size, drop_last=False,
                                    in_memory=in_memory, indexes=indexes)

    return data_loader_train_s, data_loader_test_s, data_loader_train_t, data_loader_test_t, data_loader_train_s_init
