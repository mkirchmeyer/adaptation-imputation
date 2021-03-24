import numpy as np
import torch
import logging
import os
import torch.optim as optim
import torch.nn.functional as F

from logging.handlers import RotatingFileHandler
from src.utils.network import FeatureExtractorDigits, DataClassifierDigits, DomainClassifierDigits, \
    ReconstructorDigits, FeatureExtractorCriteo, DataClassifierCriteo, DomainClassifierCriteo, \
    ReconstructorCriteo


def exp_lr_scheduler(optimizer, epoch, lr_decay_step=100, factor=0.5, name=None, decay_once=True, logger=None):
    """
    Decay current learning rate by a factor of 0.5 every lr_decay_epoch epochs.
    """
    init_lr = optimizer.param_groups[0]["lr"]
    if decay_once:
        if epoch > 0 and epoch == lr_decay_step:
            lr = init_lr * factor
            if logger:
                logger.info(f"Changing {name} LR to {lr} at step {lr_decay_step}")
            set_lr(optimizer, lr)
    else:
        if epoch > 0 and (epoch % lr_decay_step == 0):
            lr = init_lr * factor
            if logger:
                logger.info(f"Changing {name} LR to {lr} at step {lr_decay_step}")
            set_lr(optimizer, lr)
    return optimizer


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


def get_feature_extractor(model):
    return model.feat_extractor


def get_data_classifier(model):
    return model.data_classifier


def set_logger(model, logger):
    model.logger = logger


def set_nbepoch(model, nb_epoch):
    model.nb_epochs = nb_epoch


def build_label_domain(model, size, label):
    label_domain = torch.LongTensor(size)
    if model.cuda:
        label_domain = label_domain.cuda()

    label_domain.data.resize_(size).fill_(label)
    return label_domain


def create_log_name(name, config):
    name = f"{name}_{config.model.mode}_{config.model.source}_{config.model.target}_{config.run.run_id}_{config.run.metarun_id}"
    return name


def get_models(model_config, n_class, dataset):
    feat_extract = FeatureExtractorDigits(dataset)
    data_class = DataClassifierDigits(n_class)
    domain_class = DomainClassifierDigits(bigger_discrim=model_config.bigger_discrim)

    return feat_extract, data_class, domain_class


def get_models_imput(model_config, n_class, dataset):
    bigger_reconstructor = model_config.bigger_reconstructor
    bigger_discrim = model_config.bigger_discrim

    feat_extract_1 = FeatureExtractorDigits(dataset)
    feat_extract_2 = FeatureExtractorDigits(dataset)
    data_class = DataClassifierDigits(n_class, is_imput=True)
    domain_class_1 = DomainClassifierDigits(is_d1=True, bigger_discrim=bigger_discrim)
    domain_class_2 = DomainClassifierDigits(bigger_discrim=bigger_discrim)
    reconstructor = ReconstructorDigits(bigger_reconstructor=bigger_reconstructor)

    return feat_extract_1, feat_extract_2, data_class, domain_class_1, domain_class_2, reconstructor


def get_models_criteo(model_config, n_class, feature_sizes, n_missing=0):
    if model_config.use_categorical:
        total_size = np.sum([int(6 * np.power(feature_size, 1 / 4)) for feature_size in feature_sizes]) + 13
        input_size = total_size - n_missing
    else:
        input_size = 13 - n_missing

    output_size = 1024 if model_config.use_categorical else 128
    feat_extract = FeatureExtractorCriteo(feature_sizes=feature_sizes, input_size=input_size,
                                          output_size=output_size)
    data_class = DataClassifierCriteo(n_class=n_class, input_size=output_size)
    domain_class = DomainClassifierCriteo(input_size=output_size, bigger_discrim=model_config.bigger_discrim)

    return feat_extract, data_class, domain_class


def get_models_imput_criteo(model_config, n_class, feature_sizes, n_missing=3):
    bigger_reconstructor = model_config.bigger_reconstructor
    bigger_discrim = model_config.bigger_discrim
    if model_config.use_categorical:
        total_size = np.sum([int(6 * np.power(feature_size, 1 / 4)) for feature_size in feature_sizes]) + 13
        input_size = total_size - n_missing
    else:
        input_size = 13 - n_missing

    output_size = 1024 if model_config.use_categorical else 128
    feat_extract_1 = FeatureExtractorCriteo(feature_sizes=feature_sizes, input_size=input_size,
                                            output_size=output_size)
    feat_extract_2 = FeatureExtractorCriteo(input_size=n_missing, output_size=output_size)
    data_class = DataClassifierCriteo(n_class=n_class, is_imput=True, input_size=output_size)
    domain_class_1 = DomainClassifierCriteo(input_size=output_size, is_d1=True, bigger_discrim=bigger_discrim)
    domain_class_2 = DomainClassifierCriteo(input_size=output_size, bigger_discrim=bigger_discrim)
    reconstructor = ReconstructorCriteo(input_size=output_size, bigger_reconstructor=bigger_reconstructor)

    return feat_extract_1, feat_extract_2, data_class, domain_class_1, domain_class_2, reconstructor


def get_optimizer(model_config, model):
    optimizer_g = optim.Adam(model.feat_extractor.parameters(), lr=model_config.init_lr, betas=(0.8, 0.999))
    optimizer_f = optim.Adam(model.data_classifier.parameters(), lr=model_config.init_lr,
                             betas=(0.8, 0.999))
    if model_config.mode.find("dann") != -1:
        optimizer_d = optim.Adam(model.grl_domain_classifier.parameters(), lr=model_config.init_lr,
                                 betas=(0.8, 0.999))

    if model_config.mode.find("dann") != -1:
        return optimizer_f, optimizer_g, optimizer_d

    return optimizer_f, optimizer_g, optimizer_g


def get_optimizer_imput(model_config, model):
    init_lr = model_config.init_lr

    optimizer_g1 = optim.Adam(model.feat_extractor1.parameters(), lr=init_lr, betas=(0.8, 0.999))
    optimizer_g2 = optim.Adam(model.feat_extractor2.parameters(), lr=init_lr, betas=(0.8, 0.999))
    optimizer_h = optim.Adam(model.reconstructor.parameters(), lr=init_lr, betas=(0.8, 0.999))
    optimizer_data_classifier = optim.Adam(model.data_classifier.parameters(), lr=init_lr, betas=(0.8, 0.999))
    if model_config.mode.find("dann") != -1:
        optimizer_d1 = optim.Adam(model.grl_domain_classifier1.parameters(), lr=init_lr, betas=(0.8, 0.999))
        optimizer_d2 = optim.Adam(model.grl_domain_classifier2.parameters(), lr=init_lr, betas=(0.8, 0.999))

    if model_config.mode.find("dann") != -1:
        return optimizer_data_classifier, optimizer_g1, optimizer_g2, optimizer_h, optimizer_d1, optimizer_d2

    return optimizer_data_classifier, optimizer_g1, optimizer_g2, optimizer_h, optimizer_h, optimizer_h


def create_logger(outfile):
    try:
        os.mkdir("./results/")
        print(f"Directory ./results/ created")
    except FileExistsError:
        print(f"Directory ./results/ already exists replacing files in this notebook")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(outfile, "w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger


def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    """
    assert v.dim() == 2
    n, c = v.size()
    # return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))
    # mask = v.ge(0.000001)
    # mask_out = torch.masked_select(v, mask)
    # entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    # return entropy / float(v.size(0))
    b = F.softmax(v) * F.log_softmax(v)
    b = -1.0 * b.sum()
    return b / n
