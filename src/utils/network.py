import numpy as np
import torch
import torch.nn as nn


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def weight_init_glorot_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


##########
# DIGITS #
##########


class FeatureExtractorDigits(nn.Module):
    def __init__(self, dataset, kernel_size=5):
        super(FeatureExtractorDigits, self).__init__()
        self.conv1 = nn.Conv2d(dataset.channel, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64 * 2, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = self.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x


class DataClassifierDigits(nn.Module):
    def __init__(self, n_class, is_imput=False):
        super(DataClassifierDigits, self).__init__()
        factor = 2 if is_imput else 1
        input_size = 64 * 2 * factor

        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout2d()
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, n_class)

    def forward(self, input):
        x = self.dp1(self.relu1(self.bn1(self.fc1(input))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class DomainClassifierDigits(nn.Module):
    def __init__(self, is_d1=False, bigger_discrim=True):
        super(DomainClassifierDigits, self).__init__()
        self.domain_classifier = nn.Sequential()
        factor = 2 if is_d1 else 1
        input_size = 64 * 2 * factor
        output_size = 500 if bigger_discrim else 100

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, 100) if bigger_discrim else nn.Linear(output_size, 2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, input):
        x = self.relu1(self.bn1(self.fc1(input)))
        if self.bigger_discrim:
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x


class ReconstructorDigits(nn.Module):
    def __init__(self, bigger_reconstructor=False):
        super(ReconstructorDigits, self).__init__()
        self.domain_classifier = nn.Sequential()
        input_size = 64 * 2
        output_size = 512
        self.bigger_reconstructor = bigger_reconstructor

        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(output_size, input_size)
        self.bn3 = nn.BatchNorm1d(input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu1(self.bn1(self.fc1(input)))
        if self.bigger_reconstructor:
            x = self.relu2(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.bn3(self.fc3(x)))
        return x


##########
# Criteo #
##########


class FeatureExtractorCriteo(nn.Module):
    def __init__(self, feature_sizes=None, input_size=13, output_size=128):
        super(FeatureExtractorCriteo, self).__init__()
        self.nn1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.nn2 = nn.Linear(output_size, output_size)
        self.relu2 = nn.ReLU()
        self.nn3 = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()

        self.feature_sizes = feature_sizes
        if feature_sizes is not None:
            self.categorical_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, int(6 * np.power(feature_size, 1 / 4)))
                 for feature_size in self.feature_sizes])

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.relu1(self.nn1(x))
        x = self.relu2(self.nn2(x))
        x = self.sigmoid(self.nn3(x))
        x = x.view(x.size(0), -1)
        return x


class DataClassifierCriteo(nn.Module):
    def __init__(self, n_class, input_size, is_imput=False):
        super(DataClassifierCriteo, self).__init__()
        factor = 2 if is_imput else 1
        self.fc1 = nn.Linear(input_size * factor, input_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_size, n_class)

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        x = self.fc2(x)
        return x


class DomainClassifierCriteo(nn.Module):
    def __init__(self, input_size=128, is_d1=False, bigger_discrim=True):
        super(DomainClassifierCriteo, self).__init__()
        self.domain_classifier = nn.Sequential()
        factor = 2 if is_d1 else 1
        size = input_size * factor

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(size, input_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(input_size, 2)

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        if self.bigger_discrim:
            x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class ReconstructorCriteo(nn.Module):
    def __init__(self, input_size=128, bigger_reconstructor=False):
        super(ReconstructorCriteo, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.bigger_reconstructor = bigger_reconstructor

        output_size = input_size * 2
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(output_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        if self.bigger_reconstructor:
            x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
