from experiments.launcher.experiments_criteo import DannImputCriteo, SourceIgnoreCriteo, \
    DannIgnoreCriteo, SourceZeroImputCriteo, DannZeroImputCriteo
from experiments.launcher.experiments_mnist_mnistm import DannMNISTMNISTM, DannZeroImputMNISTMNISTM, \
    DannImputMNISTMNISTM, DjdotMNISTMNISTM, DjdotZeroImputMNISTMNISTM, DjdotImputMNISTMNISTM, DannIgnoreMNISTMNISTM, \
    DjdotIgnoreMNISTMNISTM
from experiments.launcher.experiments_svhn_mnist import DannSVHNMNIST, DannZeroImputSVHNMNIST, DannImputSVHNMNIST, \
    DjdotImputSVHNMNIST, DjdotSVHNMNIST, DjdotZeroImputSVHNMNIST, DjdotIgnoreSVHNMNIST, DannIgnoreSVHNMNIST
from experiments.launcher.experiments_mnist_usps import DannMNISTUSPS, DannZeroImputMNISTUSPS, DannImputMNISTUSPS, \
    DjdotImputMNISTUSPS, DjdotZeroImputMNISTUSPS, DjdotMNISTUSPS, DannIgnoreMNISTUSPS, DjdotIgnoreMNISTUSPS
from experiments.launcher.experiments_usps_mnist import DannUSPSMNIST, DannZeroImputUSPSMNIST, DannImputUSPSMNIST, \
    DjdotUSPSMNIST, DjdotZeroImputUSPSMNIST, DjdotImputUSPSMNIST, DjdotIgnoreUSPSMNIST, DannIgnoreUSPSMNIST

all_experiments = {
    "dann_mnist_usps": DannMNISTUSPS(),
    "dann_ignore_mnist_usps": DannIgnoreMNISTUSPS(),
    "dann_zeroimput_mnist_usps": DannZeroImputMNISTUSPS(),
    "dann_imput_mnist_usps": DannImputMNISTUSPS(),
    "djdot_mnist_usps": DjdotMNISTUSPS(),
    "djdot_ignore_mnist_usps": DjdotIgnoreMNISTUSPS(),
    "djdot_zeroimput_mnist_usps": DjdotZeroImputMNISTUSPS(),
    "djdot_imput_mnist_usps": DjdotImputMNISTUSPS(),

    "dann_usps_mnist": DannUSPSMNIST(),
    "dann_ignore_usps_mnist": DannIgnoreUSPSMNIST(),
    "dann_zeroimput_usps_mnist": DannZeroImputUSPSMNIST(),
    "dann_imput_usps_mnist": DannImputUSPSMNIST(),
    "djdot_usps_mnist": DjdotUSPSMNIST(),
    "djdot_ignore_usps_mnist": DjdotIgnoreUSPSMNIST(),
    "djdot_zeroimput_usps_mnist": DjdotZeroImputUSPSMNIST(),
    "djdot_imput_usps_mnist": DjdotImputUSPSMNIST(),

    "dann_svhn_mnist": DannSVHNMNIST(),
    "dann_ignore_svhn_mnist": DannIgnoreSVHNMNIST(),
    "dann_zeroimput_svhn_mnist": DannZeroImputSVHNMNIST(),
    "dann_imput_svhn_mnist": DannImputSVHNMNIST(),
    "djdot_svhn_mnist": DjdotSVHNMNIST(),
    "djdot_ignore_svhn_mnist": DjdotIgnoreSVHNMNIST(),
    "djdot_zeroimput_svhn_mnist": DjdotZeroImputSVHNMNIST(),
    "djdot_imput_svhn_mnist": DjdotImputSVHNMNIST(),

    "dann_mnist_mnistm": DannMNISTMNISTM(),
    "dann_ignore_mnist_mnistm": DannIgnoreMNISTMNISTM(),
    "dann_zeroimput_mnist_mnistm": DannZeroImputMNISTMNISTM(),
    "dann_imput_mnist_mnistm": DannImputMNISTMNISTM(),
    "djdot_mnist_mnistm": DjdotMNISTMNISTM(),
    "djdot_ignore_mnist_mnistm": DjdotIgnoreMNISTMNISTM(),
    "djdot_zeroimput_mnist_mnistm": DjdotZeroImputMNISTMNISTM(),
    "djdot_imput_mnist_mnistm": DjdotImputMNISTMNISTM(),

    "source_zeroimput_criteo": SourceZeroImputCriteo(),
    "source_ignore_criteo": SourceIgnoreCriteo(),
    "dann_zeroimput_criteo": DannZeroImputCriteo(),
    "dann_ignore_criteo": DannIgnoreCriteo(),
    "dann_imput_criteo": DannImputCriteo()
}
