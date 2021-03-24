import random
import numpy as np
import torch
import time
from experiments.launcher.config import Config, dummy_model_config
from src.dataset.utils_dataset import create_dataset
from src.models.digits.dann_imput_digits import DANNImput
from src.models.digits.dann_digits import DANN
from src.models.digits.djdot_imput_digits import DJDOTImput
from src.models.digits.djdot_digits import DeepJDOT
from src.utils.utils_network import create_logger, set_nbepoch, create_log_name

n_class = 10
debug = False

if debug:
    config = dummy_model_config
    in_memory = False
else:
    config = Config.get_config_from_args()
    in_memory = True

# python RNG
random.seed(config.model.random_seed)

# pytorch RNGs
torch.manual_seed(config.model.random_seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.model.random_seed)

# numpy RNG
np.random.seed(config.model.random_seed)

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(config.run.gpu_id)

name = create_log_name("digits", config)

logger = create_logger(f"./results/{name}.log")

logger.info(f"config: {config}")

logger.info("####################")
logger.info(f"{config.model.source} => {config.model.target}")

logger.info("===DATA===")
dataset, data_loader_train_s, data_loader_test_s, data_loader_train_t, data_loader_test_t, data_loader_train_s_init = \
    create_dataset(config, "../..", in_memory=in_memory, is_balanced=config.model.is_balanced)

n_dim = (len(data_loader_train_s.dataset), 10)
n_instances_train_s = len(data_loader_train_s.dataset)
n_instances_train_t = len(data_loader_train_t.dataset)

logger.info(f"n_instances_train_s: {n_instances_train_s}")
logger.info(f"n_instances_train_t: {n_instances_train_t}")

final_metrics = {
    "source_classif": dict(),
    "target_classif": dict()
}

start_time = time.time()

if config.model.mode == "dann":
    logger.info("===DANN===")
    logger.info(f"upper_bound: {config.model.upper_bound}")
    model = DANN(data_loader_train_s, data_loader_train_t, model_config=config.model, cuda=cuda,
                 data_loader_test_s=data_loader_test_s, data_loader_test_t=data_loader_test_t, dataset=dataset,
                 data_loader_train_s_init=data_loader_train_s_init, logger_file=logger, n_class=n_class)
    set_nbepoch(model, config.training.n_epochs)
    model.fit()

if config.model.mode == "djdot":
    logger.info("===DEEPJDOT===")
    logger.info(f"distance: {config.model.mode}")
    logger.info(f"upper_bound: {config.model.upper_bound}")
    model = DeepJDOT(data_loader_train_s, data_loader_train_t, model_config=config.model, cuda=cuda,
                     data_loader_test_t=data_loader_test_t, data_loader_test_s=data_loader_test_s, logger_file=logger,
                     data_loader_train_s_init=data_loader_train_s_init, dataset=dataset, n_class=n_class)
    set_nbepoch(model, config.training.n_epochs)
    model.fit()

if config.model.mode == "dann_imput":
    logger.info("===DANN IMPUT===")
    logger.info(f"upper_bound: {config.model.upper_bound}")
    model = DANNImput(data_loader_train_s, data_loader_train_t, model_config=config.model, cuda=cuda,
                      data_loader_train_s_init=data_loader_train_s_init, data_loader_test_s=data_loader_test_s,
                      data_loader_test_t=data_loader_test_t, dataset=dataset, n_class=n_class, logger_file=logger)
    set_nbepoch(model, config.training.n_epochs)
    model.fit()

if config.model.mode == "djdot_imput":
    logger.info("===Djdot IMPUT===")
    logger.info(f"upper_bound: {config.model.upper_bound}")
    model = DJDOTImput(data_loader_train_s, data_loader_train_t, model_config=config.model, cuda=cuda,
                       data_loader_test_s=data_loader_test_s, data_loader_test_t=data_loader_test_t,
                       dataset=dataset, data_loader_train_s_init=data_loader_train_s_init, n_class=n_class,
                       logger_file=logger)
    set_nbepoch(model, config.training.n_epochs)
    model.fit()

final_metrics["source_classif"] = {
    "test_loss": model.loss_test_s,
    "test_acc": model.acc_test_s
}
final_metrics["target_classif"] = {
    "test_loss": model.loss_test_t,
    "test_acc": model.acc_test_t
}

if config.model.mode == "dann":
    final_metrics["domain"] = {
        "test_loss": model.loss_d_test,
        "test_acc": model.acc_d_test
    }

elif config.model.mode.find("dann_imput") != -1:
    final_metrics["domain1"] = {
        "test_loss": model.loss_d1_test,
        "test_acc": model.acc_d1_test
    }
    final_metrics["domain2"] = {
        "test_loss": model.loss_d2_test,
        "test_acc": model.acc_d2_test
    }

final_metrics["elapsed_time"] = time.time() - start_time
final_metrics["status"] = "completed"

logger.info(final_metrics)
