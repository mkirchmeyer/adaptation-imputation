from time import clock as tick
import torch
from experiments.launcher.config import DatasetConfig
from src.eval.utils_eval import evaluate_data_classifier
from src.plotting.utils_plotting import plot_data_frontier_digits
from src.utils.network import weight_init_glorot_uniform
from src.utils.utils_network import set_lr, get_optimizer, get_models
import torch.nn.functional as F
import ot
from itertools import cycle

dtype = 'torch.FloatTensor'


class DeepJDOT(object):
    def __init__(self, data_loader_train_s, data_loader_train_t, model_config,
                 cuda=False, logger_file=None, data_loader_test_s=None, data_loader_test_t=None,
                 dataset=DatasetConfig(), n_class=10, data_loader_train_s_init=None):
        self.data_loader_train_s = data_loader_train_s
        self.data_loader_train_t = data_loader_train_t
        self.data_loader_test_s = data_loader_test_s
        self.data_loader_test_t = data_loader_test_t
        self.data_loader_train_s_init = data_loader_train_s_init
        self.cuda = cuda
        self.alpha = model_config.djdot_alpha
        self.epoch_to_start_align = model_config.epoch_to_start_align  # start aligning distrib from this step
        self.lr_decay_epoch = model_config.epoch_to_start_align
        self.lr_decay_factor = 0.5
        self.adapt_only_first = model_config.adapt_only_first
        self.crop_dim = 0 if model_config.upper_bound and not self.adapt_only_first else \
            int(dataset.im_size * model_config.crop_ratio)
        self.dataset = dataset
        self.output_fig = model_config.output_fig
        self.n_class = n_class
        self.initialize_model = model_config.initialize_model
        self.model_config = model_config

        feat_extractor, data_classifier, _ = get_models(model_config, n_class, dataset)
        feat_extractor.apply(weight_init_glorot_uniform)
        data_classifier.apply(weight_init_glorot_uniform)
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
        self.optimizer_feat_extractor, self.optimizer_data_classifier, _ = get_optimizer(model_config, self)
        self.init_lr = model_config.init_lr
        self.adaptive_lr = model_config.adaptive_lr
        self.logger = logger_file

    def fit(self):
        self.loss_history = []
        self.error_history = []

        if self.crop_dim != 0:
            self.mask_t = torch.ones(size=(self.dataset.channel, self.dataset.im_size, self.dataset.im_size))
            if self.cuda:
                self.mask_t = self.mask_t.cuda()
            self.mask_t[:, :self.crop_dim, :] = 0.0

        if self.initialize_model:
            self.logger.info("Initialize DJDOT")
            for epoch in range(self.epoch_to_start_align):
                self.feat_extractor.train()
                self.data_classifier.train()
                tic = tick()

                for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s_init):
                    y_batch_s = y_batch_s.view(-1)
                    self.feat_extractor.zero_grad()
                    self.data_classifier.zero_grad()
                    if self.cuda:
                        X_batch_s = X_batch_s.cuda()
                        y_batch_s = y_batch_s.cuda()
                    size = X_batch_s.size()
                    if self.adapt_only_first:
                        X_batch_s = torch.mul(X_batch_s, self.mask_t)
                    output_feat_s = self.feat_extractor(X_batch_s)
                    output_class_s = self.data_classifier(output_feat_s)
                    loss = F.cross_entropy(output_class_s, y_batch_s)

                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

                toc = tick() - tic
                self.logger.info("\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss:{:.6f}".format(
                    epoch, self.nb_epochs, toc, loss.item(), 0))

                if epoch % 5 == 0 and epoch != 0:
                    evaluate_data_classifier(self, is_test=True, is_target=False)
                    evaluate_data_classifier(self, is_test=True, is_target=True)

                self.loss_history.append(loss.item())
                self.error_history.append(loss.item())

            start_epoch = self.epoch_to_start_align
            self.logger.info(f"Finished initializing with batch size: {size}")
        else:
            start_epoch = 0

        if self.output_fig:
            if start_epoch != 0:
                plot_data_frontier_digits(self, self.data_loader_test_s, self.data_loader_test_t, "djdot_10")

        self.logger.info("Start aligning")
        for epoch in range(start_epoch, self.nb_epochs):
            self.feat_extractor.train()
            self.data_classifier.train()
            tic = tick()

            self.T_batches = cycle(iter(self.data_loader_train_t))
            for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s):
                y_batch_s = y_batch_s.view(-1)

                self.feat_extractor.zero_grad()
                self.data_classifier.zero_grad()
                p = (batch_idx + (epoch - start_epoch) * len(self.data_loader_train_s)) / (
                        len(self.data_loader_train_s) * (self.nb_epochs - start_epoch))

                if self.adaptive_lr:
                    lr = self.init_lr / (1. + 10 * p) ** 0.75
                    set_lr(self.optimizer_feat_extractor, lr)
                    set_lr(self.optimizer_data_classifier, lr)

                X_batch_t, _ = next(self.T_batches)
                if self.cuda:
                    X_batch_t = X_batch_t.cuda()
                    X_batch_s = X_batch_s.cuda()
                    y_batch_s = y_batch_s.cuda()

                if self.crop_dim != 0:
                    X_batch_t = torch.mul(X_batch_t, self.mask_t)
                if self.adapt_only_first:
                    X_batch_s = torch.mul(X_batch_s, self.mask_t)

                # Source Domain Data : forward feature extraction + data classifier
                output_feat_s = self.feat_extractor(X_batch_s)
                output_class_s = self.data_classifier(output_feat_s)
                loss = F.cross_entropy(output_class_s, y_batch_s)

                # compute distribution distance
                if epoch >= self.epoch_to_start_align:
                    g_batch_s = self.feat_extractor(X_batch_s)
                    g_batch_t = self.feat_extractor(X_batch_t)
                    M = self.alpha * dist_torch(g_batch_s, g_batch_t)
                    gamma = torch.from_numpy(ot.emd(ot.unif(g_batch_s.size(0)),
                                                    ot.unif(g_batch_t.size(0)),
                                                    M.cpu().detach().numpy())).float()
                    if self.cuda:
                        gamma = gamma.cuda()

                    dist_loss = torch.sum(gamma * M)

                    error = loss + dist_loss
                else:
                    error = loss
                    dist_loss = torch.zeros(1)

                error.backward()
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier.step()

            toc = tick() - tic
            self.logger.info("\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss:{:.6f}".format(
                epoch, self.nb_epochs, toc, loss.item(), dist_loss.item()))

            if epoch % 5 == 0 and epoch != 0:
                evaluate_data_classifier(self, is_test=True, is_target=False)
                evaluate_data_classifier(self, is_test=True, is_target=True)

            self.loss_history.append(loss.item())
            self.error_history.append(error.item())

        self.loss_test_s, self.acc_test_s, _, _ = evaluate_data_classifier(self, is_test=True, is_target=False)
        self.loss_test_t, self.acc_test_t, _, _ = evaluate_data_classifier(self, is_test=True, is_target=True)
        if self.output_fig:
            plot_data_frontier_digits(self, self.data_loader_test_s, self.data_loader_test_t, "djdot_100")


def dist_torch(x1, x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1, x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) - 2 * prod_x1x2
    return distance
