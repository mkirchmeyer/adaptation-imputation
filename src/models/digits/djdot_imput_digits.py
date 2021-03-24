import torch
import torch.nn.functional as F
from itertools import cycle
from time import clock as tick
import torch.optim as optim
from experiments.launcher.config import DatasetConfig
from src.eval.utils_eval import evaluate_data_imput_classifier, compute_mse_imput
from src.models.digits.djdot_digits import dist_torch
import ot
import numpy as np
from src.plotting.utils_plotting import plot_data_frontier_digits
from src.utils.network import weight_init_glorot_uniform
from src.utils.utils_network import set_lr, get_models_imput


class DJDOTImput(object):
    def __init__(self, data_loader_train_s, data_loader_train_t, model_config, cuda=False, logger_file=None,
                 data_loader_test_s=None, data_loader_test_t=None, dataset=DatasetConfig(),
                 data_loader_train_s_init=None, n_class=10):
        self.data_loader_train_s = data_loader_train_s
        self.data_loader_train_t = data_loader_train_t
        self.data_loader_test_t = data_loader_test_t
        self.data_loader_test_s = data_loader_test_s
        self.data_loader_train_s_init = data_loader_train_s_init
        self.n_class = 10
        self.cuda = cuda
        self.logger = logger_file
        self.crop_dim = int(dataset.im_size * model_config.crop_ratio)
        self.dataset = dataset
        self.activate_adaptation_imp = model_config.activate_adaptation_imp
        self.activate_mse = model_config.activate_mse
        self.activate_adaptation_d1 = model_config.activate_adaptation_d1
        self.lr_decay_epoch = model_config.epoch_to_start_align
        self.lr_decay_factor = 0.5
        self.epoch_to_start_align = model_config.epoch_to_start_align
        self.model_config = model_config
        self.output_fig = model_config.output_fig
        self.initialize_model = model_config.initialize_model
        self.stop_grad = model_config.stop_grad
        self.alpha = model_config.djdot_alpha

        feat_extractor1, feat_extractor2, data_classifier, domain_classifier1, domain_classifier2, reconstructor = \
            get_models_imput(model_config, n_class, dataset)
        feat_extractor1.apply(weight_init_glorot_uniform)
        feat_extractor2.apply(weight_init_glorot_uniform)
        data_classifier.apply(weight_init_glorot_uniform)
        domain_classifier1.apply(weight_init_glorot_uniform)
        domain_classifier2.apply(weight_init_glorot_uniform)
        reconstructor.apply(weight_init_glorot_uniform)
        self.feat_extractor1 = feat_extractor1
        self.feat_extractor2 = feat_extractor2
        self.data_classifier = data_classifier
        self.reconstructor = reconstructor
        if self.cuda:
            self.feat_extractor1.cuda()
            self.feat_extractor2.cuda()
            self.data_classifier.cuda()
            self.reconstructor.cuda()
        self.optimizer_g1 = optim.Adam(self.feat_extractor1.parameters(), lr=model_config.init_lr)
        self.optimizer_g2 = optim.Adam(self.feat_extractor2.parameters(), lr=model_config.init_lr)
        self.optimizer_h = optim.Adam(self.reconstructor.parameters(), lr=model_config.init_lr)
        self.optimizer_data_classifier = optim.Adam(self.data_classifier.parameters(), lr=model_config.init_lr)

        self.init_lr = model_config.init_lr
        self.adaptive_lr = model_config.adaptive_lr

    def fit(self):
        self.loss_history = []
        self.error_history = []

        self.mask_1 = torch.ones(size=(self.dataset.channel, self.dataset.im_size, self.dataset.im_size))
        self.mask_2 = torch.ones(size=(self.dataset.channel, self.dataset.im_size, self.dataset.im_size))
        if self.cuda:
            self.mask_1 = self.mask_1.cuda()
            self.mask_2 = self.mask_2.cuda()
        self.mask_1[:, :self.crop_dim, :] = 0.0
        self.mask_2[:, self.crop_dim:, :] = 0.0

        if self.initialize_model:
            self.logger.info("Initialize model")
            for epoch in range(self.epoch_to_start_align):
                self.feat_extractor1.train()
                self.feat_extractor2.train()
                self.data_classifier.train()
                tic = tick()

                for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s_init):
                    y_batch_s = y_batch_s.view(-1)

                    self.feat_extractor1.zero_grad()
                    self.feat_extractor2.zero_grad()
                    self.data_classifier.zero_grad()
                    if self.cuda:
                        X_batch_s = X_batch_s.cuda()
                        y_batch_s = y_batch_s.cuda()
                    X_batch_s1 = torch.mul(X_batch_s, self.mask_1)
                    X_batch_s2 = torch.mul(X_batch_s, self.mask_2)
                    size = X_batch_s.size()
                    output_feat_s1 = self.feat_extractor1(X_batch_s1)
                    output_feat_s2 = self.feat_extractor2(X_batch_s2)
                    output_class_s = self.data_classifier(torch.cat((output_feat_s1, output_feat_s2), 1))
                    loss = F.cross_entropy(output_class_s, y_batch_s)

                    loss.backward()
                    self.optimizer_g1.step()
                    self.optimizer_g2.step()
                    self.optimizer_data_classifier.step()

                toc = tick() - tic
                self.logger.info("\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss:{:.6f}".format(
                        epoch, self.nb_epochs, toc, loss.item(), 0))

                if epoch % 5 == 0 and epoch != 0:
                    evaluate_data_imput_classifier(self, is_test=True, is_target=False)
                    evaluate_data_imput_classifier(self, is_test=True, is_target=True)

                self.loss_history.append(loss.item())
                self.error_history.append(loss.item())

            start_epoch = self.epoch_to_start_align
            self.logger.info(f"Finished initializing with batch size: {size}")
        else:
            start_epoch = 0

        if self.output_fig and start_epoch != 0:
            plot_data_frontier_digits(self, self.data_loader_test_s, self.data_loader_test_t, "djdot_imput_10",
                                      is_imput=True)

        self.logger.info("Start aligning")
        for epoch in range(start_epoch, self.nb_epochs):
            self.feat_extractor1.train()
            self.feat_extractor2.train()
            self.data_classifier.train()
            self.reconstructor.train()

            tic = tick()
            self.T_batches = cycle(iter(self.data_loader_train_t))

            for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s):
                y_batch_s = y_batch_s.view(-1)

                p = (batch_idx + (epoch - start_epoch) * len(self.data_loader_train_s)) / (
                        len(self.data_loader_train_s) * (self.nb_epochs - start_epoch))

                if self.adaptive_lr:
                    lr = self.init_lr / (1. + 10 * p) ** 0.75
                    set_lr(self.optimizer_g1, lr)
                    set_lr(self.optimizer_g2, lr)
                    set_lr(self.optimizer_h, lr)
                    set_lr(self.optimizer_data_classifier, lr)

                self.feat_extractor1.zero_grad()
                self.feat_extractor2.zero_grad()
                self.data_classifier.zero_grad()
                self.reconstructor.zero_grad()

                X_batch_t, _ = next(self.T_batches)
                if self.cuda:
                    X_batch_t = X_batch_t.cuda()
                    X_batch_s = X_batch_s.cuda()
                    y_batch_s = y_batch_s.cuda()
                X_batch_s1 = torch.mul(X_batch_s, self.mask_1)
                X_batch_s2 = torch.mul(X_batch_s, self.mask_2)
                X_batch_t1 = torch.mul(X_batch_t, self.mask_1)

                output_feat_s1 = self.feat_extractor1(X_batch_s1)
                output_feat_s2 = self.feat_extractor2(X_batch_s2)

                self.grad_scale = 2. / (1. + np.exp(-10 * p)) - 1

                if self.stop_grad:
                    with torch.no_grad():
                        output_feat_s1_da = self.feat_extractor1(X_batch_s1)
                else:
                    output_feat_s1_da = output_feat_s1
                output_feat_s2_imputed_da = self.reconstructor(output_feat_s1_da)

                # -----------------------------------------------------------------
                # source classification
                # -----------------------------------------------------------------
                output_class_s = self.data_classifier(torch.cat((output_feat_s1, output_feat_s2), 1))
                loss = F.cross_entropy(output_class_s, y_batch_s)
                error = loss

                # -----------------------------------------------------------------
                # DJDOT domain classif
                # -----------------------------------------------------------------
                if self.activate_adaptation_d1:
                    output_feat_t1 = self.feat_extractor1(X_batch_t1)
                    reconstructed_t1 = self.reconstructor(output_feat_t1)
                    M1 = self.alpha * dist_torch(torch.cat((output_feat_s1, output_feat_s2_imputed_da), 1),
                                                 torch.cat((output_feat_t1, reconstructed_t1), 1))

                    gamma1 = torch.from_numpy(ot.emd(ot.unif(2 * output_feat_s1.size(0)),
                                                     ot.unif(2 * output_feat_t1.size(0)),
                                                     M1.cpu().detach().numpy())).float()
                    if self.cuda:
                        gamma1 = gamma1.cuda()

                    dist_loss1 = torch.sum(gamma1 * M1) * self.grad_scale
                    error += dist_loss1
                else:
                    dist_loss1 = torch.zeros(1)

                # -----------------------------------------------------------------
                # Imputation
                # -----------------------------------------------------------------
                # Adaptation Imput
                if self.activate_adaptation_imp:
                    M2 = self.alpha * dist_torch(output_feat_s2_imputed_da, output_feat_s2)
                    gamma2 = torch.from_numpy(ot.emd(ot.unif(output_feat_s2_imputed_da.size(0)),
                                                     ot.unif(output_feat_s2.size(0)),
                                                     M2.cpu().detach().numpy())).float()
                    if self.cuda:
                        gamma2 = gamma2.cuda()

                    dist_loss2 = torch.sum(gamma2 * M2) * self.grad_scale
                    error += dist_loss2
                else:
                    dist_loss2 = torch.zeros(1)

                # MSE Imput
                if self.activate_mse:
                    dist_loss_mse = torch.dist(output_feat_s2, output_feat_s2_imputed_da, 2)
                    error += dist_loss_mse
                else:
                    dist_loss_mse = torch.zeros(1)

                error.backward()

                self.optimizer_data_classifier.step()
                self.optimizer_h.step()
                self.optimizer_g1.step()
                self.optimizer_g2.step()

            toc = tick() - tic

            self.logger.info("\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss1:{:.6f} Dist_loss2:{:.6f} "
                "Dist_lossMSE:{:.6f}".format(epoch, self.nb_epochs, toc, loss.item(), dist_loss1.item(),
                                             dist_loss2.item(), dist_loss_mse.item()))
            self.loss_history.append(loss.item())
            self.error_history.append(error.item())

            if epoch % 5 == 0 and epoch != 0:
                # evaluate_data_imput_classifier(self, is_test=True, is_target=False)
                evaluate_data_imput_classifier(self, is_test=True, is_target=True)
                compute_mse_imput(self, is_target=True)
                compute_mse_imput(self, is_target=False)

        self.loss_test_s, self.acc_test_s, _, _ = evaluate_data_imput_classifier(self, is_test=True, is_target=False)
        self.loss_test_t, self.acc_test_t, _, _ = evaluate_data_imput_classifier(self, is_test=True, is_target=True)
        compute_mse_imput(self, is_target=True)
        compute_mse_imput(self, is_target=False)
        if self.output_fig:
            plot_data_frontier_digits(self, self.data_loader_test_s, self.data_loader_test_t, "djdot_imput_100",
                                      is_imput=True)
