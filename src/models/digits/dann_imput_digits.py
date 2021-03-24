import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from time import clock as tick
from experiments.launcher.config import DatasetConfig
import numpy as np
from src.eval.utils_eval import evaluate_data_imput_classifier, evaluate_domain_imput_classifier
from src.plotting.utils_plotting import plot_data_frontier_digits
from src.utils.network import weight_init_glorot_uniform
from src.utils.utils_network import build_label_domain, set_lr, get_models_imput, get_optimizer_imput, entropy_loss


class DANNImput(object):
    def __init__(self, data_loader_train_s, data_loader_train_t, model_config, cuda=False,
                 logger_file=None, data_loader_test_s=None, data_loader_test_t=None,
                 dataset=DatasetConfig(), data_loader_train_s_init=None, n_class=10):
        self.data_loader_train_s = data_loader_train_s
        self.data_loader_train_t = data_loader_train_t
        self.data_loader_test_t = data_loader_test_t
        self.data_loader_test_s = data_loader_test_s
        self.data_loader_train_s_init = data_loader_train_s_init
        self.refinement = model_config.refinement
        self.n_epochs_refinement = model_config.n_epochs_refinement
        self.lambda_regul = model_config.lambda_regul
        self.lambda_regul_s = model_config.lambda_regul_s
        self.threshold_value = model_config.threshold_value
        self.domain_label_s = self.domain_label_true2 = 1
        self.domain_label_t = self.domain_label_fake2 = 0
        self.cuda = cuda
        self.logger = logger_file
        self.crop_dim = int(dataset.im_size * model_config.crop_ratio)
        self.adaptive_lr = model_config.adaptive_lr
        self.dataset = dataset
        self.activate_adaptation_imp = model_config.activate_adaptation_imp
        self.activate_mse = model_config.activate_mse
        self.activate_adaptation_d1 = model_config.activate_adaptation_d1
        self.epoch_to_start_align = model_config.epoch_to_start_align
        self.lr_decay_epoch = model_config.epoch_to_start_align
        self.lr_decay_factor = 0.5
        self.output_fig = model_config.output_fig
        self.initialize_model = model_config.initialize_model
        self.model_config = model_config
        self.grad_scale1 = self.grad_scale2 = 1.0
        self.weight_d2 = model_config.weight_d2
        self.weight_mse = model_config.weight_mse

        feat_extractor1, feat_extractor2, data_classifier, domain_classifier1, domain_classifier2, reconstructor = \
            get_models_imput(model_config, n_class, dataset)
        feat_extractor1.apply(weight_init_glorot_uniform)
        feat_extractor2.apply(weight_init_glorot_uniform)
        data_classifier.apply(weight_init_glorot_uniform)
        domain_classifier1.apply(weight_init_glorot_uniform)
        domain_classifier2.apply(weight_init_glorot_uniform)
        reconstructor.apply(weight_init_glorot_uniform)

        _parent_class = self

        # adding gradient reversal layer transparently to the user
        class GradReverse1(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                return x.clone()

            @staticmethod
            def backward(self, grad_output):
                return grad_output.neg() * _parent_class.grad_scale1

        class GradReverse2(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                return x.clone()

            @staticmethod
            def backward(self, grad_output):
                return grad_output.neg() * _parent_class.grad_scale2

        class GRLDomainClassifier1(nn.Module):
            def __init__(self, domain_classifier):
                super(GRLDomainClassifier1, self).__init__()
                self.domain_classifier1 = domain_classifier

            def forward(self, input):
                x = GradReverse1.apply(input)
                x = self.domain_classifier1.forward(x)
                return x

        class GRLDomainClassifier2(nn.Module):
            def __init__(self, domain_classifier):
                super(GRLDomainClassifier2, self).__init__()
                self.domain_classifier2 = domain_classifier

            def forward(self, input):
                x = GradReverse2.apply(input)
                x = self.domain_classifier2.forward(x)
                return x

        self.feat_extractor1 = feat_extractor1
        self.feat_extractor2 = feat_extractor2
        self.data_classifier = data_classifier
        self.reconstructor = reconstructor
        self.grl_domain_classifier1 = GRLDomainClassifier1(domain_classifier1)
        self.grl_domain_classifier2 = GRLDomainClassifier2(domain_classifier2)
        if self.cuda:
            self.feat_extractor1.cuda()
            self.feat_extractor2.cuda()
            self.data_classifier.cuda()
            self.grl_domain_classifier1.cuda()
            self.grl_domain_classifier2.cuda()
            self.reconstructor.cuda()
        self.optimizer_data_classifier, self.optimizer_g1, self.optimizer_g2, self.optimizer_h, self.optimizer_d1, \
        self.optimizer_d2 = get_optimizer_imput(model_config, self)

        self.init_lr = model_config.init_lr

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
                self.logger.info(
                    "\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss:{:.6f}".format(
                        epoch, self.nb_epochs, toc, loss.item(), 0))

                if epoch % 5 == 0 and epoch != 0:
                    # evaluate_data_imput_classifier(self, is_test=False, is_target=False)
                    evaluate_data_imput_classifier(self, is_test=True, is_target=True)

                self.loss_history.append(loss.item())
                self.error_history.append(loss.item())

            start_epoch = self.epoch_to_start_align
            self.logger.info(f"Finished initializing with batch size: {size}")
        else:
            start_epoch = 0

        self.logger.info("Start aligning")
        for epoch in range(start_epoch, self.nb_epochs):
            self.feat_extractor1.train()
            self.feat_extractor2.train()
            self.data_classifier.train()
            self.grl_domain_classifier1.train()
            self.grl_domain_classifier2.train()
            self.reconstructor.train()

            tic = tick()

            self.T_batches = cycle(iter(self.data_loader_train_t))

            for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s):
                y_batch_s = y_batch_s.view(-1)
                p = (batch_idx + (epoch - start_epoch) * len(self.data_loader_train_s)) / (
                        len(self.data_loader_train_s) * (self.nb_epochs - start_epoch))

                if self.adaptive_lr:
                    lr = self.init_lr / (1. + 10 * p) ** 0.75
                    set_lr(self.optimizer_d1, lr)
                    set_lr(self.optimizer_d2, lr)
                    set_lr(self.optimizer_g1, lr)
                    set_lr(self.optimizer_g2, lr)
                    set_lr(self.optimizer_h, lr)
                    set_lr(self.optimizer_data_classifier, lr)

                self.feat_extractor1.zero_grad()
                self.feat_extractor2.zero_grad()
                self.data_classifier.zero_grad()
                self.grl_domain_classifier1.zero_grad()
                self.grl_domain_classifier2.zero_grad()
                self.reconstructor.zero_grad()

                X_batch_t, y_batch_t = next(self.T_batches)
                if self.cuda:
                    X_batch_t = X_batch_t.cuda()
                    X_batch_s = X_batch_s.cuda()
                    y_batch_s = y_batch_s.cuda()
                X_batch_s1 = torch.mul(X_batch_s, self.mask_1)
                X_batch_s2 = torch.mul(X_batch_s, self.mask_2)
                X_batch_t1 = torch.mul(X_batch_t, self.mask_1)
                size_t1 = X_batch_t1.size(0)
                size_s1 = X_batch_s1.size(0)
                size_s2 = X_batch_s2.size(0)

                output_feat_s1 = self.feat_extractor1(X_batch_s1)
                output_feat_s2 = self.feat_extractor2(X_batch_s2)
                output_feat_s2_imputed = self.reconstructor(output_feat_s1)

                output_feat_t1 = self.feat_extractor1(X_batch_t1)
                output_feat_t2 = self.reconstructor(output_feat_t1)

                # -----------------------------------------------------------------
                # source classification
                # -----------------------------------------------------------------
                output_class_s = self.data_classifier(torch.cat((output_feat_s1, output_feat_s2), 1))
                loss = F.cross_entropy(output_class_s, y_batch_s)

                # -----------------------------------------------------------------
                # domain classification
                # -----------------------------------------------------------------
                if self.activate_adaptation_d1:
                    self.grad_scale1 = 2. / (1. + np.exp(-10 * p)) - 1

                    input_domain_s1_1 = torch.cat((output_feat_s1, output_feat_s2_imputed), 1)
                    output_domain_s1_1 = self.grl_domain_classifier1(input_domain_s1_1)
                    label_domain_s1 = build_label_domain(self, size_s1, self.domain_label_s)
                    error_s1 = F.cross_entropy(output_domain_s1_1, label_domain_s1)

                    input_domain_t1_1 = torch.cat((output_feat_t1, output_feat_t2), 1)
                    output_domain_t1_1 = self.grl_domain_classifier1(input_domain_t1_1)
                    label_domain_t1 = build_label_domain(self, size_t1, self.domain_label_t)
                    error_t1 = F.cross_entropy(output_domain_t1_1, label_domain_t1)

                    dist_loss1 = (error_s1 + error_t1)
                else:
                    dist_loss1 = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)

                # -----------------------------------------------------------------
                # imputation
                # -----------------------------------------------------------------
                if self.activate_adaptation_imp:
                    self.grad_scale2 = 2. / (1. + np.exp(-10 * p)) - 1

                    output_domain_s2_imputed = self.grl_domain_classifier2(output_feat_s2_imputed)
                    label_domain_fake2 = build_label_domain(self, size_s2, self.domain_label_fake2)
                    error_s2_imputed = F.cross_entropy(output_domain_s2_imputed, label_domain_fake2)

                    output_domain_s2_true = self.grl_domain_classifier2(output_feat_s2)
                    label_domain_true2 = build_label_domain(self, size_s2, self.domain_label_true2)
                    error_s2 = F.cross_entropy(output_domain_s2_true, label_domain_true2)
                    dist_loss2 = self.weight_d2 * (error_s2 + error_s2_imputed)
                else:
                    dist_loss2 = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)

                # MSE
                if self.activate_mse:
                    dist_loss_mse = self.weight_mse * torch.dist(output_feat_s2, output_feat_s2_imputed, 2)
                else:
                    dist_loss_mse = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)

                error = loss + dist_loss1 + dist_loss2 + dist_loss_mse
                error.backward()

                self.optimizer_data_classifier.step()
                self.optimizer_d1.step()
                self.optimizer_d2.step()
                self.optimizer_h.step()
                self.optimizer_g1.step()
                self.optimizer_g2.step()

            toc = tick() - tic

            self.logger.info("\nTrain epoch: {}/{} {:2.2f}s \tTotalLoss: {:.6f} LossS: {:.6f} Dist_loss1:{:.6f} "
                "Dist_loss2:{:.6f} Dist_lossMSE:{:.6f}".format(epoch, self.nb_epochs, toc, error.item(), loss.item(),
                dist_loss1.item(), dist_loss2.item(), dist_loss_mse.item()))

            self.loss_history.append(loss.item())
            self.error_history.append(error.item())

            if epoch % 5 == 0 and epoch != 0:
                evaluate_data_imput_classifier(self, is_test=True, is_target=False)
                evaluate_data_imput_classifier(self, is_test=True, is_target=True)
                evaluate_domain_imput_classifier(self, self.data_loader_test_s, self.data_loader_test_t,
                                                 is_imputation=False, comments="Domain1 test")
                evaluate_domain_imput_classifier(self, self.data_loader_test_s, self.data_loader_test_t,
                                                 is_imputation=True, comments="Domain2 test")

        if self.refinement:
            self.logger.info("Refinement")
            n_epochs_refinement = self.n_epochs_refinement
            lambda_regul = self.lambda_regul
            lambda_regul_s = self.lambda_regul_s
            threshold_value = self.threshold_value

            set_lr(self.optimizer_data_classifier, self.init_lr / 10)
            set_lr(self.optimizer_g1, self.init_lr / 10)
            set_lr(self.optimizer_g2, self.init_lr / 10)
            set_lr(self.optimizer_h, self.init_lr / 10)

            for epoch in range(self.nb_epochs, self.nb_epochs + n_epochs_refinement):
                evaluate_data_imput_classifier(self, is_test=True, is_target=False)
                evaluate_data_imput_classifier(self, is_test=True, is_target=True)

                self.data_classifier.train()
                self.feat_extractor1.train()
                self.feat_extractor2.train()
                self.reconstructor.train()
                self.T_batches = cycle(iter(self.data_loader_train_t))

                for batch_idx, (X_batch_s, y_batch_s) in enumerate(self.data_loader_train_s):
                    self.data_classifier.zero_grad()
                    self.feat_extractor1.zero_grad()
                    self.feat_extractor2.zero_grad()
                    self.reconstructor.zero_grad()

                    y_batch_s = y_batch_s.view(-1)
                    X_batch_t, y_batch_t = next(self.T_batches)
                    if self.cuda:
                        X_batch_t = X_batch_t.cuda()
                        X_batch_s = X_batch_s.cuda()
                        y_batch_s = y_batch_s.cuda()
                    X_batch_s1 = torch.mul(X_batch_s, self.mask_1)
                    X_batch_s2 = torch.mul(X_batch_s, self.mask_2)
                    X_batch_t1 = torch.mul(X_batch_t, self.mask_1)

                    output_feat_s1 = self.feat_extractor1(X_batch_s1)
                    output_feat_s2 = self.feat_extractor2(X_batch_s2)
                    output_feat_t1 = self.feat_extractor1(X_batch_t1)
                    output_feat_t2 = self.reconstructor(output_feat_t1)

                    # Source Domain Data
                    output_class_s = self.data_classifier(torch.cat((output_feat_s1, output_feat_s2), 1))
                    loss = F.cross_entropy(output_class_s, y_batch_s)

                    # Target Domain Data
                    output_class_t = self.data_classifier(torch.cat((output_feat_t1, output_feat_t2), 1))
                    threshold_index = F.log_softmax(output_class_t).data.max(1)[0] > np.log(threshold_value)
                    loss_t_ent = entropy_loss(output_class_t[~threshold_index])
                    y_batch_pseudo_t = output_class_t.data.max(1)[1][threshold_index]
                    if torch.sum(threshold_index) > 0:
                        loss_t = F.cross_entropy(output_class_t[threshold_index], y_batch_pseudo_t)
                    else:
                        loss_t = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
                    n_pseudo_labelled = torch.sum(threshold_index).item()

                    error = lambda_regul_s * loss + loss_t + lambda_regul * loss_t_ent
                    error.backward()

                    self.optimizer_data_classifier.step()
                    self.optimizer_g1.step()
                    self.optimizer_g2.step()
                    self.optimizer_h.step()

                self.logger.info(
                    "\nTrain epoch: {}/{} \tTotalLoss: {:.6f} LossS: {:.6f} LossT: {:.6f} EntropyT: {:.6f}".format(
                        epoch, self.nb_epochs + n_epochs_refinement, error.item(), lambda_regul_s * loss.item(),
                        loss_t.item(), lambda_regul * loss_t_ent.item()))
                self.logger.info("N_Pseudo: {:.1f}".format(n_pseudo_labelled))

        self.loss_test_s, self.acc_test_s, _, _ = \
            evaluate_data_imput_classifier(self, is_test=True, is_target=False)
        self.loss_test_t, self.acc_test_t, _, _ = \
            evaluate_data_imput_classifier(self, is_test=True, is_target=True)
        self.loss_d1_test, self.acc_d1_test = evaluate_domain_imput_classifier(
            self, self.data_loader_test_s, self.data_loader_test_t, is_imputation=False, comments="Domain1 test")
        self.loss_d2_test, self.acc_d2_test = evaluate_domain_imput_classifier(
            self, self.data_loader_test_s, self.data_loader_test_t, is_imputation=True, comments="Domain2 test")
        if self.output_fig:
            plot_data_frontier_digits(self, self.data_loader_test_s, self.data_loader_test_t, "dann_imput_100", is_imput=True)
