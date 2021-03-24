import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from time import clock as tick
import numpy as np
from src.eval.utils_eval import evaluate_data_classifier, evaluate_domain_classifier
from src.utils.network import weight_init_glorot_uniform
from src.utils.utils_network import set_lr, build_label_domain, get_optimizer, get_models_criteo, entropy_loss


class DANN(object):
    def __init__(self, data_loader_train_s, data_loader_train_t, model_config, cuda=False, logger_file=None,
                 data_loader_test_s=None, data_loader_test_t=None, data_loader_train_s_init=None, feature_sizes=None,
                 n_class=2):
        self.cuda = cuda
        self.data_loader_train_s = data_loader_train_s
        self.data_loader_train_t = data_loader_train_t
        self.data_loader_test_t = data_loader_test_t
        self.data_loader_test_s = data_loader_test_s
        self.data_loader_train_s_init = data_loader_train_s_init
        self.domain_label_s = 1
        self.domain_label_t = 0
        self.refinement = model_config.refinement
        self.n_epochs_refinement = model_config.n_epochs_refinement
        self.lambda_regul = model_config.lambda_regul
        self.lambda_regul_s = model_config.lambda_regul_s
        self.threshold_value = model_config.threshold_value
        self.logger = logger_file
        self.crop_dim = 0
        self.epoch_to_start_align = model_config.epoch_to_start_align
        self.output_fig = model_config.output_fig
        self.stop_grad = model_config.stop_grad
        self.adaptive_lr = model_config.adaptive_lr
        self.use_categorical = model_config.use_categorical
        self.lr_decay_epoch = model_config.epoch_to_start_align
        self.lr_decay_factor = 0.5
        self.model_config = model_config
        self.initialize_model = model_config.initialize_model
        self.init_lr = model_config.init_lr
        self.best_score_auc = 0
        self.n_epochs_no_change_auc = 0
        self.best_score_loss = 1000
        self.n_epochs_no_change_loss = 0
        self.grad_scale = 1.0
        self.adapt_only_first = model_config.adapt_only_first
        self.non_missing_features = [1, 2, 3, 7, 8, 9, 12]
        self.n_missing = 0
        if model_config.adapt_only_first:
            self.n_missing = 13 - len(self.non_missing_features)

        feat_extractor, data_classifier, domain_classifier = get_models_criteo(model_config, n_class, feature_sizes,
                                                                               self.n_missing)
        feat_extractor.apply(weight_init_glorot_uniform)
        data_classifier.apply(weight_init_glorot_uniform)
        domain_classifier.apply(weight_init_glorot_uniform)

        _parent_class = self

        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                return x.clone()

            @staticmethod
            def backward(self, grad_output):
                return grad_output.neg() * _parent_class.grad_scale

        class GRLDomainClassifier(nn.Module):
            def __init__(self, domain_classifier, stop_grad):
                super(GRLDomainClassifier, self).__init__()
                self.domain_classifier = domain_classifier
                self.stop_grad = stop_grad

            def forward(self, input):
                if self.stop_grad:
                    x = GradReverse.apply(input.detach())
                else:
                    x = GradReverse.apply(input)
                x = self.domain_classifier.forward(x)
                return x

        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.grl_domain_classifier = GRLDomainClassifier(domain_classifier, self.stop_grad)

        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.grl_domain_classifier.cuda()
        self.optimizer_feat_extractor, self.optimizer_data_classifier, self.optimizer_domain_classifier = \
            get_optimizer(model_config, self)

    def construct_input(self, X_batch_s_I, X_batch_s_C):
        if self.adapt_only_first:
            for idx, id in enumerate(self.non_missing_features):
                if idx == 0:
                    X_batch_s_I_filtered = X_batch_s_I[:, id, :]
                else:
                    X_batch_s_I_filtered = torch.cat((X_batch_s_I_filtered, X_batch_s_I[:, id, :]), 1)
        else:
            X_batch_s_I_filtered = X_batch_s_I[:, :, 0]

        if self.use_categorical:
            X_batch_s_C = X_batch_s_C.to(torch.int64)
            for i, emb in enumerate(self.feat_extractor.categorical_embeddings):
                if i == 0:
                    X_batch_s_C_embedded = emb(X_batch_s_C[:, 0, :])
                else:
                    X_batch_s_C_embedded = torch.cat((X_batch_s_C_embedded, emb(X_batch_s_C[:, i, :])), 2)

            if self.n_missing == 13:
                return X_batch_s_C_embedded[:, 0, :]
            return torch.cat((X_batch_s_I_filtered, X_batch_s_C_embedded[:, 0, :]), 1)

        return X_batch_s_I_filtered

    def fit(self):
        self.loss_history = []
        self.error_history = []

        if self.initialize_model:
            self.logger.info("Initialize DANN")
            for epoch in range(self.epoch_to_start_align):
                self.feat_extractor.train()
                self.data_classifier.train()
                tic = tick()

                for batch_idx, (X_batch_s_I, X_batch_s_C, y_batch_s) in enumerate(self.data_loader_train_s_init):
                    y_batch_s = y_batch_s.view(-1)
                    self.feat_extractor.zero_grad()
                    self.data_classifier.zero_grad()

                    if self.cuda:
                        X_batch_s_I, X_batch_s_C, y_batch_s = X_batch_s_I.cuda(), X_batch_s_C.cuda(), y_batch_s.cuda()

                    X_batch_s = self.construct_input(X_batch_s_I, X_batch_s_C)
                    output_feat_s = self.feat_extractor(X_batch_s)
                    output_class_s = self.data_classifier(output_feat_s)
                    loss = F.cross_entropy(output_class_s, y_batch_s)

                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

                toc = tick() - tic
                self.logger.info(
                    "\nTrain epoch: {}/{} {:2.2f}s \tLoss: {:.6f} Dist_loss:{:.6f}".format(
                        epoch, self.nb_epochs, toc, loss.item(), 0))

                if epoch % 5 == 0:
                    evaluate_data_classifier(self, is_test=True, is_target=False, is_criteo=True)
                    loss_t, acc_t, w_acc_t, auc_t = evaluate_data_classifier(self, is_test=True, is_target=True,
                                                                             is_criteo=True)
                    evaluate_domain_classifier(self, self.data_loader_test_s, self.data_loader_test_t,
                                               comments="Domain test", is_criteo=True)
                    if auc_t > self.best_score_auc:
                        self.best_score_auc = auc_t
                        self.logger.info(f"Best AUC score: Loss {loss_t}, AUC {auc_t}, WAcc {w_acc_t}")
                        self.n_epochs_no_change_auc = 0
                    else:
                        self.n_epochs_no_change_auc += 1

                    if loss_t < self.best_score_loss:
                        self.best_score_loss = loss_t
                        self.logger.info(f"Best loss score: Loss {loss_t}, AUC {auc_t}, WAcc {w_acc_t}")
                        self.n_epochs_no_change_loss = 0
                    else:
                        self.n_epochs_no_change_loss += 1

                self.logger.info(f"n_epochs_no_change_loss: {self.n_epochs_no_change_loss} / "
                                 f"n_epochs_no_change_auc: {self.n_epochs_no_change_auc}")

                self.loss_history.append(loss.item())
                self.error_history.append(loss.item())

            start_epoch = self.epoch_to_start_align
            self.logger.info(f"Finished initializing with init batch size")
        else:
            start_epoch = 0

        self.logger.info("Start aligning")
        for epoch in range(start_epoch, self.nb_epochs):
            self.feat_extractor.train()
            self.data_classifier.train()
            self.grl_domain_classifier.train()
            tic = tick()

            self.T_batches = cycle(iter(self.data_loader_train_t))

            for batch_idx, (X_batch_s_I, X_batch_s_C, y_batch_s) in enumerate(self.data_loader_train_s):
                size_s = X_batch_s_I.size(0)
                y_batch_s = y_batch_s.view(-1)

                p = (batch_idx + (epoch - start_epoch) * len(self.data_loader_train_s)) / (
                        len(self.data_loader_train_s) * (self.nb_epochs - start_epoch))

                if self.adaptive_lr:
                    lr = self.init_lr / (1. + 10 * p) ** 0.75
                    set_lr(self.optimizer_feat_extractor, lr)
                    set_lr(self.optimizer_data_classifier, lr)
                    set_lr(self.optimizer_domain_classifier, lr)
                self.feat_extractor.zero_grad()
                self.data_classifier.zero_grad()
                self.grl_domain_classifier.zero_grad()

                X_batch_t_I, X_batch_t_C, _ = next(self.T_batches)

                if self.cuda:
                    X_batch_t_I, X_batch_t_C, X_batch_s_I, X_batch_s_C, y_batch_s = \
                        X_batch_t_I.cuda(), X_batch_t_C.cuda(), X_batch_s_I.cuda(), X_batch_s_C.cuda(), y_batch_s.cuda()

                X_batch_t = self.construct_input(X_batch_t_I, X_batch_t_C)
                size_t = X_batch_t.size(0)
                X_batch_s = self.construct_input(X_batch_s_I, X_batch_s_C)

                output_feat_s = self.feat_extractor(X_batch_s)
                output_class_s = self.data_classifier(output_feat_s)
                loss = F.cross_entropy(output_class_s, y_batch_s)

                # -----------------------------------------------------------------
                # domain classification
                # -----------------------------------------------------------------
                self.grad_scale = (2. / (1. + np.exp(-10 * p)) - 1)
                output_domain_s = self.grl_domain_classifier(output_feat_s)
                label_domain_s = build_label_domain(self, size_s, self.domain_label_s)
                error_s = F.cross_entropy(output_domain_s, label_domain_s)

                output_feat_t = self.feat_extractor(X_batch_t)
                output_domain_t = self.grl_domain_classifier(output_feat_t)
                label_domain_t = build_label_domain(self, size_t, self.domain_label_t)
                error_t = F.cross_entropy(output_domain_t, label_domain_t)
                dist_loss = (error_s + error_t)

                error = loss + dist_loss
                error.backward()
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier.step()
                self.optimizer_domain_classifier.step()

            toc = tick() - tic

            self.logger.info(
                "\nTrain epoch: {}/{} {:.1f}% {:2.2f}s \tTotalLoss: {:.6f} LossS: {:.6f} Dist_loss:{:.6f}".format(
                    epoch, self.nb_epochs, p * 100, toc, error.item(), loss.item(), dist_loss.item()))

            self.loss_history.append(loss.item())
            self.error_history.append(error.item())

            if epoch % 5 == 0:
                evaluate_data_classifier(self, is_test=True, is_target=False, is_criteo=True)
                loss_t, acc_t, w_acc_t, auc_t = evaluate_data_classifier(self, is_test=True, is_target=True, is_criteo=True)
                evaluate_domain_classifier(self, self.data_loader_test_s, self.data_loader_test_t,
                                           comments="Domain test", is_criteo=True)
                if auc_t > self.best_score_auc:
                    self.best_score_auc = auc_t
                    self.logger.info(f"Best AUC score: Loss {loss_t}, AUC {auc_t}, WAcc {w_acc_t}")
                    self.n_epochs_no_change_auc = 0
                else:
                    self.n_epochs_no_change_auc += 1

                if loss_t < self.best_score_loss:
                    self.best_score_loss = loss_t
                    self.logger.info(f"Best loss score: Loss {loss_t}, AUC {auc_t}, WAcc {w_acc_t}")
                    self.n_epochs_no_change_loss = 0
                else:
                    self.n_epochs_no_change_loss += 1

            self.logger.info(f"n_epochs_no_change_loss: {self.n_epochs_no_change_loss} / "
                             f"n_epochs_no_change_auc: {self.n_epochs_no_change_auc}")

        self.logger.info(f"Best Loss {self.best_score_loss}, best AUC {self.best_score_auc}")

        if self.refinement:
            self.logger.info("Refinement")
            n_epochs_refinement = self.n_epochs_refinement
            lambda_regul = self.lambda_regul
            lambda_regul_s = self.lambda_regul_s
            threshold_value = self.threshold_value

            for epoch in range(self.nb_epochs, self.nb_epochs + n_epochs_refinement):
                self.data_classifier.train()
                self.T_batches = cycle(iter(self.data_loader_train_t))

                evaluate_data_classifier(self, is_test=True, is_target=False, is_criteo=True)
                evaluate_data_classifier(self, is_test=True, is_target=True, is_criteo=True)

                for batch_idx, (X_batch_s_I, X_batch_s_C, y_batch_s) in enumerate(self.data_loader_train_s):
                    y_batch_s = y_batch_s.view(-1)
                    self.data_classifier.zero_grad()
                    X_batch_t_I, X_batch_t_C, y_batch_t = next(self.T_batches)

                    if self.cuda:
                        X_batch_t_I, X_batch_t_C, X_batch_s_I, X_batch_s_C, y_batch_s, y_batch_t = \
                            X_batch_t_I.cuda(), X_batch_t_C.cuda(), X_batch_s_I.cuda(), X_batch_s_C.cuda(), \
                            y_batch_s.cuda(), y_batch_t.cuda()

                    X_batch_t = self.construct_input(X_batch_t_I, X_batch_t_C)
                    X_batch_s = self.construct_input(X_batch_s_I, X_batch_s_C)

                    # Source Domain Data : forward feature extraction + data classifier
                    output_feat_s = self.feat_extractor(X_batch_s)
                    output_class_s = self.data_classifier(output_feat_s)
                    loss = F.cross_entropy(output_class_s, y_batch_s)

                    # Target Domain Data
                    output_feat_t = self.feat_extractor(X_batch_t)
                    output_class_t = self.data_classifier(output_feat_t)
                    threshold_index = F.log_softmax(output_class_t).data.max(1)[0] > np.log(threshold_value)
                    loss_t_ent = entropy_loss(output_class_t[~threshold_index])
                    # pseudo-labels
                    y_batch_pseudo_t = output_class_t.data.max(1)[1][threshold_index]
                    if torch.sum(threshold_index) > 0:
                        loss_t = F.cross_entropy(output_class_t[threshold_index], y_batch_pseudo_t)
                    else:
                        loss_t = torch.zeros(1).cuda() if self.cuda else torch.zeros(1)
                    n_pseudo_labelled = torch.sum(threshold_index).item()

                    error = lambda_regul_s * loss + loss_t + lambda_regul * loss_t_ent
                    error.backward()

                    self.optimizer_data_classifier.step()

                self.logger.info(
                    "\nTrain epoch: {}/{} \tTotalLoss: {:.6f} LossS: {:.6f} LossT: {:.6f} "
                    "EntropyT: {:.6f}".format(epoch, self.nb_epochs + n_epochs_refinement, error.item(),
                                              lambda_regul_s * loss.item(), loss_t.item(), lambda_regul * loss_t_ent.item()))
                self.logger.info("N_Pseudo: {:.1f}".format(n_pseudo_labelled))

        self.loss_test_s, self.acc_test_s, self.w_acc_test_s, self.auc_test_s = \
            evaluate_data_classifier(self, is_test=True, is_target=False, is_criteo=True)
        self.loss_test_t, self.acc_test_t, self.w_acc_test_t, self.auc_test_t = \
            evaluate_data_classifier(self, is_test=True, is_target=True, is_criteo=True)
        self.loss_d_test, self.acc_d_test = \
            evaluate_domain_classifier(self, self.data_loader_test_s, self.data_loader_test_t, comments="Domain test",
                                       is_criteo=True)
