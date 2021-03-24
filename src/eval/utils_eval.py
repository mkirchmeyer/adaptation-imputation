import torch.nn.functional as F

from src.utils.utils_network import build_label_domain
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
np.set_printoptions(precision=4, formatter={'float_kind':'{:3f}'.format})


def evaluate_domain_classifier_class(model, data_loader, domain_label, is_target, is_criteo=False):
    model.feat_extractor.eval()
    model.data_classifier.eval()
    model.grl_domain_classifier.eval()

    loss = 0
    correct = 0
    ave_pred = 0

    if is_criteo:
        for dataI, dataC, _ in data_loader:
            target = build_label_domain(model, dataI.size(0), domain_label)
            if model.cuda:
                dataI, dataC, target = dataI.cuda(), dataC.cuda(), target.cuda()
            data = model.construct_input(dataI, dataC)
            output_feat = model.feat_extractor(data)
            output = model.grl_domain_classifier(output_feat)
            loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()
            ave_pred += torch.mean(F.softmax(output)[:, 1]).item()

    else:
        for data, _ in data_loader:
            target = build_label_domain(model, data.size(0), domain_label)
            if model.cuda:
                data, target = data.cuda(), target.cuda()
            if model.adapt_only_first:
                data = torch.mul(data, model.mask_t)
                output_feat = model.feat_extractor(data)
            else:
                if is_target and model.crop_dim != 0:
                    data = torch.mul(data, model.mask_t)
                output_feat = model.feat_extractor(data)
            output = model.grl_domain_classifier(output_feat)
            loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()
            ave_pred += torch.mean(F.softmax(output)[:, 1]).item()

    return loss, correct, ave_pred


def evaluate_domain_classifier(model, data_loader_s, data_loader_t, comments="Domain", is_criteo=False):
    model.feat_extractor.eval()
    model.data_classifier.eval()
    model.grl_domain_classifier.eval()

    loss_s, correct_s, ave_pred_s = evaluate_domain_classifier_class(model, data_loader_s, model.domain_label_s,
                                                                     is_target=False, is_criteo=is_criteo)
    loss_t, correct_t, ave_pred_t = evaluate_domain_classifier_class(model, data_loader_t, model.domain_label_t,
                                                                     is_target=True, is_criteo=is_criteo)
    loss_s += loss_t
    loss_s /= (len(data_loader_s) + len(data_loader_t))
    nb_source = len(data_loader_s.dataset)
    nb_target = len(data_loader_t.dataset)
    nb_tot = nb_source + nb_target
    model.logger.info(
        "{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy S: {}/{} ({:.2f}%), Accuracy T: {}/{} "
        "({:.2f}%)".format(
            comments, loss_s, correct_s + correct_t, nb_tot, 100. * (correct_s + correct_t) / nb_tot, correct_s,
            nb_source, 100. * correct_s / nb_source, correct_t, nb_target, 100. * correct_t / nb_target))
    return loss_s, (correct_s + correct_t) / nb_tot


def evaluate_data_classifier(model, is_test=True, is_target=False, is_criteo=False):
    model.feat_extractor.eval()
    model.data_classifier.eval()

    comments = ""
    if is_test:
        comments += "Test"
        if is_target:
            comments += " T"
            data_loader = model.data_loader_test_t
        else:
            comments += " S"
            data_loader = model.data_loader_test_s
    else:
        comments += "Train"
        if is_target:
            comments += " T"
            data_loader = model.data_loader_train_t
        else:
            comments += " S"
            data_loader = model.data_loader_train_s

    test_loss = 0
    naive_loss = 0
    correct = 0
    prediction_prob = []
    test_y = []
    naive_pred = 246872. / 946493.

    if is_criteo:
        for dataI, dataC, target in data_loader:
            target = target.view(-1)
            naive_output = torch.Tensor([1 - naive_pred, naive_pred]).repeat(dataI.size(0), 1)
            if model.cuda:
                dataI, dataC, target = dataI.cuda(), dataC.cuda(), target.cuda()
                naive_output = naive_output.cuda()
            data = model.construct_input(dataI, dataC)
            output_feat = model.feat_extractor(data)
            output = model.data_classifier(output_feat)
            test_loss += F.cross_entropy(output, target).item()
            naive_loss += F.nll_loss(torch.log(naive_output), target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()
            prediction_prob = np.hstack([prediction_prob, output.cpu().detach().numpy()[:, 1]])
            test_y = np.hstack([test_y, target.cpu().numpy()])
    else:
        for data, target in data_loader:
            target = target.view(-1)
            if model.cuda:
                data, target = data.cuda(), target.cuda()
            if model.adapt_only_first:
                data = torch.mul(data, model.mask_t)
            elif is_target and model.crop_dim != 0:
                data = torch.mul(data, model.mask_t)
            output_feat = model.feat_extractor(data)
            output = model.data_classifier(output_feat)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(data_loader)  # loss function already averages over batch size
    naive_loss /= len(data_loader)

    auc_roc = 0.0
    weighted_acc = 0.0

    model.logger.info(
        "{}: Mean Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            comments, test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    if is_criteo:
        model.logger.info("{}: Naive classifier mean Loss: {:.4f}".format(comments, naive_loss))
        auc_roc = roc_auc_score(test_y, prediction_prob)
        model.logger.info("{}: auc_roc: {:.4f}".format(comments, auc_roc))

    return test_loss, correct / len(data_loader.dataset), weighted_acc, auc_roc


##############
# Imputation #
##############

def compute_mse_imput(model, is_target=False, is_criteo=False):
    model.feat_extractor1.eval()
    model.feat_extractor2.eval()
    model.reconstructor.eval()

    dist = 0

    if is_target:
        data_loader = model.data_loader_test_t
        comments = "T"
    else:
        data_loader = model.data_loader_test_s
        comments = "S"

    if is_criteo:
        for dataI, dataC, _ in data_loader:
            if model.cuda:
                dataI, dataC = dataI.cuda(), dataC.cuda()

            data1 = model.construct_input1(dataI, dataC)
            output_feat1 = model.feat_extractor1(data1)
            output_feat2_reconstr = model.reconstructor(output_feat1)

            data2 = model.construct_input2(dataI)
            output_feat2 = model.feat_extractor2(data2)

            mean_norm = (torch.norm(output_feat2).item() + torch.norm(output_feat2_reconstr).item()) / 2
            dist += torch.dist(output_feat2, output_feat2_reconstr, 2).item() / mean_norm
    else:
        for data, _ in data_loader:
            if model.cuda:
                data = data.cuda()
            data2 = torch.mul(data, model.mask_2)
            output_feat2 = model.feat_extractor2(data2)

            data1 = torch.mul(data, model.mask_1)
            output_feat1 = model.feat_extractor1(data1)
            output_feat2_reconstr = model.reconstructor(output_feat1)

            mean_norm = (torch.norm(output_feat2).item() + torch.norm(output_feat2_reconstr).item()) / 2
            dist += torch.dist(output_feat2, output_feat2_reconstr, 2).item() / mean_norm

    dist /= len(data_loader)

    if model.logger is not None:
        model.logger.info(f"Mean NMSE {comments}: {dist}")

    return dist


def evaluate_data_imput_classifier(model, is_test=True, is_target=False, is_criteo=False):
    model.feat_extractor1.eval()
    model.feat_extractor2.eval()
    model.data_classifier.eval()
    model.reconstructor.eval()

    comments = "Imput"
    if is_test:
        comments += " test"
        if is_target:
            comments += " T"
            data_loader = model.data_loader_test_t
        else:
            comments += " S"
            data_loader = model.data_loader_test_s
    else:
        comments += " train"
        if is_target:
            comments += " T"
            data_loader = model.data_loader_train_t
        else:
            comments += " S"
            data_loader = model.data_loader_train_s

    test_loss = 0
    correct = 0
    prediction_prob = []
    test_y = []

    if is_criteo:
        for dataI, dataC, target in data_loader:
            target = target.view(-1)
            if model.cuda:
                dataI, dataC, target = dataI.cuda(), dataC.cuda(), target.cuda()

            data1 = model.construct_input1(dataI, dataC)
            output_feat1 = model.feat_extractor1(data1)

            if is_target:
                output_feat2 = model.reconstructor(output_feat1)
            else:
                data2 = model.construct_input2(dataI)
                output_feat2 = model.feat_extractor2(data2)

            output = model.data_classifier(torch.cat((output_feat1, output_feat2), 1))
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()
            prediction_prob = np.hstack([prediction_prob, output.cpu().detach().numpy()[:, 1]])
            test_y = np.hstack([test_y, target.cpu().numpy()])
    else:
        for data, target in data_loader:
            target = target.view(-1)
            if model.cuda:
                data, target = data.cuda(), target.cuda()
            data1 = torch.mul(data, model.mask_1)
            output_feat1 = model.feat_extractor1(data1)
            if is_target:
                output_feat2 = model.reconstructor(output_feat1)
            else:
                data2 = torch.mul(data, model.mask_2)
                output_feat2 = model.feat_extractor2(data2)
            output = model.data_classifier(torch.cat((output_feat1, output_feat2), 1))
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(data_loader)  # loss function already averages over batch size

    model.logger.info(
        "{}: Mean Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            comments, test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

    auc_roc = 0
    w_acc = 0
    if is_criteo:
        auc_roc = roc_auc_score(test_y, prediction_prob)
        model.logger.info("{}: auc_roc: {:.4f}".format(comments, auc_roc))

    return test_loss, correct / len(data_loader.dataset), w_acc, auc_roc


def evaluate_domain_imput_classifier(model, data_loader_s, data_loader_t, is_imputation, comments="Domain",
                                     is_criteo=False):
    model.feat_extractor1.eval()
    model.feat_extractor2.eval()
    model.data_classifier.eval()
    model.grl_domain_classifier1.eval()
    model.grl_domain_classifier2.eval()
    model.reconstructor.eval()

    if is_imputation:
        loss_s, correct_s = evaluate_domain_imput_classifier_class(model, data_loader_s,
                                                                   model.domain_label_true2,
                                                                   is_imputation=True, is_target=False,
                                                                   is_criteo=is_criteo)
        loss_t, correct_t = evaluate_domain_imput_classifier_class(model, data_loader_s,
                                                                   model.domain_label_fake2,
                                                                   is_imputation=True, is_target=True,
                                                                   is_criteo=is_criteo)
        compute_mse_imput(model, is_target=True, is_criteo=is_criteo)
        compute_mse_imput(model, is_target=False, is_criteo=is_criteo)
    else:
        loss_s, correct_s = evaluate_domain_imput_classifier_class(model, data_loader_s, model.domain_label_s,
                                                                   is_imputation=False, is_target=False,
                                                                   is_criteo=is_criteo)
        loss_t, correct_t = evaluate_domain_imput_classifier_class(model, data_loader_t, model.domain_label_t,
                                                                   is_imputation=False, is_target=True,
                                                                   is_criteo=is_criteo)
    loss_s += loss_t
    nb_source = len(data_loader_s.dataset)
    if is_imputation:
        nb_target = len(data_loader_s.dataset)
    else:
        nb_target = len(data_loader_t.dataset)
    nb_tot = nb_source + nb_target
    model.logger.info(
        "{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy S: {}/{} ({:.2f}%), Accuracy T: "
        "{}/{} ({:.0f}%)".format(
            comments, loss_s, correct_s + correct_t, nb_tot, 100. * (correct_s + correct_t) / nb_tot, correct_s,
            nb_source, 100. * correct_s / nb_source, correct_t, nb_target, 100. * correct_t / nb_target))
    return loss_s, (correct_s + correct_t) / nb_tot


def evaluate_domain_imput_classifier_class(model, data_loader, domain_label, is_imputation=True, is_target=False,
                                           is_criteo=False):
    model.feat_extractor1.eval()
    model.feat_extractor2.eval()
    model.data_classifier.eval()
    model.grl_domain_classifier1.eval()
    model.grl_domain_classifier2.eval()
    model.reconstructor.eval()

    loss = 0
    correct = 0

    if is_criteo:
        for dataI, dataC, target in data_loader:
            target = target.view(-1)
            if model.cuda:
                dataI, dataC, target = dataI.cuda(), dataC.cuda(), target.cuda()
            if is_imputation:
                if not is_target:
                    data2 = model.construct_input2(dataI)
                    output_feat2 = model.feat_extractor2(data2)
                    output = model.grl_domain_classifier2(output_feat2)
                else:
                    data1 = model.construct_input1(dataI, dataC)
                    output_feat1 = model.feat_extractor1(data1)
                    output_feat2 = model.reconstructor(output_feat1)
                    output = model.grl_domain_classifier2(output_feat2)
            else:
                data1 = model.construct_input1(dataI, dataC)
                output_feat1 = model.feat_extractor1(data1)
                output_feat2 = model.reconstructor(output_feat1)
                output = model.grl_domain_classifier1(torch.cat((output_feat1, output_feat2), 1))
            loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    else:
        for data, _ in data_loader:
            target = build_label_domain(model, data.size(0), domain_label)
            if model.cuda:
                data, target = data.cuda(), target.cuda()
            if is_imputation:
                if not is_target:
                    data2 = torch.mul(data, model.mask_2)
                    output_feat2 = model.feat_extractor2(data2)
                    output = model.grl_domain_classifier2(output_feat2)
                else:
                    data1 = torch.mul(data, model.mask_1)
                    output_feat1 = model.feat_extractor1(data1)
                    output_feat2 = model.reconstructor(output_feat1)
                    output = model.grl_domain_classifier2(output_feat2)
            else:
                data1 = torch.mul(data, model.mask_1)
                output_feat1 = model.feat_extractor1(data1)
                output_feat2 = model.reconstructor(output_feat1)
                output = model.grl_domain_classifier1(torch.cat((output_feat1, output_feat2), 1))
            loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    return loss, correct
