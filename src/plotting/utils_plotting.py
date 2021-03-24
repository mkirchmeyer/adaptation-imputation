import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas import DataFrame
from sklearn.manifold import TSNE
import numpy as np
import torch
import os

from src.utils.utils_network import get_data_classifier, get_feature_extractor


def colored_scattered_plot(X_train_s, X_train_t, y_sparse_train_s, y_sparse_train_t, name=None, data_type=None):
    # scatter plot, dots colored by class value
    df_s = DataFrame(dict(x=X_train_s[:, 0], y=X_train_s[:, 1], label=y_sparse_train_s))
    df_t = DataFrame(dict(x=X_train_t[:, 0], y=X_train_t[:, 1], label=y_sparse_train_t))
    colors_s = {0: 'red', 1: 'blue'}
    colors_t = {0: 'magenta', 1: 'cyan'}
    marker_s = {0: 'o', 1: 'o'}
    marker_t = {0: 'x', 1: 'x'}
    fig, ax = plt.subplots()
    grouped_s = df_s.groupby('label')
    grouped_t = df_t.groupby('label')
    for key, group in grouped_s:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=str(key) + "_source", color=colors_s[key],
                   marker=marker_s[key])
    for key, group in grouped_t:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=str(key) + "_target", color=colors_t[key],
                   marker=marker_t[key])
    if data_type:
        plt.title(f"Source and target {data_type} data distribution on dimension 1 and 2")
        plt.savefig(f"../../figures/{data_type}/{name}/data_{data_type}")


def plot_data_frontier(X_train, X_test, y_train, y_test, net, data_type=None, name=None):
    dim = X_train.shape[1]
    feat_extract = get_feature_extractor(net)
    data_class = get_data_classifier(net)
    if dim == 2:
        colored_scattered_plot(X_train, X_test, y_train, y_test)

        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]

        Z_feat = feat_extract((torch.from_numpy(np.atleast_2d(Z)).float()))
        Z_class = data_class(Z_feat)
        classe = Z_class.data.max(1)[1].numpy()

        classe = classe.reshape(xx.shape)
        plt.contour(xx, yy, classe, levels=[0], colors="r")

    # Get embeddings
    subset = 100

    x = torch.from_numpy(X_train[:subset, :]).float()
    X_train_map = feat_extract(x).data.numpy()
    x = torch.from_numpy(X_test[:subset, :]).float()
    X_test_map = feat_extract(x).data.numpy()

    emb_all = np.vstack([X_train_map, X_test_map])
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(emb_all)

    num = X_train[:subset, :].shape[0]

    colored_scattered_plot(pca_emb[:num, :], pca_emb[num:, :], y_train[:subset], y_test[:subset])
    plt.show()

    if data_type is not None and name is not None:
        plt.savefig(f"../../figures/{data_type}/{name}/frontier_{data_type}")


def plot_data_frontier_digits(net, data_loader_s, data_loader_t, epoch=None, is_imput=False, is_pca=False):
    if not is_imput:
        feat_extract = net.feat_extractor
    else:
        feat_extract1 = net.feat_extractor1.eval()
        recons = net.reconstructor.eval()
    model_config = net.model_config

    S_batches = iter(data_loader_s)
    T_batches = iter(data_loader_t)
    X_train, y_train = next(S_batches)
    X_test, y_test = next(T_batches)

    if net.cuda:
        X_train = X_train.cuda()
        X_test = X_test.cuda()

    if not is_imput:
        X_train_map = feat_extract(X_train).cpu().data.numpy()
        X_test_map = feat_extract(X_test).cpu().data.numpy()
    else:
        data1_train = torch.mul(X_train, net.mask_1)
        output_feat1_train = feat_extract1(data1_train)
        output_feat2_train = recons(output_feat1_train)
        X_train_map = torch.cat((output_feat1_train, output_feat2_train), 1).cpu().data.numpy()

        data1_test = torch.mul(X_test, net.mask_1)
        output_feat1_test = feat_extract1(data1_test)
        output_feat2_test = recons(output_feat1_test)
        X_test_map = torch.cat((output_feat1_test, output_feat2_test), 1).cpu().data.numpy()

    emb_all = np.vstack([X_train_map, X_test_map])

    if is_pca:
        pca = PCA(n_components=2)
    else:
        pca = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    pca_emb = pca.fit_transform(emb_all)

    num = X_train.shape[0]

    fig_dir = f"./figures/{model_config.mode}/{model_config.source}_{model_config.target}"
    try:
        os.mkdir("./figures")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"./figures/{model_config.mode}")
    except FileExistsError:
        pass
    try:
        os.mkdir(fig_dir)
        print(f"Directory {fig_dir} created ")
    except FileExistsError:
        pass

    colored_scattered_plot_digits(pca_emb[:num, :], pca_emb[num:, :], y_train.data.numpy(), y_test.data.numpy(),
                                  is_pca=is_pca)
    plt.savefig(f"{fig_dir}/frontier_{epoch}")
    colored_scattered_plot_digits(pca_emb[:num, :], pca_emb[num:, :], y_train.data.numpy(), y_test.data.numpy(), mode=2,
                                  is_pca=is_pca)
    plt.savefig(f"{fig_dir}/frontier_label_{epoch}")


def colored_scattered_plot_digits(X_train_s, X_train_t, y_sparse_train_s, y_sparse_train_t, mode=1, is_pca=False):
    # scatter plot, dots colored by class value
    df_s = DataFrame(dict(x=X_train_s[:, 0], y=X_train_s[:, 1], label=y_sparse_train_s))
    df_t = DataFrame(dict(x=X_train_t[:, 0], y=X_train_t[:, 1], label=y_sparse_train_t))

    fig, ax = plt.subplots()

    grouped_s = df_s.groupby('label')
    grouped_t = df_t.groupby('label')

    colors = plt.cm.rainbow(np.linspace(0, 1, 10))

    if mode == 1:
        colors = {0: 'red', 1: 'blue'}
        for key, group in grouped_s:
            group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[0], marker='x')
        for key, group in grouped_t:
            group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[1], marker='o')

    elif mode == 2:
        for key, group in grouped_s:
            group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key], marker='x')
        for key, group in grouped_t:
            group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key], marker='o')

    if is_pca:
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
