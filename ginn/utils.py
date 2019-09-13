# Copyright 2019 Indro Spinelli. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import random
import numpy as np
import networkx as nx
from math import sqrt
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error


def encode_classes(col):
    """
    Input:  categorical vector of any type
    Output: categorical vector of int in range 0-num_classes
    """
    classes = set(col)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, col)), dtype=np.int32)
    return labels


def onehot_classes(col):
    """
    Input:  categorical vector of int in range 0-num_classes
    Output: one-hot representation of the input vector
    """
    col2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
    col2onehot[np.arange(col.size), col] = 1
    return col2onehot


def miss_deg_num(nxg, mask):
    """
    Input: networkx graph, mask representing missing node features
    Outputs:
        vector containing range of missing node features (sorted)
        vector containing mean degree for each value in the range
        vector containing the number of nodes in the graph for each
            value in the range
    """
    list_missingness = list(mask.shape[1] - mask.sum(axis=1))
    t = []
    for i, j in zip(list_missingness, nxg.degree):
        t.append((int(i), j[1]))
    d_deg = dict(
        set(
            (a, sum(y for x, y in t if x == a) / sum(1 for x, _ in t if x == a))
            for a, b in t
        )
    )
    d_num = dict((i, list_missingness.count(i)) for i in set(list_missingness))
    sk = np.sort(list(d_deg.keys()))
    deg = [d_deg[i] for i in sk]
    num = [d_num[i] for i in sk]
    return sk, deg, num


def degrade_dataset(X, missingness, rand, v):
    """
    Inputs:
        dataset to corrupt
        % of data to eliminate[0,1]
        rand random state
        replace with = 'zero' or 'nan'
      Outputs:
        corrupted Dataset 
        binary mask
    """
    X_1d = X.flatten()
    n = len(X_1d)
    mask_1d = np.ones(n)

    corrupt_ids = random.sample(range(n), int(missingness * n))
    for i in corrupt_ids:
        X_1d[i] = v
        mask_1d[i] = 0

    cX = X_1d.reshape(X.shape)
    mask = mask_1d.reshape(X.shape)

    return cX, mask


def data2onehot(data, mask, num_cols, cat_cols):
    """
    Inputs:
        corrupted dataset
        mask of the corruption
        vector contaning indexes of columns having numerical values
        vector contaning indexes of columns having categorical values
   Outputs:
        one-hot encoding of the dataset
        one-hot encoding of the corruption mask
        mask of the numerical entries of the one-hot dataset
        mask of the categorical entries of the one-hot dataset
        vector containing start-end idx for each categorical variable
    """
    # find most frequent class
    fill_with = []
    for col in cat_cols:
        l = list(data[:, col])
        fill_with.append(max(set(l), key=l.count))

    # meadian imputation
    filled_data = data.copy()
    for i, col in enumerate(cat_cols):
        filled_data[:, col] = np.where(mask[:, col], filled_data[:, col], fill_with[i])

    for i, col in enumerate(num_cols):
        filled_data[:, col] = np.where(
            mask[:, col], filled_data[:, col], np.nanmedian(data[:, col])
        )

    # encode into 0-N lables
    for col in cat_cols:
        filled_data[:, col] = encode_classes(filled_data[:, col])

    num_data = filled_data[:, num_cols]
    num_mask = mask[:, num_cols]
    cat_data = filled_data[:, cat_cols]
    cat_mask = mask[:, cat_cols]

    # onehot encoding for masks and categorical variables
    onehot_cat = []
    cat_masks = []
    for j in range(cat_data.shape[1]):
        col = cat_data[:, j].astype(int)
        col2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
        col2onehot[np.arange(col.size), col] = 1
        mask2onehot = np.zeros((col.size, col.max() + 1), dtype=float)
        for i in range(cat_data.shape[0]):
            if cat_mask[i, j] > 0:
                mask2onehot[i, :] = 1
            else:
                mask2onehot[i, :] = 0
        onehot_cat.append(col2onehot)
        cat_masks.append(mask2onehot)

    cat_starting_col = []
    oh_data = num_data
    oh_mask = num_mask

    # build the big mask
    for i in range(len(onehot_cat)):
        cat_starting_col.append(oh_mask.shape[1])

        oh_data = np.c_[oh_data, onehot_cat[i]]
        oh_mask = np.c_[oh_mask, cat_masks[i]]

    oh_num_mask = np.zeros(oh_data.shape)
    oh_cat_mask = np.zeros(oh_data.shape)

    # build numerical mask
    oh_num_mask[:, range(num_data.shape[1])] = num_mask

    # build categorical mask
    oh_cat_cols = []
    for i in range(len(cat_masks)):
        start = cat_starting_col[i]
        finish = start + cat_masks[i].shape[1]
        oh_cat_mask[:, start:finish] = cat_masks[i]
        oh_cat_cols.append((start, finish))

    return oh_data, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols


def preprocess(data, mask, num_cols, cat_cols):
    a, b, c, d, e = data2onehot(data, mask, num_cols, cat_cols)
    l = list(a, b, c, d, e)
    return l


def similarity(x, mx, y, my, metric, weight_missingness):
    """
    Inputs:
        feature vector
        mask of the feature vector
        feature vector
        mask of the feature vector
        boolean, weight or not the missing elements on the feature vector
        metric, cosine similarity or euclidean similarity
    Output:
        similarity score of the two feature vectors
    """
    xy_to_keep = np.multiply(mx, my)

    if np.sum(xy_to_keep) < 1.0:
        return 0.0

    # keep elements present in both vectors
    rx = np.multiply(x, xy_to_keep)
    ry = np.multiply(y, xy_to_keep)

    if metric == "cosine":
        dot = np.dot(rx, ry)
        den = np.linalg.norm(rx) * np.linalg.norm(ry)
        sim = float(dot / max(den, 1e-6))
    elif metric == "euclidean":
        sim = 1 / (1 + np.linalg.norm(rx - ry))

    if weight_missingness:
        # compute weighting factor
        ones = np.ones(xy_to_keep.shape)
        wdot = np.dot(xy_to_keep, ones)
        wden = np.linalg.norm(xy_to_keep) * np.linalg.norm(ones)
        sim *= float(wdot / wden)

    return sim


def similarity_matrix(X, mask, metric, weight_missingness):
    """
    Inputs:
        corrupted dataset
        mask of the corruption
        boolean, weight or not the missing elements on the feature vector
        metric, cosine similarity or euclidean similarity
    Output:
        matrix containing pairwise similarity
    """
    obs_len = len(X[:, 0])
    M_cos_sim = np.zeros((obs_len, obs_len), dtype=float)
    for i in range(0, obs_len):
        for j in range(i, obs_len):
            M_cos_sim[i][j] = similarity(
                X[i], mask[i], X[j], mask[j], metric, weight_missingness
            )
            M_cos_sim[j][i] = M_cos_sim[i][j]
    return M_cos_sim


def compute_weighted_adj(M, percentile):
    """
    Inputs:
        similarity matrix
        percentile of connections to keep
    Output:
        weighted adjacency matrix
    """
    if False:
        m_len = len(M[0, :])
        m_sample = []
        for i in range(0, m_len):
            for j in range(i, m_len):
                if i != j:
                    m_sample.append(M[i][j])

        treshold = np.percentile(m_sample, percentile)
        M[M < treshold] = 0.0
        np.fill_diagonal(M, 0.0)
    else:
        for i in range(M.shape[0]):
            # first pruning
            treshold = np.percentile(M[i], percentile)
            M[i][M[i] < treshold] = 0.0
        # second pruning
        v = M.flatten()
        treshold = np.percentile(v, percentile)

        M[M < treshold] = 0.0
        np.fill_diagonal(M, 0.0)
    return M


def dataset2nxg(cX, mask, percentile, metric, weight_missingness):
    """
    Inputs:
        corrupted dataset
        mask of the corruption
        percentile of connections to keep
        boolean, weight or not the missing elements on the feature vector
        metric, cosine similarity or euclidean similarity
    Outputs:
       networkx MultiDiGraph
    """

    cX_sim = similarity_matrix(cX, mask, metric, weight_missingness)
    cX_wadj = compute_weighted_adj(cX_sim, percentile)
    ngx = nx.DiGraph(cX_wadj)
    return ngx


def new_edges(x_tr, mask_tr, x_te, mask_te, percentile, metric, weight_missingness):
    """
    Inputs:
        old dataset
        old dataset mask of the corruption
        new dataset
        new dataset mask of the corruption
        percentile of connections to keep
        boolean, weight or not the missing elements on the feature vector
        metric, cosine similarity or euclidean similarity
    Output:
        list containg [src, dest] of edges that needs to be added in order
            to integrate the new dataset in the older one 
    """

    M_sim = np.zeros((x_te.shape[0], np.r_[x_tr, x_te].shape[0]))
    for i in range(x_te.shape[0]):
        for j in range(x_tr.shape[0]):
            M_sim[i][j] = similarity(
                x_tr[j], mask_tr[j], x_te[i], mask_te[i], metric, weight_missingness
            )
    for i in range(x_te.shape[0]):
        for j in range(0, i):
            M_sim[i][x_tr.shape[0] + j] = similarity(
                x_te[i], mask_te[i], x_te[j], mask_te[j], metric, weight_missingness
            )
    if False:
        # One step pruning
        treshold = np.percentile(M_sim.flatten(), percentile)
        edges = np.argwhere(M_sim >= treshold)
    else:
        # Two steps pruning
        for i in range(M_sim.shape[0]):
            treshold = np.percentile(M_sim[i], percentile)
            M_sim[i][M_sim[i] < treshold] = 0.0

        treshold = np.percentile(M_sim.flatten(), percentile)

        M_sim[M_sim < treshold] = 0.0
        edges = np.argwhere(M_sim > 0.0)

    return edges


def imputation_accuracy(target, output, inv_mask):
    """
    Inputs:
        target dataset
        imputed dataset
        inverse of the matrix used for the dataset corruption
    Outputs:
        mean absolute error
        mean root mean squared error
    """
    mae = mean_absolute_error(target[inv_mask], output[inv_mask])
    rmse = sqrt(mean_squared_error(target[inv_mask], output[inv_mask]))
    return mae, rmse


def proper_onehot(x, oh_cat_cols):
    """
    Inputs:
        dataset with soft categorical variables
        vector containing start-end idx for each categorical variable
    Output: dataset with hard categorical variables
    """
    for start, finish in oh_cat_cols:
        x[:, start:finish] = (
            x[:, start:finish] == x[:, start:finish].max(1)[:, None]
        ).astype(int)
    return x


def batch_mask(size, batch_size):
    """
    Inputs:
        lenght of the mask
        number of elements to mask
    Output: mask for the critic
    """
    b_mask = np.zeros(size)
    pos = random.sample(range(size), batch_size)
    b_mask[pos] = 1
    return b_mask
