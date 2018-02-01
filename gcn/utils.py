import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from numpy import linalg as LA
from scipy.sparse.linalg import inv
from scipy.sparse.linalg.eigen.arpack import eigsh

from gen_simulate import graph_forge

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    # Simulated data
    if dataset_str == 'simu':
        return graph_forge(opt='label-graph-feat')

    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    ttt = []
    for _ in ally:
        ttt.append(_.tolist().index(1))

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, normalize=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if normalize:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = sp.coo_matrix(adj)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, target_mat=None, support_inv=None,
                        GCN_flag=False):
    """Construct feed dictionary."""

    feed_dict = dict()

    if GCN_flag:
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    elif FLAGS.model == 'rat_pre_train':
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        # feed_dict.update({placeholders['support_inv'][i]: support_inv[i] for i in range(len(support_inv))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    elif FLAGS.model == 'rat_element':
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support']: support[0]})
        feed_dict.update({placeholders['eigen_vec']: support[1]})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        feed_dict.update({placeholders['target_mat']: target_mat})
    else:
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        feed_dict.update({placeholders['target_mat']: target_mat})

    return feed_dict


def chebyshev_polynomials(adj, k, normalize=True):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    if normalize:
        adj_normalized = normalize_adj(adj)
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
    else:
        laplacian = sp.coo_matrix(adj)
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def chebyshev_polynomials_inv(adj, k):
    """Calculate rational up to order of k. Return a list of sparse matrices"""
    print("Calculating rational approximation up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return [inv(_) for _ in t_k]


def chebyshev_rational(adj, k):
    """Calculate rational up to order of k. Return a list of sparse matrices"""
    print("Calculating rational approximation up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def normal_rational(adj, k):
    """Calculate rational up to order of k. Return a list of sparse matrices"""
    print("Calculating rational approximation up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def normal_recurrence(scaled_lap, k):
        return scaled_lap.power(k)

    for i in range(2, k + 1):
        t_k.append(normal_recurrence(scaled_laplacian, i))

    return sparse_to_tuple(t_k)


def element_rational(adj, k, eig_dim=0, normalize=True):
    """Calculate rational up to order of k. Return a list of sparse matrices"""
    print("Calculating rational approximation up to order {}...".format(k))

    if normalize:
        adj_normalized = normalize_adj(adj)
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
    else:
        # laplacian =  degree_mt - ajd
        laplacian = sp.coo_matrix(adj)
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    eigen_val, eigen_vec = LA.eigh(laplacian.toarray())

    t_k = list()
    t_k.append(np.ones(eigen_val.shape))
    t_k.append(eigen_val)

    def normal_recurrence(scaled_lap, k):
        return np.power(scaled_lap, k)

    for i in range(2, k + 1):
        t_k.append(normal_recurrence(eigen_val, i))

    return [np.array(t_k), eigen_vec]


def pfd_rational(adj, k):
    """Calculate rational up to order of k. Return a list of sparse matrices"""
    print("Calculating rational approximation up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(scaled_laplacian)

    return sparse_to_tuple(t_k)
