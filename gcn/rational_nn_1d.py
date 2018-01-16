from __future__ import print_function

import tensorflow as tf
import time
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial as P
from models import test_model

low_border = -10.0
hi_border = 10.0


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def f(x, opt=9):
    if opt == 1:
        return np.sqrt(abs(x - 3))
    elif opt == 2:
        return np.minimum(abs(x), np.exp(x))
    elif opt == 3:
        return np.sign(x)
    elif opt == 4:
        return np.sqrt(abs(x - 3))
    elif opt == 5:
        return 10 * abs(x)
    elif opt == 6:
        return 1 + x/(abs(x)+0.1)
    elif opt == 7: # Need all positive X
        return np.sqrt(x)
    elif opt == 8:
        return np.maximum(.85, np.sin(x + x ** 2)) - x / 20
    elif opt == 9:
        return -x - x ** 2 + np.exp(-(30 * (x - .47)) ** 2)


# set a function f, such as f = chebfun('sqrt(abs(x-3))',[0,4],'splitting','on');
# generate data samples from function f, [input, output]
def generate_samples(DATA_NUM=1000, low=low_border, hi=hi_border):
    x_all = (hi - low) * np.random.random_sample((DATA_NUM, 1)) + low
    y_all = f(x_all)

    train_rate = 0.7
    val_rate = 0.2
    test_rate = 0.1

    train_num = int(DATA_NUM * train_rate)
    val_num = int(DATA_NUM * val_rate)
    test_num = int(DATA_NUM * test_rate)

    idx_train = range(train_num)
    idx_val = range(train_num, train_num + val_num)
    idx_test = range(DATA_NUM - test_num, DATA_NUM)

    train_mask = sample_mask(idx_train, y_all.shape[0])
    val_mask = sample_mask(idx_val, y_all.shape[0])
    test_mask = sample_mask(idx_test, y_all.shape[0])

    y_train = np.zeros(y_all.shape)
    y_val = np.zeros(y_all.shape)
    y_test = np.zeros(y_all.shape)

    y_train[train_mask] = y_all[train_mask]
    y_val[val_mask] = y_all[val_mask]
    y_test[test_mask] = y_all[test_mask]

    return x_all, y_all, y_train, y_val, y_test, train_mask, val_mask, test_mask


def construct_feed_dict(support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    return feed_dict


def evaluate(support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(support, labels, mask, placeholders)
    outs_val = sess.run([model.outputs, model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


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


def get_cheby_support(adj, k):
    t_k = list()
    t_k.append(np.ones(adj.shape))
    t_k.append(adj)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap * t_k_minus_one - t_k_minus_two

    for i in range(2, k):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], adj))

    return t_k


def get_normal_support(adj, k):
    t_k = list()
    t_k.append(np.ones(adj.shape))
    t_k.append(adj)

    for i in range(2, k):
        t_k.append(np.power(adj, i))

    return t_k


if __name__ == '__main__':
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'simu', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'simu'
    flags.DEFINE_string('model', 'rat', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'rat'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    # generate and divide data
    x_all, y_all, y_train, y_val, y_test, val_mask, train_mask, test_mask = generate_samples()

    # generate support
    num_supports = 1 + FLAGS.max_degree
    support = get_normal_support(y_train, num_supports)

    # init place holders and model
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'labels': tf.placeholder(tf.float32, shape=(None, 1)),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0.5, shape=()),
    }
    model = test_model(placeholders, input_dim=1)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(200):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: 0.5})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        _, cost, acc, duration = evaluate(support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(50 + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    test_output, test_cost, test_acc, test_duration = evaluate(support, y_test, test_mask, placeholders)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # plot the orignal function and approximated one
    # np.sqrt(abs(input - 3))

    # init x points
    t1 = np.arange(low_border, hi_border, 0.5)
    t2 = np.arange(low_border, hi_border, 0.05)

    # apply normal polynomial approximation
    x_all_ind = np.argsort(np.squeeze(x_all))
    x_poly = np.squeeze(x_all)[x_all_ind]
    y_poly = np.squeeze(y_all)[x_all_ind]
    z_poly = np.polyfit(x_poly, y_poly, 5)
    p = np.poly1d(z_poly)
    y_poly_pred = p(t1)

    # apply Chebyshev polynomial approximation
    c_poly = P.chebyshev.chebfit(x_poly, y_poly, 5)
    cp = P.Polynomial(P.chebyshev.cheb2poly(c_poly))
    cy_poly_pred = cp(t1)

    # appply neural rational approxiamtion
    t_pred = np.squeeze(x_all[test_mask])
    t_pred_ind = np.argsort(t_pred)
    t_pred = t_pred[t_pred_ind]
    y_pred = np.squeeze(y_test[test_mask])[t_pred_ind]

    # plt.figure(1)

    # plt.subplot(411)
    # plt.plot(t1, f(t1), 'b--')
    #
    # plt.subplot(412)
    # plt.plot(t1, y_poly_pred, 'y--')
    #
    # plt.subplot(413)
    # plt.plot(t1, cy_poly_pred, 'g--')
    #
    # plt.subplot(414)
    # plt.plot(t_pred, y_pred, 'r--')

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
    # l1, = ax1.plot(t1, f(t1), 'b--')
    # l2, = ax2.plot(t1, y_poly_pred, 'y--')
    # l3, = ax3.plot(t1, cy_poly_pred, 'g--')
    # l4, = ax4.plot(t_pred, y_pred, 'r--')
    # plt.legend([l1, l2, l3, l4], ["func", "poly", "cheby", "rat"])

    ax1.plot(t1, f(t1), 'b--', label='func')
    ax1.legend(loc="upper right")
    ax2.plot(t1, y_poly_pred, 'y--',label='poly')
    ax2.legend(loc="upper right")
    ax3.plot(t1, cy_poly_pred, 'g--',label='cheby')
    ax3.legend(loc="upper right")
    ax4.plot(t_pred, y_pred, 'r--',label='rat')
    ax4.legend(loc="upper right")

    ax1.set_title('approximation')

    plt.show()
