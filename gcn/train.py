from __future__ import division
from __future__ import print_function

import cPickle as pk
import os.path
import time
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from models import GCN, MLP, RAT, RAT_ELEMENT, RAT_after_GCN, DATA_NUM
from utils import *


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, GCN_flag=False, support_inv=None):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, GCN_flag=GCN_flag,
                                        support_inv=support_inv)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


def evaluate_roc(features, support, labels, mask, placeholders, name='model', GCN_flag=False, support_inv=None):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders, GCN_flag=GCN_flag,
                                        support_inv=support_inv)
    loss, acc, outputs_logits, _ = sess.run([model.loss, model.accuracy, model.outputs, model.activations],
                                            feed_dict=feed_dict_val)
    # plot
    # fpr, tpr, _ = roc_curve(np.where(labels == 1)[1], np.argmax(outs_val[2], axis=1)[mask])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = np.shape(labels)[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], outputs_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), outputs_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot ROC curves for the multiclass problem
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'bisque', 'seagreen', 'magenta', 'b', 'c', 'r', 'plum', 'cyan',
         'lime'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}, Precison:{}'.format(name, acc))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(name)

    return loss, acc, (time.time() - t_test)


if __name__ == '__main__':

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'simu', 'Dataset string.')  # 'cora:2708', 'citeseer:3327', 'pubmed:19717', 'simu'
    flags.DEFINE_string('model', 'rat_element', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'rat'
    flags.DEFINE_float('learning_rate', 0.4, 'Initial learning rate.')  # 0.1-0.5 best for RAT, 0.01 best for GCN
    flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 200, 'Toerance for early stopping (# of epochs).')
    flags.DEFINE_integer('early_stopping_lookback', 5, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 4, 'Maximum Chebyshev polynomial degree.')  # 4 is better than 3 for RAT

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    mat_size = adj.shape[0]
    support_inv = []

    target_mat = np.eye(DATA_NUM)
    for i in xrange(DATA_NUM):
        for j in xrange(DATA_NUM):
            if i == j:
                target_mat[i, j] = 2
            elif i + 1 == j:
                target_mat[i, j] = -1
            elif i == j + 1:
                target_mat[i, j] = -1

    largest_eigval, _ = LA.eigh(target_mat)
    norm_lap = target_mat / largest_eigval[-1]
    eigen_val, eigen_vec = LA.eigh(norm_lap)
    eigen_val = np.power(eigen_val, 0.5)
    target_mat = np.dot(np.dot(eigen_vec, np.diag(eigen_val)), np.transpose(eigen_vec))

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj, normalize=False)]
        num_supports = 1
        model_func = GCN
        print('gcn')
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree, normalize=True)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
        print('gcn_cheby')
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
        print('dense')
    elif FLAGS.model == 'rat':
        support = normal_rational(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = RAT
        print('rational')
    elif FLAGS.model == 'rat_pre_train':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        support_inv = chebyshev_polynomials_inv(adj, FLAGS.max_degree)  # for the denominator == 1
        num_supports = 1 + FLAGS.max_degree
        model_func = RAT_after_GCN
        print('gcn_cheby')
    elif FLAGS.model == 'rat_element':
        support = None
        # cache eigendecompositon result for saving time
        if False and os.path.isfile('rat_element_sup.pkl'):
            support = pk.load(open('rat_element_sup.pkl', 'rb'))
        else:
            support = element_rational(adj, FLAGS.max_degree, normalize_lap=False)
            pk.dump(support, open('rat_element_sup.pkl', 'wb'))
        num_supports = 1 + FLAGS.max_degree
        model_func = RAT_ELEMENT
        print('rational')
    elif FLAGS.model == 'rat_pfd':
        support = pfd_rational(adj, FLAGS.max_degree)
        num_supports = 1
        model_func = RAT
        print('rational_pfd')
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = dict()
    if FLAGS.model == 'rat_element':  # Note eigen_dim does matter!
        placeholders = {
            'support': tf.placeholder(tf.float32, shape=(FLAGS.max_degree + 1, mat_size), name='support'),
            'eigen_vec': tf.placeholder(tf.float32, shape=(None, mat_size), name='eigen_vec'),
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64), name='feat'),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
            'labels_mask': tf.placeholder(tf.int32, name='labels_mask'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_feat_nzero'),
            'target_mat': tf.placeholder(tf.float32, shape=(mat_size, mat_size), name='target_mat')
        }
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # logdir = 'tflog/'
        # tb_write = tf.summary.FileWriter(logdir)
        # tb_write.add_graph(sess.graph)

        cost_val = []
        acc_val = []

        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders,
                                            target_mat=target_mat)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.layers[0].vars['weights'],
                             model.layers[0].vars['weights_de']], feed_dict=feed_dict)
            cost_val.append(outs[2])

            # Validation
            # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            # cost_val.append(cost)
            # acc_val.append(acc)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]),
                  # "val_loss=", "{:.5f}".format(cost),
                  # "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and (
                    cost_val[-1] < 0.001 or cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1])):
                print("Early stopping...")
                # print(outs[-2], outs[-1])
                print(
                    '({}+{}x+{}x^2+{}x^3+{}x^4)/({}+{}x+{}x^2+{}x^3+{}x^4)'.format(outs[-2][0][0], outs[-2][0][1],
                                                                                   outs[-2][0][2], outs[-2][0][3],
                                                                                   outs[-2][0][4], outs[-1][0][0],
                                                                                   outs[-1][0][1],
                                                                                   outs[-1][0][2], outs[-1][0][3],
                                                                                   outs[-1][0][4]))
                break

        print("Optimization Finished!")

        # test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
        #                                                   name=FLAGS.model)
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    elif FLAGS.model == 'rat_pre_train':
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'support_inv': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64), name='feat'),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
            'labels_mask': tf.placeholder(tf.int32, name='labels_mask'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_feat_nzero')
        }

        model_func = GCN
        model = model_func(placeholders, input_dim=features[2][1], logging=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # logdir = 'tflog/'
        # tb_write = tf.summary.FileWriter(logdir)
        # tb_write.add_graph(sess.graph)

        cost_val = []
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary

            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders, GCN_flag=True)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders, GCN_flag=True)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
                                                          name=FLAGS.model, GCN_flag=True)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

        # ################################
        # rational training after gcn
        # ################################
        print('Start rational training...')

        rat_model = RAT_after_GCN(placeholders, input_dim=features[2][1], gcn=model, support_inv=support_inv,
                                  logging=True)
        model = rat_model
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders,
                                            support_inv=support_inv)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders, support_inv=support_inv)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Rational Optimization Finished!")

        # Testing

        test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
                                                          name=FLAGS.model, support_inv=support_inv)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    else:
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='support') for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64, name='feat')),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
            'labels_mask': tf.placeholder(tf.int32, name='lables_mask'),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
            'target_mat': tf.placeholder(tf.float32, shape=(mat_size, mat_size), name='target_mat')
        }

        model = model_func(placeholders, input_dim=features[2][1], logging=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # logdir = 'tflog/'
        # tb_write = tf.summary.FileWriter(logdir)
        # tb_write.add_graph(sess.graph)

        cost_val = []
        acc_val = []

        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders,
                                            target_mat=target_mat)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.layers[0].vars],
                            feed_dict=feed_dict)
            cost_val.append(outs[1])

            # Validation
            # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            # cost_val.append(cost)
            # acc_val.append(acc)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]),
                  # "val_loss=", "{:.5f}".format(cost),
                  # "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and (
                    outs[2] < 0.00001 or cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1])):
                print("Early stopping...")
                print(
                    'poly={}+{}x+{}x^2+{}x^3+{}x^4'.format(outs[-2][0][0], outs[-2][0][1],
                                                           outs[-2][0][2], outs[-2][0][3],
                                                           outs[-2][0][4]))
                break

        print("Optimization Finished!")

        # test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
        #                                                   name=FLAGS.model, )
        # print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        #       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
