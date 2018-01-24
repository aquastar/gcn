from __future__ import division
from __future__ import print_function

import time
from itertools import cycle

import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc

from gcn.models import GCN, MLP, RAT, RAT_ELEMENT, RAT_after_GCN
from gcn.utils import *


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
    seed = 88
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'simu', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed', 'simu'
    flags.DEFINE_string('model', 'rat_element', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'rat'
    flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('early_stopping_lookback', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('eig_dim', 3000, 'Maximum eigen value number.')

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    support_inv = []

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
        print('gcn')
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
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
        support = element_rational(adj, FLAGS.max_degree, eig_dim=FLAGS.eig_dim)
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
            'support': tf.placeholder(tf.float32, shape=(FLAGS.max_degree + 1, FLAGS.eig_dim), name='support'),
            'eigen_vec': tf.placeholder(tf.float32, shape=(None, FLAGS.eig_dim), name='eigen_vec'),
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64), name='feat'),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels'),
            'labels_mask': tf.placeholder(tf.int32, name='labels_mask'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='num_feat_nzero')
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
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(cost)
            acc_val.append(acc)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and (
                    acc >= 1.0
                    # or
                    # acc_val[-1] < np.mean(acc_val[-(FLAGS.early_stopping_lookback + 1):-1])
                    # or
                    # cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping_lookback + 1):-1])
                    ):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
                                                          name=FLAGS.model)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
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
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
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
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(cost)
            acc_val.append(acc)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and acc >= 1.00  :
                print("Early stopping...")
                break

        print("Optimization Finished!")

        test_cost, test_acc, test_duration = evaluate_roc(features, support, y_test, test_mask, placeholders,
                                                          name=FLAGS.model, )
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
