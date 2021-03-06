import collections

from gen_simulate import DATA_NUM
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])

        self.loss += tf.nn.l2_loss(self.outputs - self.placeholders['target_mat'])

    def _accuracy(self):
        # self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                 self.placeholders['labels_mask'])

        self.accuracy = tf.nn.l2_loss(self.outputs - self.placeholders['target_mat'])

    def _build(self):
        # self.layers.append(GraphConvolution(
        #                                     input_dim=self.input_dim,
        #                                     featureless=True,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     sparse_inputs=True,
        #                                     logging=self.logging))

        self.layers.append(GraphConvolution(
            # input_dim=FLAGS.hidden1,
            input_dim=DATA_NUM,
            sparse_inputs=True,
            featureless=True,
            output_dim=DATA_NUM,
            # output_dim=self.output_dim,
            placeholders=self.placeholders,
            act=lambda x: x,
            dropout=True,
            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class RAT_after_GCN(Model):
    def __init__(self, placeholders, input_dim, gcn, support_inv, **kwargs):
        super(RAT_after_GCN, self).__init__(**kwargs)

        self.vars = [[], []]
        od = collections.OrderedDict(sorted(gcn.vars.items()))
        for i, (k, v) in enumerate(od.items()):
            self.vars[i / (FLAGS.max_degree + 1)].append(v)
        # self.support_inv = support_inv

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate * 10)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution_after_gcn(input_dim=self.input_dim,
                                                      output_dim=FLAGS.hidden1,
                                                      placeholders=self.placeholders,
                                                      act=tf.nn.relu,
                                                      dropout=True,
                                                      sparse_inputs=True,
                                                      gcn_var=self.vars[0],
                                                      # support_inv=self.support_inv,
                                                      lay_no=1,
                                                      logging=self.logging))

        self.layers.append(GraphConvolution_after_gcn(input_dim=FLAGS.hidden1,
                                                      output_dim=self.output_dim,
                                                      placeholders=self.placeholders,
                                                      act=lambda x: x,
                                                      dropout=True,
                                                      gcn_var=self.vars[1],
                                                      # support_inv=self.support_inv,
                                                      lay_no=2,
                                                      logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class RAT(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(RAT, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        with tf.device("/cpu:0"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])
        self.loss += tf.nn.l2_loss(self.outputs - self.placeholders['target_mat'])

    def _accuracy(self):
        # self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                 self.placeholders['labels_mask'])

        self.accuracy = tf.nn.l2_loss(self.outputs - self.placeholders['target_mat'])

    def _build(self):
        self.layers.append(GraphConvolution_rat_test(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden1,
                                                     placeholders=self.placeholders,
                                                     act=tf.nn.relu,
                                                     dropout=True,
                                                     sparse_inputs=True,
                                                     logging=self.logging))

        self.layers.append(GraphConvolution_rat_test(input_dim=FLAGS.hidden1,
                                                     output_dim=self.output_dim,
                                                     placeholders=self.placeholders,
                                                     act=lambda x: x,
                                                     dropout=True,
                                                     logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class RAT_ELEMENT(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(RAT_ELEMENT, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])

        # self.loss += tf.nn.l2_loss(self.outputs-self.placeholders['target_mat'])

        loss = tf.pow(self.outputs - self.placeholders['target_mat'], 2)
        self.loss += tf.reduce_sum(loss)

    def _accuracy(self):
        # self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                 self.placeholders['labels_mask'])

        # self.accuracy = tf.nn.l2_loss(self.outputs-self.placeholders['target_mat'])

        self.accuracy = tf.reduce_mean(tf.pow(self.outputs - self.placeholders['target_mat'], 2))

    def _build(self):
        # self.layers.append(GraphConvolution_Rational_Element(input_dim=self.input_dim,
        #                                                      output_dim=FLAGS.hidden1,
        #                                                      placeholders=self.placeholders,
        #                                                      act=tf.nn.relu,
        #                                                      dropout=True,
        #                                                      sparse_inputs=True,
        #                                                      logging=self.logging))

        self.layers.append(GraphConvolution_Rational_Element(
            # input_dim=FLAGS.hidden1,
            input_dim=DATA_NUM,
            # output_dim=self.output_dim,
            output_dim=DATA_NUM,
            placeholders=self.placeholders,
            sparse_inputs=True,
            act=lambda x: x,
            dropout=True,
            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class test_model(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(test_model, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_regress_loss(self.outputs, self.placeholders['labels'],
                                         self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_regress_accuracy(self.outputs, self.placeholders['labels'],
                                                self.placeholders['labels_mask'])

    def _build(self):
        # self.layers.append(test_layer(input_dim=self.input_dim,
        #                                              output_dim=FLAGS.hidden1,
        #                                              placeholders=self.placeholders,
        #                                              act=tf.nn.relu,
        #                                              dropout=True,
        #                                              sparse_inputs=True,
        #                                              logging=self.logging))

        self.layers.append(test_layer(input_dim=self.input_dim,
                                      output_dim=self.output_dim,
                                      placeholders=self.placeholders,
                                      act=lambda x: x,
                                      dropout=True,
                                      logging=self.logging))

    def predict(self):
        return self.outputs
