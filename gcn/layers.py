from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

from gen_simulate import DATA_NUM

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_after_gcn(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, gcn_var=None, support_inv=None, lay_no=-1, **kwargs):
        super(GraphConvolution_after_gcn, self).__init__(**kwargs)

        self.lay_no = lay_no

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        # self.support_inv = placeholders['support_inv']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = tf.Variable(gcn_var[i], name='weights_' + str(i))
            # for option 1
            # for i in range(len(self.support)):
            #     self.vars['weights_de_' + str(i)] = tf.Variable(
            #         np.array(support_inv[i].toarray() / (FLAGS.max_degree + 1), dtype=np.float32),
            #         name='weights_de_' + str(i))

            # for option 2
            self.vars['weights_de'] = glorot([DATA_NUM, DATA_NUM], name='weights_de')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        supports_de = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)

            # Option 1
            # support_de = dot(self.support[i], self.vars['weights_de_' + str(i)], sparse=True)
            # supports_de.append(support_de)

        output = tf.add_n(supports)

        # Option 1
        # output_de = tf.add_n(supports_de)
        # output = dot(tf.py_func(np.linalg.pinv, [output_de], tf.float32), output)

        # Option 2
        output = dot(self.vars['weights_de'], output)
        # self.vars['weights_de'] = tf.Print(self.vars['weights_de'], [self.vars['weights_de']],
        #                                    message="self.vars['weights_de']:")

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_rat_test(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_rat_test, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([DATA_NUM, DATA_NUM], name='weights_' + str(i))
            for i in range(len(self.support)):
                self.vars['weights_de_' + str(i)] = glorot([DATA_NUM, DATA_NUM], name='weights_de_' + str(i))

            self.vars['weights_uni'] = glorot([input_dim, output_dim], name='weights_uni')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        supports_de = list()
        for i in range(len(self.support)):
            # if not self.featureless:
            #     pre_sup = dot(x, self.vars['weights_uni'], sparse=self.sparse_inputs)
            # else:
            #     pre_sup = self.vars['weights_uni' + str(i)]

            # pre_sup_de = self.vars['weights_de_' + str(i)]

            support = dot(self.support[i], self.vars['weights_' + str(i)], sparse=True)
            support_de = dot(self.support[i], self.vars['weights_de_' + str(i)], sparse=True)
            supports.append(support)
            supports_de.append(support_de)
        output = tf.add_n(supports)
        output_de = tf.add_n(supports_de)
        output = tf.div(output, output_de)

        pre_right = dot(x, self.vars['weights_uni'], sparse=self.sparse_inputs)
        output = dot(output, pre_right)
        # output = dot(output_de, output)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_Rational(Layer):
    """Graph convolution Rational layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_Rational, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(2 * len(self.support)):
                self.vars['weights_' + str(i)] = random_normal([DATA_NUM, DATA_NUM], name='weights_' + str(i))

            self.vars['weights_uni'] = glorot([input_dim, output_dim], name='weights_uni')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # shape: [input #, input_dim]
        x = inputs
        output = None

        with tf.name_scope("layer_droput"):
            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

        # rational convole
        pre_right = dot(x, self.vars['weights_uni'], sparse=self.sparse_inputs)

        supports_no = list()
        supports_de = list()

        with tf.name_scope("output_append"):
            for i in range(len(self.support)):
                sup = dot(self.support[i], self.vars['weights_' + str(i)], sparse=True)
                supports_no.append(sup)
                sup = dot(self.support[i], self.vars['weights_' + str(i + len(self.support))],
                          sparse=True)
                supports_de.append(sup)

        with tf.name_scope("output_add_n"):
            output_no = tf.add_n(supports_no)
            output_de = tf.add_n(supports_de)

        with tf.name_scope("output_div"):
            # pre_left = dot(output_no, tf.matrix_inverse(output_de))
            pre_left = tf.div(output_no, output_de)
            output = dot(pre_left, pre_right)

        # bias
        if self.bias:
            output += self.vars['bias']

        # try norm_batch
        # bn = tf.layers.batch_normalization(output,axis=1,center=True,scale=False)
        # output = slim.batch_norm(output, is_training=True)

        return self.act(output)


class GraphConvolution_Rational_Element(Layer):
    """Graph convolution Rational layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_Rational_Element, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.eigen_vec = placeholders['eigen_vec']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([1, FLAGS.max_degree + 1], name='weights')
            self.vars['weights_de'] = glorot([1, FLAGS.max_degree + 1], name='weights_de')

            self.vars['weights_uni'] = glorot([input_dim, output_dim], name='weights_uni')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # shape: [input #, input_dim]
        x = inputs

        with tf.name_scope("layer_droput"):
            # dropout
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

        # rational convolution
        pre_right = dot(x, self.vars['weights_uni'], sparse=self.sparse_inputs)
        # pre_right = tf.Print(pre_right, [pre_right], message="val: ")

        # calculate element wise eigenvalue approximation
        # g(lambda) = diag(P(lambda_1)/Q(lambda_1), P(lambda_2)/Q(lambda_2)...)

        sup = tf.squeeze(dot(self.vars['weights'], self.support))
        # sup = tf.reduce_sum(sup, 0)

        sup_de = tf.squeeze(dot(self.vars['weights_de'], self.support))
        # sup_de = tf.reduce_sum(sup_de, 0)

        eigen_val = tf.div(sup, sup_de)

        # multiply U and U^t, get estimated function of laplacian graph
        # self.eigen_vec = tf.Print(self.eigen_vec, [self.eigen_vec, tf.transpose(self.eigen_vec), tf.shape(self.eigen_vec)], message="self.eigen_vec: ")
        # output = tf.Print(output, [output], message="before pre_left: ")

        pre_left = dot(self.eigen_vec, tf.diag(eigen_val))
        # pre_left = tf.Print(pre_left, [pre_left, tf.shape(pre_left)], message="left pre_left: ")

        pre_left = dot(pre_left, tf.transpose(self.eigen_vec))
        # pre_left = tf.Print(pre_left, [pre_left, tf.shape(pre_left)], message="after pre_left: ")

        # pre_right = tf.Print(pre_right, [tf.shape(pre_right), tf.shape(pre_left)], message="This is a: ")

        # multiply feat x and parameters for going next layer
        output = dot(pre_left, pre_right)

        # bias
        if self.bias:
            output += self.vars['bias']

        # try norm_batch
        # bn = tf.layers.batch_normalization(output,axis=1,center=True,scale=False)
        # output = slim.batch_norm(output, is_training=True)

        return self.act(output)


class GraphConvolution_Rational_PFD(Layer):
    """Graph convolution Rational layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_Rational_PFD, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(2 * FLAGS.max_degree):
                self.vars['weights_' + str(i)] = random_normal([DATA_NUM, DATA_NUM], name='weights_' + str(i))

            self.vars['weights_uni'] = glorot([input_dim, output_dim], name='weights_uni')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # shape: [input #, input_dim]
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # rational convole
        pre_right = dot(x, self.vars['weights_uni'], sparse=self.sparse_inputs)

        supports = list()
        for i in range(FLAGS.max_degree):
            sup = tf.matrix_inverse(tf.add_n([dot(self.support[0], self.vars['weights_' + str(i)], sparse=True),
                                              self.vars['weights_' + str(i + FLAGS.max_degree)]]))
            supports.append(sup)

        pre_left = tf.add_n(supports)

        output = dot(pre_left, pre_right)

        # bias
        if self.bias:
            output += self.vars['bias']

        # try norm_batch
        # bn = tf.layers.batch_normalization(output,axis=1,center=True,scale=False)
        # output = slim.batch_norm(output, is_training=True)

        return self.act(output)


class test_layer(Layer):
    """Graph convolution Rational layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(test_layer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            for i in range(2 * len(self.support)):
                self.vars['weights_' + str(i)] = random_normal([1, 1], name='weights_' + str(i))

            self.vars['weights_uni'] = glorot([input_dim, output_dim], name='weights_uni')

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # shape: [input #, input_dim]
        # x = inputs

        # dropout
        # x = tf.nn.dropout(x, 1 - self.dropout)

        # rational convole
        pre_right = self.vars['weights_uni']

        supports_no = list()
        supports_de = list()
        for i in range(len(self.support)):
            sup = tf.multiply(self.support[i], self.vars['weights_' + str(i)])
            supports_no.append(sup)
            sup = tf.multiply(self.support[i], self.vars['weights_' + str(i + len(self.support))])
            supports_de.append(sup)

        output_no = tf.add_n(supports_no)
        output_de = tf.add_n(supports_de)
        pre_left = tf.div(output_no, output_de)
        output = tf.multiply(pre_left, pre_right)

        # bias
        if self.bias:
            output += self.vars['bias']

        # try norm_batch
        # bn = tf.layers.batch_normalization(output,axis=1,center=True,scale=False)
        # output = slim.batch_norm(output, is_training=True)

        # output = tf.nn.dropout(output, 1 - self.dropout)

        return self.act(output)
