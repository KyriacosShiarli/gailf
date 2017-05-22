import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import pdb
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0,indim = None):
    if indim is None:
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    else:
        w = tf.get_variable(name + "/w", [indim, size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class LSMTAbstract(object):
    def __init__(self,ob_space,ac_space):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        self.conv_out = flatten(x)

    def attach_heads(self,type,size):
        with tf.variable_scope(type):
            x = self.conv_out
            #if type =="reward":
                #self.action = tf.placeholder(tf.float32, [None] + list([self.ac_space]))
                #x = tf.concat([self.conv_out,self.action],1)
            x = tf.expand_dims(x, [0])

            if use_tf100_api:
                lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            else:
                lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

            if type == 'reward': self.r_state_size = lstm.state_size
            if type == 'policy': self.p_state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]
            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            if type == 'reward': self.r_state_init = [c_init, h_init]
            if type == 'policy': self.p_state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            if type == 'reward': self.r_state_in = [c_in, h_in]
            if type == 'policy': self.p_state_in = [c_in, h_in]
            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state

            if type == 'reward':self.r_state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            if type == 'policy': self.p_state_out = [lstm_c[:1, :], lstm_h[:1, :]]
            x = tf.reshape(lstm_outputs, [-1, size])
            if type == "policy":
                self.p_logits = linear(x, self.ac_space, "action", normalized_columns_initializer(0.01))
                self.probs = tf.nn.softmax(self.p_logits, name='action_probs')
                self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
                self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
                self.sample = categorical_sample(self.p_logits, self.ac_space)[0, :]

            if type == "reward":
                self.r_logits = linear(x, self.ac_space, "r_logits", normalized_columns_initializer(0.01))
                eps = tf.constant(1e-5)
                self.d = tf.nn.softmax(self.r_logits)
                self.rew = tf.log(self.d + eps)
                self.rew_norm = self.rew /1 + 1.8 #(-tf.log(eps))

class LSTMImitator(LSMTAbstract):
    def __init__(self,ob_space,ac_space):
        size = 256
        super(LSTMImitator, self).__init__(ob_space,ac_space)
        self.type = 'lstm'
        self.shared = True
        self.attach_heads('reward',size)
        self.attach_heads('policy', size)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, ob,*args):
        c, h = args[0], args[1]
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.p_state_out,
                        {self.x: ob, self.p_state_in[0]: c, self.p_state_in[1]: h})
    def get_probs(self, ob, *args):
        c, h = args[0], args[1]
        sess = tf.get_default_session()
        return sess.run([self.vf, self.probs] + self.state_out,
                        {self.x: ob, self.p_state_in[0]: c, self.p_state_in[1]: h})
    def value(self, ob,*args):
        c, h = args[0], args[1]
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.p_state_in[0]: c, self.p_state_in[1]: h})[0]
    def get_initial_features(self):
        return self.p_state_init, self.r_state_init
    #def reward(self, ob, ac, c, h):
    #    sess = tf.get_default_session()
    #    return sess.run([self.rew_norm, self.d, self.r_state_out],
    #                {self.x: ob, self.action: ac, self.r_state_in[0]: c, self.r_state_in[1]: h})
    def reward(self, ob,*args):
        ac,c,h = args[0],args[1],args[2]
        sess = tf.get_default_session()
        return sess.run([self.rew_norm, self.d, self.r_state_out],
                        {self.x: ob, self.r_state_in[0]: c, self.r_state_in[1]: h})


class LSTMPolicy(LSMTAbstract):
    def __init__(self,ob_space,ac_space):
        size = 256
        with tf.variable_scope('policy'):
            super(LSTMPolicy, self).__init__(ob_space,ac_space)
        self.attach_heads('policy', size)
        self.type = 'lstm'
        with tf.variable_scope('policy'):
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    def get_initial_features(self):
        return self.p_state_init
    def act(self, ob, *args):
        c,h = args[0],args[1]
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.p_state_out,
                        {self.x: ob, self.p_state_in[0]: c, self.p_state_in[1]: h})
    def get_probs(self, ob,*args):
        c,h = args[0],args[1]
        sess = tf.get_default_session()
        return sess.run([self.vf, self.probs] + self.state_out,
                        {self.x: ob, self.p_state_in[0]: c, self.p_state_in[1]: h})
    def value(self, ob, *args):
        c,h = args[0],args[1]
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: ob, self.p_state_in[0]: c, self.p_state_in[1]: h})[0]


class LSTMDiscriminator(LSMTAbstract):
    def __init__(self,ob_space,ac_space):
        size = 256
        self.type ='lstm'
        with tf.variable_scope('reward'):
            super(LSTMDiscriminator, self).__init__(ob_space,ac_space)
        self.attach_heads('reward', size)
        with tf.variable_scope('reward'):
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    def get_initial_features(self):
        return self.r_state_init
    # def reward(self, ob, ac, c, h):
    #     sess = tf.get_default_session()
    #     return sess.run([self.rew_norm, self.d, self.r_state_out],
    #                 {self.x: ob, self.action: ac, self.r_state_in[0]: c, self.r_state_in[1]: h})
    #
    def reward(self, ob, *args):
        c,h = args[0],args[1]
        sess = tf.get_default_session()
        return sess.run([self.rew_norm, self.d, self.r_state_out],
                        {self.x: ob, self.r_state_in[0]: c, self.r_state_in[1]: h})

class CONVDiscriminator(object):
    def __init__(self, ob_space, ac_space,do = 0.7,mem_size = 4,reward_form = 'action',failure= False):
        with tf.variable_scope('reward'):
            self.mem_size = mem_size # memory size in the input
            self.do =do
            self.type = 'conv'
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space[:-1]) + [self.mem_size])
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            self.conv_out = flatten(x)
            x = tf.nn.dropout(self.conv_out,self.keep_prob)
            self.action = tf.placeholder(tf.float32, [None] + list([self.ac_space]))
            if reward_form !='action':
                x = tf.concat([x,self.action],1)
            x = tf.nn.elu(linear(x,256,"fc1",normalized_columns_initializer(0.01)))

            if reward_form == 'action':
                self.r_logits = linear(x, self.ac_space,"actions", normalized_columns_initializer(0.01))
                self.max = tf.arg_max(self.action,1)
            else:
                if not failure:
                    self.r_logits = linear(x, 2, "actions", normalized_columns_initializer(0.01))
                else:
                    self.r_logits = linear(x, 3, "actions", normalized_columns_initializer(0.01))

            eps = tf.constant(1e-5)
            self.d = tf.nn.softmax(self.r_logits)
            self.rew =self.d #-tf.log(1-self.d + eps)

            self.rew_norm = self.rew #+1.78 #+ 1.8  # (-tf.log(eps))
            if reward_form=='action':
                self.rew_out = tf.reduce_sum(self.rew_norm*self.action,reduction_indices=[1])
            elif reward_form =='binary_prob':
                if failure:
                    self.rew_out = self.rew_norm[:, 0] - self.rew_norm[:, 2]
                else:
                    self.rew_out = self.rew_norm[:,0] - 0.5
            elif reward_form =='binary_logprob':
                if failure:
                    self.rew_out = tf.log(self.rew_norm[:, 0]+eps) - tf.log(self.rew_norm[:, 2]+eps)
                else:
                    self.rew_out = tf.log(self.rew_norm[:,0]+eps)/-tf.log(eps)
            elif reward_form =='binary_nlogprob':
                if failure:
                    self.rew_out = -tf.log(1- self.rew_norm[:, 0]+eps) + tf.log(1-self.rew_norm[:, 2]+eps)
                else:
                    self.rew_out = tf.log(1-self.rew_norm[:,0]+eps)/-tf.log(eps)
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def reward(self, ob,*args):
        ac = args[0]
        return self._reward(ob,ac)
    def _reward(self, ob,ac):
        sess = tf.get_default_session()
        # CAlling this function is not training so keep probability is 0
        return sess.run([self.rew_out, self.d],{self.x: ob,self.keep_prob:1.0,self.action :ac})


class CONVPolicy(object):
    def __init__(self, ob_space, ac_space,do = 0.7,mem_size = 4):
        with tf.variable_scope('policy'):
            self.mem_size = mem_size # memory size in the input
            self.do =do
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.ob_space = ob_space
            self.type = 'conv'
            self.ac_space = ac_space
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space[:-1]) + [self.mem_size])
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            self.conv_out = flatten(x)
            x = tf.nn.dropout(self.conv_out,self.keep_prob)
            x = tf.nn.elu(linear(x,256,"fc1",normalized_columns_initializer(0.01)))
            self.p_logits = linear(x, self.ac_space, "action", normalized_columns_initializer(0.01))
            self.probs = tf.nn.softmax(self.p_logits, name='action_probs')
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.sample = categorical_sample(self.p_logits, self.ac_space)[0, :]
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    def act(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: ob,self.keep_prob:1.0})
    def get_probs(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run([self.vf, self.probs],
                        {self.x: ob,self.keep_prob:1.0})
    def value(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: ob,self.keep_prob:1.0})[0]


class CONVImitator(object):
    def __init__(self, ob_space, ac_space,do = 0.3,mem_size = 4):
        self.mem_size = mem_size # memory size in the input
        self.do =do
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.ob_space = ob_space
        self.type = 'conv'
        self.ac_space = ac_space
        with tf.variable_scope('shared'):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space[:-1]) + [self.mem_size])
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            self.conv_out = flatten(x)
        with tf.variable_scope('policy'):
            x = tf.nn.dropout(self.conv_out,self.keep_prob)
            x = tf.nn.elu(linear(x,256,"fc1",normalized_columns_initializer(0.01)))

            self.p_logits = linear(x, self.ac_space, "action", normalized_columns_initializer(0.01))
            self.probs = tf.nn.softmax(self.p_logits, name='action_probs')
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.sample = categorical_sample(self.p_logits, self.ac_space)[0, :]
        with tf.variable_scope('reward'):
            y = tf.nn.dropout(self.conv_out, self.keep_prob)
            y = tf.nn.elu(linear(y, 256, "fc1", normalized_columns_initializer(0.01)))
            self.r_logits = linear(y, self.ac_space,"actions", normalized_columns_initializer(0.01))
            eps = tf.constant(1e-15)
            self.d = tf.nn.softmax(self.r_logits)
            self.rew = tf.log(self.d + eps)
            self.rew_norm = self.rew +1.78 #+ 1.8  # (-tf.log(eps))
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    def act(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: ob,self.keep_prob:1.0})
    def get_probs(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run([self.vf, self.probs],
                        {self.x: ob,self.keep_prob:1.0})
    def value(self, ob, *args):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: ob,self.keep_prob:1.0})[0]
