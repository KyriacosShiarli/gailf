import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import pdb
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')
import pickle
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

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

inpt = x = tf.placeholder(tf.float32, [None,None] + list([42,42,1]))
print "SHAPE", inpt.get_shape()

#timesteps = tf.placeholder(tf.float32, shape = (1))
#batch = tf.placeholder(tf.float32, shape=(1))
#ou = timesteps*batch
x = tf.reshape(x,[-1,42,42,1])

for i in range(4):
    x = tf.nn.elu(conv2d(x, 5, "l{}".format(i + 1), [3, 3], [2, 2]))
# introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
dms = [tf.shape(inpt)[0],tf.shape(inpt)[1],-1]
#print dms
ya = tf.reshape(x,dms)
x = tf.expand_dims(flatten(x), [0])

size = 256
if use_tf100_api:
    lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
else:
    lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

state_size = lstm.state_size
step_size = tf.shape(x)[:1]

c_init = np.zeros((1, lstm.state_size.c), np.float32)
h_init = np.zeros((1, lstm.state_size.h), np.float32)
state_init = [c_init, h_init]
c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
state_in = [c_in, h_in]

if use_tf100_api:
    state_in = rnn.LSTMStateTuple(c_in, h_in)
else:
    state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
    lstm, x, initial_state=state_in, sequence_length=step_size,
    time_major=False)

init_all_op = tf.initialize_all_variables()

bs = 10
ts = 100
w = 42; h =42
channels = 1

with tf.Session() as sess:
    sess.run(init_all_op)
    fake_inpt = np.random.uniform(0,1,size = [bs,ts,w,h,1])

    out = sess.run([ya,x],{inpt:fake_inpt,c_in:c_init,h_in:h_init})
    print out[1].shape
    print out[0].shape
with open("./data/test.pkl",'wb') as handle:
    pickle.dump(out,handle)