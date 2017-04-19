from __future__ import print_function
import tensorflow as tf
from model import LSTMPolicy, LSTMDiscriminator
import six.moves.queue as queue
import distutils.version
from helpers import process_rollout, process_irl_rollout

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
from runner import RunnerThread
import pickle
import pdb
import numpy as np


class RewardInterface(object):
    def __init__(self, env, task, data=None):
        # Reward interface for IRL in A3C. This is in essence a discriminator along with the data used for learning.
        # The reward interface provides a reward function instead of the environment. This reward function is learned
        # Iteratively using principles from IRL
        self.env = env
        self.task = task
        self.data = data  # Data is in the form of a demonstration manager defined in another class.
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("RIF"):  # rif stands for reward interface
                self.discriminator = LSTMDiscriminator(env.observation_space.shape, env.action_space.n)
                # self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                # trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("rif"):
                self.reward_f = reward_f = LSTMDiscriminator(env.observation_space.shape, env.action_space.n)
            self.label = tf.placeholder(tf.float32, [None, 2], name="y")
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=reward_f.logits, labels=self.label,
                                                                name="cent_loss")
            #
            grads = tf.gradients(self.loss, reward_f.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(reward_f.var_list, self.discriminator.var_list)])

            grads_and_vars = list(zip(grads, self.discriminator.var_list))
            # inc_step = self.global_step.assign_add(tf.shape(reward_f.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = opt.apply_gradients(grads_and_vars)
            self.summary_writer = None
            self.local_steps = 0
    def train(self,sess, rollout):
        batch_a = process_irl_rollout(rollout)
        # data_e = self.data.get()
        # batch_e = process_irl_rollout(data_e)

        y_a = np.repeat(np.array([[0., 1.]],dtype=np.float32), batch_a.si.shape[0], axis=0)

        feed_dict_a = {
            self.reward_f.x: batch_a.si,
            self.reward_f.action: batch_a.a,
            self.label: y_a,
            self.reward_f.state_in[0]: batch_a.features[0],
            self.reward_f.state_in[1]: batch_a.features[1],
        }

        fetches = [self.train_op]
        sess.run(fetches, feed_dict=feed_dict_a)
        pdb.set_trace()


class A3C(object):
    def __init__(self, env, task, visualise, reward_iface=None):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("policy"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)

                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            if reward_iface is None:
                self.runner = RunnerThread(env, pi, 20, visualise, reward_f=None)
            else:
                self.runner = RunnerThread(env, pi, 20, visualise, reward_f=reward_iface.reward_f)

            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0
            # self.demonstrations = DemonstrationManager("./data/dummies")

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server. In the gail case we will need to decide wether or not this will need to be saved to same data structure. Hopefully in the same 
rollout structure.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
        return rollout
        # if self.local_steps%10==0:
        #    self.demonstrations.save()
        #    self.demonstrations.clear()
