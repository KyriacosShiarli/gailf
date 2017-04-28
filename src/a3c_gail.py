from __future__ import print_function
import tensorflow as tf
from model import LSTMPolicy, LSTMDiscriminator
import six.moves.queue as queue
import distutils.version
from helpers import process_rollout, process_irl_rollout,PartialRollout
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
from runner import RunnerThread
import pickle
import pdb
from demonstration_manager import DemonstrationManager
import numpy as np


class RewardInterface(object):
    def __init__(self, env, task, data_path,reward_f = None,context_size = 10):
        # Reward interface for IRL in A3C. This is in essence a discriminator along with the data used for learning.
        # The reward interface provides a reward function instead of the environment. This reward function is learned
        # Iteratively using principles from IRL
        self.env = env
        self.task = task
        self.context_size = context_size
        # load the data pickle file
        with open(data_path + ".pkl", 'r+') as handle:
            data = pickle.load(handle) # Data is in the form of a demonstration manager defined in another class.

        self.data_manager= DemonstrationManager("./dummy_path")
        self.data_manager.trajectories = data
        if reward_f is None:
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
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=reward_f.r_logits, labels=self.label,
                                                                    name="cent_loss")

                self.loss_trunc = tf.reduce_mean(self.loss[self.context_size:])
                #

                #tf.summary.scalar("discriminator/loss", self.loss_trunc)
                #tf.summary.scalar("discriminator/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("discriminator/var_global_norm", tf.global_norm(self.reward_f.var_list))

                grads = tf.gradients(self.loss_trunc, reward_f.var_list)

                grads, _ = tf.clip_by_global_norm(grads, 40.0)

                # copy weights from the parameter server to the local model
                self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(reward_f.var_list, self.discriminator.var_list)])

                grads_and_vars = list(zip(grads, self.discriminator.var_list))
                # inc_step = self.global_step.assign_add(tf.shape(reward_f.x)[0])

                # each worker has a different set of adam optimizer parameters
                opt = tf.train.AdamOptimizer(1e-6)
                self.train_op = opt.apply_gradients(grads_and_vars)
                self.summary_writer = None
                self.local_steps = 0



    def train(self,sess, rollout,summary_writer):
        # the rollout here is the rollout used from the policy gradient method and passed on to the discriminator.
        # This training function is still quite basic and does not really do any batching appart from taking a full
        # trajectory. It would be nice to implement a batch version of the discriminator however. Shouldnt take much time.

        label_epsilon = 0.1
        sess.run(self.sync)
        batch_a = process_irl_rollout(rollout)
        data_e = self.data_manager.get(number=1.,length =batch_a.si.shape[0])
        batch_e = process_irl_rollout(data_e[0]) # TODO: FOR NOW THIS ONLY TAKES ONE ROLLOUT

        y_a = np.repeat(np.array([[label_epsilon, 1.-label_epsilon]],dtype=np.float32), batch_a.si.shape[0], axis=0)
        y_e = np.repeat(np.array([[1.-label_epsilon, label_epsilon]], dtype=np.float32), batch_e.si.shape[0], axis=0)


        feed_dict_a = {
            self.reward_f.x: batch_a.si,
            self.reward_f.action: batch_a.a,
            self.label: y_a,
            self.reward_f.r_state_in[0]: np.zeros(batch_a.features[0].shape),
            self.reward_f.r_state_in[1]: np.zeros(batch_a.features[1].shape),
        }

        feed_dict_e = {
            self.reward_f.x: batch_e.si,
            self.reward_f.action: batch_e.a,
            self.label: y_e,
            self.reward_f.r_state_in[0]: np.zeros(batch_a.features[0].shape),
            self.reward_f.r_state_in[1]: np.zeros(batch_a.features[1].shape),
        }
        fetches = [self.train_op,self.loss_trunc]
        if self.local_steps<10000000:
            summary = tf.Summary()
            _,loss_trunc_e = sess.run(fetches, feed_dict=feed_dict_a)
            _, loss_trunc_a = sess.run(fetches, feed_dict=feed_dict_e)
            summary.value.add(tag="discriminator/loss", simple_value=np.mean([loss_trunc_a,loss_trunc_e]))
            summary_writer.add_summary(summary,self.local_steps)
            summary_writer.flush()
            self.local_steps += 1


class A3C_gail(object):
    def __init__(self, env, task, visualise,data_path):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.env = env
        self.task = task
        self.reward_iface = RewardInterface(env,task,data_path)
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

            log_prob_tf = tf.nn.log_softmax(pi.p_logits)
            prob_tf = tf.nn.softmax(pi.p_logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            self.pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_mean(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = self.pi_loss + 0.5 * vf_loss - entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20, visualise, reward_f=self.reward_iface.reward_f)
            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                #tf.summary.scalar("discriminator/loss", self.reward_iface.loss_trunc)
                #tf.scalar_summary("discriminator/", vf_loss / bs)
                #tf.summary.scalar("discriminator/grad_global_norm", tf.global_norm(self.reward_iface.grads))
                #tf.summary.scalar("discriminator/var_global_norm", tf.global_norm(self.reward_iface.reward_f.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))

                tf.scalar_summary("discriminator/loss", self.reward_iface.loss)
                #tf.scalar_summary("discriminator/", vf_loss / bs)
                tf.scalar_summary("discriminator/grad_global_norm", tf.global_norm(self.reward_iface.grads))
                tf.scalar_summary("discriminator/var_global_norm", tf.global_norm(self.reward_iface.reward_f.var_list))
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

        inject_prob = 0.8
        inject = np.random.binomial(1,inject_prob)

        sess.run(self.sync)  # copy weights from shared to local

        if inject:
            rollout = self.inject_demonstration()
            print("injectin demonstration")
        else:
            rollout = self.pull_batch_from_queue()

        batch = process_rollout(rollout, gamma=0.94, lambda_=1.0)

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
            self.local_network.p_state_in[0]: batch.features[0],
            self.local_network.p_state_in[1]: batch.features[1],
        }
        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        # Update the distctiminator
        sess = tf.get_default_session()
        if not inject:
            self.reward_iface.train(sess,rollout,self.summary_writer)
        self.local_steps += 1
        return rollout

    def inject_demonstration(self):
        # get a dataset from the reward interface.
        # compute the rewards the return and the value for these quantities.
        # put the whole thing in a rollout along with an importance weight.
        size = 100
        context = 20
        data_rollout = self.reward_iface.data_manager.get(1,size+context)[0]

        batch = process_irl_rollout(data_rollout)
        policy_rollout = PartialRollout()

        rewards,probs,r_features = self.reward_iface.reward_f.reward(batch.si,batch.a,np.zeros((1,256)),np.zeros((1,256)))
        #rewards = rewards[context:,0] #-rewards[context:,1] # this is if reward is binary.
        rewards = rewards[context:,np.argmax(batch.a[context:,:])]



        print("Mean injected reward",np.mean(rewards))
        fetched = self.local_network.get_probs(batch.si[:context],np.zeros((1,256)),np.zeros((1,256)))
        value, probs,features = fetched[0],fetched[1],fetched[2:]
        fetched = self.local_network.get_probs(batch.si[context:],features[0],features[1])
        value, probs, features_unused = fetched[0], fetched[1], fetched[2:]

        taken_action_probs = np.array(probs)[:,np.argmax(batch.a[context:,:],axis=1)]
        importance_ratios = list(taken_action_probs[0])


        policy_rollout.rewards = list(rewards)
        policy_rollout.states = data_rollout.states[context:]
        policy_rollout.actions = data_rollout.actions[context:]
        policy_rollout.features = [features]
        policy_rollout.r_features = [r_features]
        policy_rollout.values = list(value)
        policy_rollout.r = value[-1]
        policy_rollout.importance_ratios = importance_ratios
        return policy_rollout





