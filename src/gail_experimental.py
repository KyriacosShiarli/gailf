from __future__ import print_function
import tensorflow as tf
from model import LSTMImitator,LSTMDiscriminator,LSTMPolicy,CONVDiscriminator
import six.moves.queue as queue
from collections import deque
import distutils.version
from helpers import process_rollout, process_irl_rollout,PartialRollout,process_conv_rollout
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
from runner import RunnerThread
import pickle
import pdb
from demonstration_manager import DemonstrationManager
import numpy as np

class A3C_gail(object):
    def __init__(self, env, task, visualise,data_path, shared_network = False):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.env = env
        self.task = task
        with open(data_path + ".pkl", 'r+') as handle:
            data = pickle.load(handle) # Data is in the form of a demonstration manager defined in another class.
        self.data_manager= DemonstrationManager("./dummy_path")
        self.data_manager.trajectories = data
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        self.context_size = 0 # Context size for reward function updates
        self.num_local_steps = 100 ##local steps before policy update.
        self.rollout_history = deque([],maxlen=2000)
        self.pretrain = True
        self.plr_init = 5e-7
        self.plr_max = 4e-5
        self.rlr = 1e-4

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if shared_network:
                    self.network =self.r_network = LSTMImitator(env.observation_space.shape, env.action_space.n)
                else:
                    self.network =LSTMPolicy(env.observation_space.shape, env.action_space.n)
                    self.r_network = CONVDiscriminator(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if shared_network:
                    self.local_network = self.reward_f = pi = LSTMImitator(env.observation_space.shape, env.action_space.n)
                else:
                    self.local_network= pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                    self.reward_f = CONVDiscriminator(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            ####################### COMMON OPS ###################################333
            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            self.plr = tf.placeholder(tf.float32, shape=[])

            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            # copy weights from the parameter server to the local model
            self.p_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            ####################### Policy ops ######################################
            log_prob_tf = tf.nn.log_softmax(pi.p_logits)
            prob_tf = tf.nn.softmax(pi.p_logits)
            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            self.pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
            # loss of value function
            vf_loss = tf.reduce_mean(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_mean(prob_tf * log_prob_tf)
            bs = tf.to_float(tf.shape(pi.x)[0])
            self.p_loss = self.pi_loss + 1.* vf_loss - entropy * 0.01
            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, self.num_local_steps, visualise, reward_f=self.reward_f,shared=shared_network)
            p_grads = tf.gradients(self.p_loss, pi.var_list)
            p_grads_clipped, _ = tf.clip_by_global_norm(p_grads, 40.0)
            p_grads_and_vars = list(zip(p_grads_clipped, self.network.var_list))
            # each worker has a different set of adam optimizer parameters
            p_opt = tf.train.AdamOptimizer(self.plr)
            self.p_train_op = tf.group(p_opt.apply_gradients(p_grads_and_vars), inc_step)

            ################################ Reward Ops ####################################################
            self.label = tf.placeholder(tf.float32, [None, env.action_space.n], name="y")
            self.r_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.reward_f.var_list, self.r_network.var_list)])
            self.sync = [self.p_sync,self.r_sync]

            self.r_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.reward_f.r_logits, labels=self.label,
                                                                name="cent_loss")
            self.r_loss_trunc = tf.reduce_mean(self.r_loss[self.context_size:])
            r_grads = tf.gradients(self.r_loss_trunc, self.reward_f.var_list)
            r_grads_clipped, _ = tf.clip_by_global_norm(r_grads, 40.0)
            r_grads_and_vars = list(zip(r_grads_clipped, self.r_network.var_list))
            r_opt = tf.train.AdamOptimizer(self.rlr)
            self.r_train_op = r_opt.apply_gradients(r_grads_and_vars)

            bs = 1
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
                #tf.summary.scalar("discriminator/loss", self.r_loss_trunc / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(p_grads))
                #tf.summary.scalar("discriminator/grad_global_norm", tf.global_norm(r_grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.summary.scalar("discriminator/var_global_norm", tf.global_norm(self.reward_f.var_list))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", self.pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(p_grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.scalar_summary("discriminator/loss", self.r_loss)
                #tf.scalar_summary("discriminator/", vf_loss / bs)
                tf.scalar_summary("discriminator/grad_global_norm", tf.global_norm(r_grads))
                self.summary_op = tf.merge_all_summaries()
            self.summary_writer = None
            self.local_steps = 0

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

        inject_prob = 0.0
        inject = np.random.binomial(1,inject_prob)

        sess.run(self.p_sync)  # copy weights from shared to local

        if inject:
            rollout = self.inject_demonstration()
            print("injectin demonstration")
        else:
            rollout = self.pull_batch_from_queue()

        batch = process_rollout(rollout, gamma=0.95, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.p_train_op, self.global_step]
        else:
            fetches = [self.p_train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.p_state_in[0]: batch.features[0],
            self.local_network.p_state_in[1]: batch.features[1],
            self.plr: self.plr_init
        }
        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        if self.plr_init<self.plr_max:
            self.plr_init*=1.0004
        print(self.plr_init)
        # Update the distctiminator
        sess = tf.get_default_session()
        if not inject:
            self.train_conv(sess,rollout)
        self.local_steps += 1
        return rollout

    def inject_demonstration(self):
        # get a dataset from the reward interface.
        # compute the rewards the return and the value for these quantities.
        # put the whole thing in a rollout along with an importance weight.
        size = 100
        context = 40
        data_rollout = self.data_manager.get(1,size+context)[0]

        batch = process_irl_rollout(data_rollout)
        policy_rollout = PartialRollout()

        rewards,probs,r_features = self.reward_f.reward(batch.si,batch.a,np.zeros((1,256)),np.zeros((1,256)))
        #rewards = rewards[context:,0] #-rewards[context:,1]
        rewards = rewards[range(context,size+context),np.argmax(batch.a[context:,:],axis=1)]

        print("Mean injected reward",np.mean(rewards))
        fetched = self.local_network.get_probs(batch.si[:context],np.zeros((1,256)),np.zeros((1,256)))
        value, probs,features = fetched[0],fetched[1],fetched[2:]
        fetched = self.local_network.get_probs(batch.si[context:],features[0],features[1])
        value, probs, features_unused = fetched[0], fetched[1], fetched[2:]
        taken_action_probs = np.array(probs)[range(size),np.argmax(batch.a[context:,:],axis=1)]
        importance_ratios = list(taken_action_probs)
        print(importance_ratios)



        policy_rollout.rewards = list(rewards)
        policy_rollout.states = data_rollout.states[context:]
        policy_rollout.actions = data_rollout.actions[context:]
        policy_rollout.features = [features]
        policy_rollout.r_features = [r_features]
        policy_rollout.values = list(value)
        policy_rollout.r = value[-1]
        policy_rollout.importance_ratios = importance_ratios
        return policy_rollout


    def train_reward(self, sess, rollout):
        # the rollout here is the rollout used from the policy gradient method and passed on to the discriminator.
        # This training function is still quite basic and does not really do any batching appart from taking a full
        # trajectory. It would be nice to implement a batch version of the discriminator however. Shouldnt take much time.

        self.rollout_history.append(rollout)
        #if len(self.rollout_history)<2:

        idx = np.random.randint(0,len(self.rollout_history))
        #idx = 0

        r_rollout = self.rollout_history[idx]

        label_epsilon = 0.01
        sess.run(self.r_sync)
        batch_a = process_irl_rollout(r_rollout)

        data_e = self.data_manager.get(number=1., length=batch_a.si.shape[0])
        batch_e = process_irl_rollout(data_e[0])  # TODO: FOR NOW THIS ONLY TAKES ONE ROLLOUT
        if self.pretrain is True:
            train_repeat = 1
        else:
            train_repeat = 1
        for _ in range(train_repeat):



            #y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)
            #y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon]], dtype=np.float32), batch_e.si.shape[0], axis=0)

            n_actions = batch_a.a.shape[1]
            eps =0 #label_epsilon/(n_actions-1)
            y_a = np.ones((batch_a.si.shape[0],n_actions))/(float(n_actions))
            #y_a[range(batch_a.si.shape[0]),np.argmax(batch_a.a,axis = 1)] = 0
            #y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)

            #y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)
            #y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon]], dtype=np.float32), batch_e.si.shape[0], axis=0)

            y_e = eps*np.ones((batch_e.si.shape[0],n_actions))
            y_e[range(batch_e.si.shape[0]),np.argmax(batch_e.a,axis = 1)] = 1#-label_epsilon



            feed_dict_a = {
                self.reward_f.x: batch_a.si,
                #self.reward_f.action: batch_a.a,
                self.label: y_a,
                self.reward_f.r_state_in[0]: np.zeros(batch_a.features[0].shape),
                self.reward_f.r_state_in[1]: np.zeros(batch_a.features[1].shape),
            }

            feed_dict_e = {
                self.reward_f.x: batch_e.si,
                #self.reward_f.action: batch_e.a,
                self.label: y_e,
                self.reward_f.r_state_in[0]: np.zeros(batch_a.features[0].shape),
                self.reward_f.r_state_in[1]: np.zeros(batch_a.features[1].shape),
            }
            fetches = [self.r_train_op, self.r_loss_trunc,self.r_loss]
            _, loss_trunc_a,loss = sess.run(fetches, feed_dict=feed_dict_a)
            _, loss_trunc_e,loss_e = sess.run(fetches, feed_dict=feed_dict_e)
            summary = tf.Summary()
            summary.value.add(tag="discriminator/loss", simple_value=np.mean([loss_trunc_a,loss_trunc_e]))
            self.summary_writer.add_summary(summary,self.local_steps)
            self.summary_writer.flush()
            print("LOSES",loss_trunc_a,loss_trunc_e)
            #print("first_losses")
            #print("last_losses")
            sess.run(self.r_sync)
        self.pretrain = False


    def train_conv(self, sess, rollout):
        # the rollout here is the rollout used from the policy gradient method and passed on to the discriminator.
        # This training function is still quite basic and does not really do any batching appart from taking a full
        # trajectory. It would be nice to implement a batch version of the discriminator however. Shouldnt take much time.

        if len(rollout.actions)>10:
            self.rollout_history.append(rollout)
        # if len(self.rollout_history)<2:

        idx = np.random.randint(0, len(self.rollout_history))
        # idx = 0

        r_rollout = self.rollout_history[idx]

        label_epsilon = 0.01
        sess.run(self.r_sync)
        batch_a = process_conv_rollout(r_rollout,mem_size=4)

        data_e = self.data_manager.get(number=1., length=batch_a.si.shape[0])
        batch_e = process_conv_rollout(data_e[0],mem_size=4)  # TODO: FOR NOW THIS ONLY TAKES ONE ROLLOUT
        if self.pretrain is True:
            train_repeat= 200
        else:
            train_repeat = 1
        for _ in range(train_repeat):
            # y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)
            # y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon]], dtype=np.float32), batch_e.si.shape[0], axis=0)

            n_actions = batch_a.a.shape[1]
            eps = 0  # label_epsilon/(n_actions-1)
            y_a = np.ones((batch_a.si.shape[0], n_actions)) / (float(n_actions))
            # y_a[range(batch_a.si.shape[0]),np.argmax(batch_a.a,axis = 1)] = 0
            # y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)

            # y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32), batch_a.si.shape[0], axis=0)
            # y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon]], dtype=np.float32), batch_e.si.shape[0], axis=0)

            y_e = eps * np.ones((batch_e.si.shape[0], n_actions))
            y_e[range(batch_e.si.shape[0]), np.argmax(batch_e.a, axis=1)] = 1  # -label_epsilon

            feed_dict_a = {
                self.reward_f.x: batch_a.si,
                # self.reward_f.action: batch_a.a,
                self.reward_f.keep_prob: self.reward_f.do,
                self.label: y_a,
            }

            feed_dict_e = {
                self.reward_f.x: batch_e.si,
                # self.reward_f.action: batch_e.a,
                self.reward_f.keep_prob : self.reward_f.do,
                self.label: y_e,
            }
            fetches = [self.r_train_op, self.r_loss_trunc, self.r_loss]
            _, loss_trunc_a, loss = sess.run(fetches, feed_dict=feed_dict_a)
            _, loss_trunc_e, loss_e = sess.run(fetches, feed_dict=feed_dict_e)
            summary = tf.Summary()
            summary.value.add(tag="discriminator/loss", simple_value=np.mean([loss_trunc_a, loss_trunc_e]))
            self.summary_writer.add_summary(summary, self.local_steps)
            self.summary_writer.flush()
            print("LOSES", loss_trunc_a, loss_trunc_e)
            # print("first_losses")
            # print("last_losses")
            sess.run(self.r_sync)
        self.pretrain = False


