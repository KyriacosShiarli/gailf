from __future__ import print_function
import tensorflow as tf
from model import LSTMImitator,LSTMDiscriminator,LSTMPolicy,CONVDiscriminator,CONVPolicy
import six.moves.queue as queue
from collections import deque
import distutils.version
from helpers import process_rollout, process_irl_rollout,PartialRollout,process_conv_rollout
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
from runner import RunnerThread
import pickle
import yaml
import pdb
from demonstration_manager import DemonstrationManager
import numpy as np
#import matplotlib.pyplot as plt

class A3C_gail(object):
    def __init__(self, env, task, visualise,data_path, cfg = None):
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
        self.failure = bool(cfg['failure'] if cfg is not None else False)
        self.num_local_steps = cfg['local_steps'] if cfg is not None else 100 ##local steps before policy update.
        self.rollout_history = deque([],maxlen=cfg["rollout_history"] if cfg is not None else 20000)
        self.pretrain_steps =cfg["pretrain_steps"] if cfg is not None else 100
        self.pretrain = False if self.pretrain_steps < 2 else True
        self.plr_init = float(cfg["plr_init"] if cfg is not None else 4e-5)
        self.plr_max = float(cfg["plr_max"] if cfg is not None else 4e-5)
        self.plr_inc = cfg["plr_inc"] if cfg is not None else 1.0004
        self.rlr = float(cfg["rlr"] if cfg is not None else 1e-4)
        self.p_loss_weights = {"policy":0.8,"value":1.,"entropy":0.01}
        self.gamma = cfg["gamma"] if cfg is not None else 0.95
        self.grad_clip = cfg["grad_clip"] if cfg is not None else 40.
        self.lambda_ = cfg["lambda"] if cfg is not None else 0.9
        self.policy_type = cfg["policy_type"] if cfg is not None else 'lstm'
        self.reward_type = cfg["reward_type"] if cfg is not None else 'conv'
        self.shared = cfg["shared"] if cfg is not None else False
        self.reward_form = cfg['reward_form'] if cfg is not None else 'action'
        self.dtau = cfg['dtau'] if cfg is not None else 0.01

        models = {"policy_lstm":LSTMPolicy,"policy_conv":CONVPolicy,
                  "reward_conv":CONVDiscriminator,"reward_lstm":LSTMDiscriminator,
                  "shared_conv": LSTMImitator, "shared_lstm": LSTMImitator}

        if self.failure is True:
            with open(data_path + "_failure.pkl", 'r+') as handle:
                f_data = pickle.load(handle) # Data is in the form of a demonstration manager defined in another class.
            self.data_manager_f = DemonstrationManager("./dummy_path")
            self.data_manager_f.trajectories = f_data

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if self.shared:
                    self.network =self.r_network = LSTMImitator(env.observation_space.shape, env.action_space.n)
                else:
                    self.network =models["policy_"+self.policy_type](env.observation_space.shape, env.action_space.n)
                    self.r_network = models["reward_"+self.reward_type](env.observation_space.shape, env.action_space.n,
                                                                        reward_form = self.reward_form,failure = self.failure)
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if self.shared:
                    self.local_network = self.reward_f = pi = LSTMImitator(env.observation_space.shape, env.action_space.n)
                else:
                    self.local_network= pi = models["policy_"+self.policy_type](env.observation_space.shape, env.action_space.n)
                    self.reward_f = models["reward_"+self.reward_type](env.observation_space.shape, env.action_space.n,
                                                                       reward_form = self.reward_form,failure = self.failure)
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
            self.p_loss = self.p_loss_weights["policy"]*self.pi_loss + \
                            self.p_loss_weights["value"]*vf_loss - self.p_loss_weights["entropy"]*entropy
            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, self.num_local_steps, visualise, reward_f=self.reward_f,shared=self.shared)
            p_grads = tf.gradients(self.p_loss, pi.var_list)
            p_grads_clipped, _ = tf.clip_by_global_norm(p_grads, self.grad_clip)
            p_grads_and_vars = list(zip(p_grads_clipped, self.network.var_list))
            # each worker has a different set of adam optimizer parameters
            p_opt = tf.train.AdamOptimizer(self.plr)
            self.p_train_op = tf.group(p_opt.apply_gradients(p_grads_and_vars), inc_step)

            v_grads = tf.gradients(vf_loss, pi.var_list)
            v_grads_clipped, _ = tf.clip_by_global_norm(v_grads, self.grad_clip)
            v_grads_and_vars = list(zip(v_grads_clipped, self.network.var_list))
            # each worker has a different set of adam optimizer parameters
            self.v_train_op = tf.group(p_opt.apply_gradients(v_grads_and_vars), inc_step)



            ################################ Reward Ops ####################################################
            if self.reward_form == 'action':
                self.label = tf.placeholder(tf.float32, [None, env.action_space.n], name="y")
            elif self.failure:
                self.label = tf.placeholder(tf.float32, [None, 3], name="y")
            else:
                self.label = tf.placeholder(tf.float32, [None, 2], name="y")

            self.r_sync_soft = tf.group(*[v1.assign((1-self.dtau)*v1 + self.dtau*v2) for v1, v2 in zip(self.reward_f.var_list, self.r_network.var_list)])
            self.r_sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.reward_f.var_list, self.r_network.var_list)])
            self.sync = [self.p_sync,self.r_sync]

            self.r_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.reward_f.r_logits, labels=self.label,
                                                                name="cent_loss")
            self.r_loss_trunc = tf.reduce_mean(self.r_loss[self.context_size:])
            r_grads = tf.gradients(self.r_loss_trunc, self.reward_f.var_list)
            r_grads_clipped, _ = tf.clip_by_global_norm(r_grads, self.grad_clip)
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
            rollout = self.inject_conv_demonstration()
            print("injectin demonstration")
        else:
            rollout = self.pull_batch_from_queue()

        batch = process_rollout(rollout, gamma=self.gamma, lambda_=self.lambda_,policy_type=self.policy_type)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.p_train_op, self.global_step]
        else:
            fetches = [self.p_train_op, self.global_step]
        compute = True
        if self.policy_type=='lstm':
            feed_dict = {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
                self.local_network.p_state_in[0]: batch.features[0],
                self.local_network.p_state_in[1]: batch.features[1],
                self.plr: self.plr_init
            }
        elif self.policy_type=='conv':

            compute = True if len(batch.si)>0 else False
            feed_dict = {
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
                self.plr: self.plr_init,
                self.local_network.keep_prob : self.local_network.do
            }
        if compute:
            fetched = sess.run(fetches, feed_dict=feed_dict)

            if should_compute_summary:
                self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
                self.summary_writer.flush()
            if self.plr_init<self.plr_max:
                self.plr_init*=self.plr_inc
            print(self.plr_init)
            # Update the distctiminator
            sess = tf.get_default_session()
            if not inject:
                self.train_reward(sess,rollout)
            self.local_steps += 1

    def train_reward(self, sess, rollout):
        # the rollout here is the rollout used from the policy gradient method and passed on to the discriminator.
        # This training function is still quite basic and does not really do any batching appart from taking a full
        # trajectory. It would be nice to implement a batch version of the discriminator however. Shouldnt take much time.
        if len(rollout.states)>10:
            self.rollout_history.append(rollout)
        #if len(self.rollout_history)<2:

        idx = np.random.randint(0,len(self.rollout_history))
        #idx = 0

        r_rollout = self.rollout_history[idx]

        label_epsilon = 0.01
        if self.local_steps == 0:
            sess.run(self.r_sync)
        else:
            sess.run(self.r_sync_soft)


        data_e = self.data_manager.get(number=1., length=len(r_rollout.states))

        if self.reward_type =='conv':
            batch_a = process_conv_rollout(r_rollout,mem_size = self.reward_f.mem_size)
            batch_e = process_conv_rollout(data_e[0],mem_size=self.reward_f.mem_size)  # TODO: FOR NOW THIS ONLY TAKES ONE ROLLOUT
            if self.failure:
                data_f = self.data_manager_f.get(number=1., length=len(r_rollout.states))
                batch_f = process_conv_rollout(data_f[0],
                                               mem_size=self.reward_f.mem_size)
        elif self.reward_type =='lstm':
            batch_a = process_irl_rollout(r_rollout)
            batch_e = process_irl_rollout(data_e[0])  # TODO: FOR NOW THIS ONLY TAKES ONE ROLLOUT

        if self.pretrain is True:
            train_repeat = self.pretrain_steps
        else:
            train_repeat = 1
        for _ in range(train_repeat):
            n_actions = batch_a.a.shape[1]
            eps =0

            if self.reward_form == 'action':
                y_a = np.ones((batch_a.si.shape[0],n_actions))/(float(n_actions-1))
                y_a[range(batch_a.si.shape[0]),np.argmax(batch_a.a,axis = 1)] = 0
                y_e = eps*np.ones((batch_e.si.shape[0],n_actions))
                y_e[range(batch_e.si.shape[0]),np.argmax(batch_e.a,axis = 1)] = 1#-label_epsilon
            else:
                if self.failure:
                    y_a = np.repeat(np.array([[label_epsilon/2, 1. - label_epsilon,label_epsilon/2]], dtype=np.float32),
                                batch_a.si.shape[0], axis=0)
                    y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon/2,label_epsilon/2]], dtype=np.float32),
                                batch_e.si.shape[0], axis=0)
                    y_f = np.repeat(np.array([[label_epsilon/2, label_epsilon/2,1 - label_epsilon]], dtype=np.float32),
                                batch_a.si.shape[0], axis=0)

                    s = np.vstack([batch_a.si,batch_e.si,batch_f.si])
                    a = np.vstack([batch_a.a, batch_e.a,batch_f.a])
                    y = np.vstack([y_a,y_e,y_f])
                else:
                    y_a = np.repeat(np.array([[label_epsilon, 1. - label_epsilon]], dtype=np.float32),
                                    batch_a.si.shape[0], axis=0)
                    y_e = np.repeat(np.array([[1. - label_epsilon, label_epsilon]], dtype=np.float32),
                                    batch_e.si.shape[0], axis=0)
                    s = np.vstack([batch_a.si,batch_e.si])
                    a = np.vstack([batch_a.a, batch_e.a])
                    y = np.vstack([y_a,y_e])

            if self.reward_type =='conv':
                feed_dict = {
                    self.reward_f.x: s,
                    self.reward_f.action: a,
                    self.reward_f.keep_prob: self.reward_f.do,
                    self.label: y,
                }

            elif self.reward_type =='lstm':
                feed_dict = {
                    self.reward_f.x: s,
                    #self.reward_f.action: batch_a.a,
                    self.label: y,
                    self.reward_f.r_state_in[0]: np.zeros(batch_a.features[0].shape),
                    self.reward_f.r_state_in[1]: np.zeros(batch_a.features[1].shape),
                }

            fetches = [self.r_train_op, self.r_loss_trunc,self.r_loss]
            _, loss_trunc,loss = sess.run(fetches, feed_dict=feed_dict)

            summary = tf.Summary()
            summary.value.add(tag="discriminator/loss", simple_value=np.mean([loss_trunc]))
            self.summary_writer.add_summary(summary,self.local_steps)
            self.summary_writer.flush()
            print("LOSES",loss_trunc)
        self.pretrain = False





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
        # Simply update the value function with one step bootstraps.



        # taken_action_probs = np.array(probs)[range(size),np.argmax(batch.a[context:,:],axis=1)]
        # importance_ratios = list(taken_action_probs)
        # print(importance_ratios)
        #
        # policy_rollout.rewards = list(rewards)
        # policy_rollout.states = data_rollout.states[context:]
        # policy_rollout.actions = data_rollout.actions[context:]
        # policy_rollout.features = [features]
        # policy_rollout.r_features = [r_features]
        # policy_rollout.values = list(value)
        # policy_rollout.r = value[-1]
        # policy_rollout.importance_ratios = importance_ratios
        # return policy_rollout



    def inject_conv_demonstration(self,sess):
        # get a dataset from the reward interface.
        # compute the rewards the return and the value for these quantities.
        # put the whole thing in a rollout along with an importance weight.
        size = self.num_local_steps
        context = 50
        mem_size = 4
        data_rollout = self.data_manager.get(1,size+context)[0]

        batch = process_conv_rollout(data_rollout)
        batch_p = process_irl_rollout(data_rollout)
        policy_rollout = PartialRollout()

        rewards,probs = self.reward_f.reward(batch.si)
        #rewards = rewards[context:,0] #-rewards[context:,1]
        rewards = rewards[range(context-mem_size,size+context-mem_size),np.argmax(batch.a[context-mem_size:,:],axis=1)]
        print("Mean injected reward",np.mean(rewards))

        fetched = self.local_network.get_probs(batch_p.si[:context],np.zeros((1,256)),np.zeros((1,256)))
        value, probs,features = fetched[0],fetched[1],fetched[2:]
        fetched = self.local_network.get_probs(batch_p.si[context:],features[0],features[1])
        value, probs, features_unused = fetched[0], fetched[1], fetched[2:]
        taken_action_probs = np.array(probs)[range(size),np.argmax(batch_p.a[context:,:],axis=1)]

        targets = self.gamma*value[1:] + rewards[:-1]

        feed_dict = {
            self.local_network.x: batch.si,
            self.r: batch.targets,
            self.local_network.p_state_in[0]: features[0],
            self.local_network.p_state_in[1]: features[1],
            self.plr: self.plr_init
        }
        fetches = [self.v_train_op]
        fetched = sess.run(fetches, feed_dict=feed_dict)

        # importance_ratios = list(taken_action_probs)
        # print(importance_ratios)
        # policy_rollout.rewards = list(rewards)
        # policy_rollout.states = data_rollout.states[context:]
        # policy_rollout.actions = data_rollout.actions[context:]
        # policy_rollout.features = [features]
        # policy_rollout.values = list(value)
        # policy_rollout.r = value[-1]
        # policy_rollout.importance_ratios = importance_ratios
        #return policy_rollout


