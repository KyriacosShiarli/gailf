from helpers import PartialRollout
import threading
import six.moves.queue as queue
import tensorflow as tf
import pdb
from collections import deque
from demonstration_manager import DemonstrationManager
import numpy as np

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps, visualise, reward_f = None,record = False,shared=False,enemy = False):
        threading.Thread.__init__(self)
        self.record = record
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.shared=shared
        self.same_colours = enemy

        self.summary_writer = None
        self.visualise = visualise
        self.reward_f = reward_f

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        if self.record:
            rollout_provider = recording_runner(self.env, self.policy, self.num_local_steps, self.summary_writer,
                                          self.visualise)
        else:
            #rollout_provider = conv_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise,reward_f=self.reward_f,shared=self.shared)
            rollout_provider = self.runner()
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


    def runner(self):

        """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """

    # ok so theres a bunch of options here. Theres a record mode. a convolution or lstm option for both policy and reward function
        # skip the recording part for now such that the two run similarly.
        ## Define here the configuration of the whole thing.

        external_reward = self.reward_f is not None
        shared = hasattr(self.policy,"shared")
        policy_type = self.policy.type
        reward_type = self.reward_f.type if external_reward else None

        last_state = self.env.reset()
        last_features = self.policy.get_initial_features() if self.policy.type =='lstm' else [None]

        if external_reward:
            if shared is False and reward_type=='lstm':
                r_features = self.reward_f.get_initial_features()
            elif shared is True and reward_type=='lstm':
                last_features,r_features = self.policy.get_initial_features()
            else:
                r_features =[None]

            if reward_type == 'conv':
                r_mem_size = self.reward_f.mem_size
                r_obs = np.zeros(self.reward_f.ob_space[:-1] + (r_mem_size,))
            else:
                r_obs = last_state
            irl_rewards = []

        if policy_type == 'conv':
            p_mem_size = self.policy.mem_size
            p_obs = np.zeros(self.policy.ob_space[:-1] + (p_mem_size,))
        else:
            p_obs = last_state


        length = 0
        rewards = 0

        while True:
            terminal_end = False
            rollout = PartialRollout()
            for _ in range(self.num_local_steps):

                if policy_type=='conv':
                    p_obs[:, :, :p_mem_size - 1] = p_obs[:, :, 1:p_mem_size]
                    p_obs[:, :, -1] = last_state[:, :, 0]
                elif policy_type=='lstm':
                    p_obs = last_state
                fetched = self.policy.act([p_obs], *last_features)
                action, value_, = fetched[0], fetched[1]
                features = fetched[2:] if policy_type =='lstm' else [None]

                # argmax to convert from one-hot
                state, reward, terminal, info = self.env.step(action.argmax())
                if self.same_colours:
                    wh = np.where(state > np.amin(state))
                    state[wh[0], wh[1]] = 0.6
                actual_reward = reward
                if self.visualise:
                    self.env.render()

                if external_reward:
                    # If there is an external reward function use that.
                    if reward_type == 'conv':
                        r_obs[:, :, :r_mem_size - 1] = r_obs[:, :, 1:r_mem_size]
                        r_obs[:, :, -1] = last_state[:, :, 0]
                    else:
                        r_obs = last_state

                    r_fetched = self.reward_f.reward([r_obs],[action*(1-self.same_colours)])
                    #reward = r_fetched[0][0,0] #-r_fetched[0][0,1] #if reward is binary class.
                    reward = r_fetched[0][0]
                    irl_rewards.append(reward)
                    r_features = r_fetched[2] if reward_type == 'lstm' else [None]
                    rollout.add(last_state, action, reward, value_, terminal, last_features,r_features)
                else:
                    rollout.add(last_state, action, reward, value_, terminal, last_features)

                # collect the experience

                length += 1
                rewards += actual_reward

                last_state = state
                last_features = features

                if info:
                    summary = tf.Summary()
                    for k, v in info.items():
                        summary.value.add(tag=k, simple_value=float(v))
                    if self.reward_f is not None:
                        summary.value.add(tag="global/discriminator_reward", simple_value=float(reward))
                        summary.value.add(tag="global/discriminator_reward_variance", simple_value=np.var(irl_rewards))
                    self.summary_writer.add_summary(summary, self.policy.global_step.eval())
                    self.summary_writer.flush()

                timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                if terminal or length >= timestep_limit:
                    terminal_end = True
                    if length >= timestep_limit or not self.env.metadata.get('semantics.autoreset'):
                        last_state = self.env.reset()
                    last_features = self.policy.get_initial_features() if self.policy.type == 'lstm' else [None]
                    if policy_type == 'conv':
                        p_obs = np.zeros(self.policy.ob_space[:-1] + (p_mem_size,))
                    if external_reward:
                        if shared is False and reward_type == 'lstm':
                            r_features = self.reward_f.get_initial_features()
                        elif shared is True and reward_type == 'lstm':
                            last_features, r_features = self.policy.get_initial_features()
                        else:
                            r_features = [None]
                        if reward_type == 'conv':
                            r_mem_size = self.reward_f.mem_size
                            r_obs = np.zeros(self.reward_f.ob_space[:-1] + (r_mem_size,))
                        else:
                            r_obs = last_state
                    print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                    #with tf.device(tf.train.replica_device_setter(1)):
                    if external_reward:
                        print("IRL REWARDS: {}. Average: {}".format(np.sum(irl_rewards),np.mean(irl_rewards)))
                        if len(irl_rewards) > 0:
                            print("Max reward {}".format(np.amax(irl_rewards)))
                        irl_rewards=[]

                    length = 0
                    rewards = 0

                    break

            if not terminal_end:
                rollout.r = self.policy.value([p_obs], *last_features)

            # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
            yield rollout


def recording_runner(env, policy, num_local_steps, summary_writer, render):
    """
    A thread runner that records the best and worse trajectories of the thread
    """
    recorder = DemonstrationManager("../data/pong/demonstrations")
    recorder_failure = DemonstrationManager("../data/pong/demonstrations_failure")
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    demonstration = PartialRollout()
    while True:
        terminal_end = False
        rollout = PartialRollout()
        for _ in range(num_local_steps):
            fetched = policy.act([last_state], *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]


            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())
            if render:
                env.render()

            rollout.add(last_state, action, reward, value_, terminal, last_features)

            demonstration.add(last_state, action, reward, value_, terminal, last_features)

            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                recorder.append_to_best(demonstration)
                recorder_failure.append_to_worst(demonstration)
                demonstration = PartialRollout()
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value([last_state], *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout