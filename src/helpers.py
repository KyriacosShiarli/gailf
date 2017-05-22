import scipy.signal
import numpy as np
from collections import namedtuple
import pdb
import os
Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])
IRL_Batch = namedtuple("IRL_Batch", ["si", "a",'features'])

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def mkdir(dir):
    if os.path.exists(dir):
        return 0
    else:
        os.mkdir(dir)



def process_rollout(rollout, gamma, lambda_=1.0,policy_type = 'lstm',mem_size=4,n_step = True):
    """
given a rollout, compute its returns and the advantage
"""
    if policy_type =='lstm':
        batch_si = np.asarray(rollout.states)
        batch_a = np.asarray(rollout.actions)
        features = rollout.features[0]

    elif policy_type=='conv':
        batch_si = []
        for i in range(mem_size, len(rollout.states)):
            input = rollout.states[i - mem_size:i]
            batch_si.append(np.squeeze(np.array(input).swapaxes(0, 3)))

        batch_si = np.asarray(batch_si)
        batch_a = np.asarray(rollout.actions[mem_size:])
        features = None
        rollout.rewards = rollout.rewards[mem_size:]
        rollout.values = rollout.values[mem_size:]
    #pdb.set_trace()
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    if n_step:
        batch_r = discount(rewards_plus_v, gamma)[:-1]
    else:
        batch_r = rollout.rewards +gamma*rollout.values[1:]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)
    if len(rollout.importance_ratios) !=0:
        batch_adv *= np.array(rollout.importance_ratios)

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

def process_irl_rollout(rollout):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    #rewards = np.asarray(rollout.rewards)
    #vpred_t = np.asarray(rollout.values + [rollout.r])
    if len(rollout.r_features)!=0:
        features = rollout.r_features[0] # TODO: FIGURE THIS OUT MAYBE ASK SOMEBODY. I will be setting these to 0
    else:
        features = None
    return IRL_Batch(batch_si, batch_a,features)

def process_conv_rollout(rollout,mem_size=4):
    # process the rollout such that the right amount of memory is used.
    # you basically need a batch of [batch_size,width,height,mem_size] to go into the network.
    batch_si = []
    for i in range(mem_size,len(rollout.states)):
        input = rollout.states[i-mem_size:i]
        batch_si.append(np.squeeze(np.array(input).swapaxes(0,3)))

    batch_si = np.asarray(batch_si)
    batch_a = np.asarray(rollout.actions[mem_size:])
    return IRL_Batch(batch_si, batch_a,None)



class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.r_features = []
        self.importance_ratios = []

    def add(self, state, action, reward, value, terminal, features,r_features=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.r_features += [r_features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
