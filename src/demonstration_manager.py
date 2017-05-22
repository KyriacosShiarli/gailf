from helpers import PartialRollout
import pickle
import numpy as np
import pdb

class DemonstrationManager(object):
    def __init__(self,base_filename,max_size = 10,demonstration_length=None):
        #Here I need to define wether on not this will be symmetric or not. In theory I can use a non symetric demonstrations
        # if I will be updating the discriminator with a batch size of 1. and I can have batched if this is not the case.
        # It looks also wise that if I will follow the A3C convention I only use one trajectory at a time. if I use DQN things would change.
        # Functions of the demonstration manager:
        # Collect demonstrations and make sure that they are of a certain size if a certain flag is turned on. If not arbitrary
        # demonstration shapes can be saved
        # Load demonstrations from a file. so they can be used as data. Provide shuffle and batch methods for that data.
        # DEmonstrations should be saved with a date.
        # Demonstration manager should have modes. Recording, learning and any other mode thats usefull
        # looking at this its probably a good idea to have this split in two. A demonstration Recorder and a demonstartion loader.
        # The loader should also have the choice of splitting the potentially uneven demonstrations into equan chunks to be used for DQN.
        self.base_filename = base_filename
        self.max_size = max_size
        self.trajectories = [] # This will be a list of the type Partial Rollout
        self.counter = 0
    def append(self,rollout):
        self.trajectories.append(rollout)
    def get(self,number=1,length=None):
        # get random number of demonstrations of size length each.
        # Byt default returns a random demonstration from the rollout in full length.
        idx = np.array(np.random.randint(0,high=len(self.trajectories),size = (int(number))))
        out = []
        for i in idx:
            if i < len(self.trajectories):
                traj = self.trajectories[i]
                out_traj = PartialRollout()
                if length!=None and length<len(traj.states):
                    start_idx = np.random.randint(low = 0, high = len(traj.states)-length)
                    out_traj.states = traj.states[start_idx:start_idx+length]
                    out_traj.actions = traj.actions[start_idx:int(start_idx+length)]
                    out.append(out_traj)
        return out

    def append_to_best(self,rollout):
        print "Demonstration manager coutner incremented.",self.counter
        self.counter+=1
        if len(self.trajectories)<self.max_size:
            self.trajectories.append(rollout)
        else:
            reward_avg = np.mean(rollout.rewards)
            stored_avg = [np.mean(rol.rewards) for rol in self.trajectories]
            print "STORED AVERAGE REWARDS",stored_avg
            if reward_avg>np.amin(stored_avg):
                self.trajectories[np.argmin(stored_avg)] = rollout
        if self.counter > 100:
            self.save()

    def append_to_worst(self,rollout):
        print "Demonstration manager coutner incremented.",self.counter
        self.counter+=1
        if len(self.trajectories)<self.max_size:
            self.trajectories.append(rollout)
        else:
            reward_avg = np.mean(rollout.rewards)
            stored_avg = [np.mean(rol.rewards) for rol in self.trajectories]
            print "STORED AVERAGE REWARDS",stored_avg
            if reward_avg<np.amax(stored_avg):
                self.trajectories[np.argmax(stored_avg)] = rollout
        if self.counter > 100:
            self.save()
    def save(self):
        with open(self.base_filename + ".pkl", 'wb') as handle:
            pickle.dump(self.trajectories, handle)

