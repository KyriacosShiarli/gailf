import argparse
import gym
from gym.spaces import Box, Discrete
from keras.layers import Input, Dense, Lambda, Reshape, merge,Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import model_from_json, Model
import theano.tensor as T
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from functools import partial
import os
import time
import pdb
from bokeh.plotting import figure, output_file, show,save
import pickle
import tensorflow as tf
from social_policy_search.replay_memory import ReplayMemory

def scaled_tanh(x):
    return 2*T.tanh(x)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest="batch_norm")
parser.add_argument('--outf', type=str, default="./results/")
parser.add_argument('--max_norm', type=int)
parser.add_argument('--unit_norm', action='store_true', default=False)
parser.add_argument('--l2_reg', type=float,default = 0.00001)
parser.add_argument('--l1_reg', type=float)
parser.add_argument('--replay_size', type=int, default=50000)
parser.add_argument('--train_repeat', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--expert_model', type=str, default=None) #'./results/pendulum/01_Jan_2017_18_18_25'
parser.add_argument('--expert_episodes', type=int, default=10)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', type=float, default=0.0002)
parser.add_argument('--d_lr', type=float, default=0.00001)
parser.add_argument('--noise', choices=['linear_decay', 'exp_decay', 'fixed', 'covariance','epsilon_greedy'], default='linear_decay')
parser.add_argument('--noise_scale', type=float, default=0.001)
parser.add_argument('--display', action='store_true', default=False)
parser.add_argument('--priority_prob', type=float, default=0.5)
parser.add_argument('--d_sampling', type=bool, default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--gym_record')
parser.add_argument('--environment',default = 'Pendulum-v0')
args = parser.parse_args()
args.outf = args.outf+args.environment +'/'



class NafNet(object):
    def __init__(self,env, sess, scope = 'NAF'):
        self.sess = sess
        self.env = env

        self.x= x = Input(shape=env.observation_space.shape, name='x')
        self.u = Input(shape=env.action_space.shape, name='u')
        if args.batch_norm:
            h = BatchNormalization()(x)
        else:
            h = x
        for i in xrange(args.layers):
            h = Dense(args.hidden_size, activation=args.activation, name='h' + str(i + 1),
                      W_constraint=W_constraint, W_regularizer=regularizer())(h)
            if args.batch_norm and i != args.layers - 1:
                h = BatchNormalization()(h)
        self.v = Dense(1, name='v', W_constraint=W_constraint, W_regularizer=regularizer())(h)
        m = Dense(num_actuators, name='m_pre', W_constraint=W_constraint, W_regularizer=regularizer(), activation='tanh')(h)
        self.m = Lambda(scale_layer, name="m", arguments={"scales": list(env.action_space.high)})([m])
        self.l0 = Dense(num_actuators * (num_actuators + 1) / 2, name='l0',
                   W_constraint=W_constraint, W_regularizer=regularizer())(h)

        pivot = 0
        rows = []
        num_actions = self.env.action_space.shape[0]
        for idx in xrange(num_actions):
            start_index = idx + pivot

            diag_elem = tf.exp(tf.slice(self.l0, (0, start_index + idx), (-1, 1)))
            non_diag_elems = tf.slice(self.l0, (0, start_index), (-1, idx))
            row = tf.pad(tf.concat((non_diag_elems, diag_elem), 1), ((0, 0), (0, num_actions - idx - 1)))
            rows.append(row)

            pivot += idx

        L = tf.stack(rows, axis=1)
        P = tf.batch_matmul(L, tf.transpose(L, (0, 2, 1)))

        tmp = tf.expand_dims(u - self.m, -1)
        A = -tf.batch_matmul(tf.transpose(tmp, [0, 2, 1]), tf.batch_matmul(P, tmp)) / 2
        A = tf.reshape(A, [-1, 1])
        Q = A + V
        self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(Q)), name='loss')
        self.variables = get_variables(scope)

    def get_outs(self,obs):
        feed_dict = {self.x : obs}
        return self.sess.run([self.L],feed_dict)



def scale_layer(args,**kwargs):
    actuator_scales = np.array(kwargs['scales'])
    layer = args
    return layer*actuator_scales


def collect_demonstrations(separate_episodes = False):
    model_f = args.expert_model + "/model.json"
    weights_f = args.expert_model + "/weights.h5"
    with open(model_f, 'r') as model_file:
        expert = model_from_json(model_file.read())
    expert.load_weights(weights_f)
    #expert.compile(optimizer="SGD", loss='mse')
    nb_actions = expert.layers[-1].output_shape[-1]
    actions = np.linspace(-2, 2, nb_actions)

    for i in expert.layers:
        if str(i.name)=="m":
            m=i
        elif str(i.name)=="x":
            x = i
        elif str(i.name)=="u":
            u = i
    fmu = K.function([K.learning_phase(), x.input], m.output)
    expert.mu = lambda x: fmu([0, x])

    render = False
    ## Gather data from the policy by saving trajectories over episodes in simulation.

    env = gym.make(args.environment)
    if separate_episodes:
        expert_episodes = []
    else:
        O_all = False
        A_all = False
    expert_rew = []
    for e in xrange(args.expert_episodes):
        # R = np.zeros(args.batch_size+max_4q)
        O = np.zeros([args.max_timesteps, env.observation_space.shape[0]])
        A = np.zeros([args.max_timesteps, env.action_space.shape[0]])
        if True:
            oprev = env.reset()  # returns initial observation
        a = expert.mu(np.array([oprev]))[0,0]
        ar = 0
        for t in xrange(args.max_timesteps):

            O[t, :] = oprev
            A[t] = a
            (onext, r_real, done, info) = env.step(np.array([a]))
            ar += r_real
            if render is True:
                env.render()
            env.render()
            a = expert.mu(np.array([onext]))[0,0]
            oprev = onext
            if done:
                break
        expert_rew.append(ar / args.max_timesteps)
        print "DEMONSRATION AVERAGE REWARD", expert_rew[-1]
        if separate_episodes:
            expert_episodes.append({"states": O, "actions": A})
        else:
            if ar/args.max_timesteps>-2.5 and O_all is False:
                O_all = O
                A_all = A
            elif ar/args.max_timesteps>-2.5:
                O_all = np.vstack((O_all,O))
                A_all = np.vstack((A_all, A))
            expert_episodes = {"states":np.array(O_all),"actions":np.array(A_all)}
    return expert_episodes



if args.expert_model is not None:
    expert_demos = collect_demonstrations()

#assert K._BACKEND == 'tf', "only works with Theano as backend"
sess = tf.Session()
K.set_session(sess)
# create environment
env = gym.make(args.environment)

assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"
assert len(env.action_space.shape) == 1
num_actuators = env.action_space.shape[0]
print "num_actuators:", num_actuators

# start monitor for OpenAI Gym
if args.gym_record:
  env.monitor.start(args.gym_record)

# optional norm constraint
if args.max_norm:
  W_constraint = maxnorm(args.max_norm)
elif args.unit_norm:
  W_constraint = unitnorm()
else:
  W_constraint = None

# optional regularizer
def regularizer():
  if args.l2_reg:
    return l2(args.l2_reg)
  elif args.l1_reg:
    return l1(args.l1_reg)
  else:
    return None


def create_discriminator():
    # Discriminator uses two input layers one for states and one for actions
    inp_s = Input(shape=env.observation_space.shape)
    inp_a = Input(shape=env.action_space.shape)
    # These are then merged
    h = merge([inp_s, inp_a], mode='concat')
    for i in xrange(args.layers):
        h = Dense(args.hidden_size+100, activation="relu", W_regularizer=regularizer())(h)
        h = Dropout(0.3)(h)
    d_out = Dense(2, activation="softmax")(h)
    dsc = Model(input=[inp_s, inp_a], output=d_out)
    opt = Adam(lr=args.d_lr)
    dsc.compile(optimizer=opt, loss="categorical_crossentropy")
    return dsc




net = NafNet(env,sess)



obs = env.reset()
out = net.get_outs([obs])


pdb.set_trace()

# main model
model = Model(input=[x,u], output=q)
model.summary()

if args.optimizer == 'adam':
  optimizer = Adam(args.optimizer_lr)
elif args.optimizer == 'rmsprop':
  optimizer = RMSprop(args.optimizer_lr)
else:
  assert False
model.compile(optimizer=optimizer, loss='mse')

# another set of layers for target model
x, u, m, v, q, p, a = createLayers()

# V() function uses target model weights
fV = K.function([K.learning_phase(), x], v)
V = lambda x: fV([0, x])

# target model is initialized from main model
target_model = Model(input=[x,u], output=q)
target_model.set_weights(model.get_weights())

discriminator = create_discriminator()
# replay memory
R = ReplayMemory(args.replay_size)
loss_discriminator = []
loss_generator = []
# the main learning loop
total_reward = 0
Rw = []
for i_episode in xrange(args.episodes):
    observation = env.reset()
    #print "initial state:", observation
    episode_reward = 0
    rew = 0
    for t in xrange(args.max_timesteps):
        if args.display:
          env.render()

        # predict the mean action from current observation
        x = np.array([observation])
        uu = mu(x)[0]

        # add exploration noise to the action
        if args.noise == 'linear_decay':
          action = uu + np.random.randn(num_actuators) / (args.noise_scale*i_episode + 1)
        elif args.noise == 'exp_decay':
          action = uu + np.random.randn(num_actuators) * 10 ** -(i_episode*args.noise_scale)
        elif args.noise == 'epsilon_greedy':
            g_prob = args.noise_scale*10**-(i_episode*0.01)
            greedy = np.random.choice(2,p=[g_prob,1-g_prob])
            if greedy ==1:
                action = uu + np.random.randn(num_actuators) / (1. * i_episode + 1)
            else:
                action = np.array([np.random.uniform(-2,2)])

        elif args.noise == 'fixed':
          action = uu + np.random.randn(num_actuators) * args.noise_scale
        elif args.noise == 'covariance':
          if num_actuators == 1:
            std = np.minimum(args.noise_scale / P(x)[0], 1)
            #print "std:", std
            action = np.random.normal(u, std, size=(1,))
          else:
            cov = np.minimum(np.linalg.inv(P(x)[0]) * args.noise_scale, 1)
            #print "covariance:", cov
            action = np.random.multivariate_normal(u, cov)
        else:
          assert False
        #print "action:", action, "Q:", Q(x, np.array([action])), "V:", V(x)
        #print "action:", action, "advantage:", A(x, np.array([action]))
        #print "mu:", u, "action:", action
        #print "Q(mu):", Q(x, np.array([u])), "Q(action):", Q(x, np.array([action]))

        # take the action and record reward
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward
        #print "poststate:", observation

        # add experience to replay memory
        R.append([np.array(x[0]), np.array(action),np.array(observation), reward , done])
        rew+=reward
        loss = 0

        # perform train_repeat Q-updates
        if R.get_size() > args.batch_size:
            for k in xrange(args.train_repeat):

                mem_samp_g, samp_idx = R.sample(args.batch_size)

                rewards = mem_samp_g['rewards']
                if args.expert_model!=None:
                    priority = np.random.choice(2, p=[1 - args.priority_prob, args.priority_prob])
                    if priority == 1 and args.d_sampling is True:
                        mem_samp_d, samp_idx = R.sample_prioritised(args.batch_size)
                    elif priority == 0 and args.d_sampling is True:
                        mem_samp_d, samp_idx = R.sample(args.batch_size)
                    else:
                        mem_samp_d, samp_idx = R.get_recent(args.batch_size)


                    expert_indexes = np.random.randint(low=0, high=expert_demos['states'].shape[0], size=args.batch_size)
                    O_e = expert_demos["states"][expert_indexes]
                    #up_acts = mu(preobs)
                    A_e =expert_demos["actions"][expert_indexes]
                    y_e = np.tile(np.array([1., 0.]), args.batch_size).reshape(args.batch_size, 2)
                    y_a = np.tile(np.array([0., 1.]), args.batch_size).reshape(args.batch_size, 2)
                    O = np.vstack([mem_samp_d['preobs'], O_e])
                    A = np.vstack([mem_samp_d['actions'],A_e])
                    #A = np.vstack([up_acts, A_e])
                    y = np.vstack([y_a, y_e])
                    loss_d = discriminator.train_on_batch([O,A],y)
                    loss_discriminator.append(loss_d)
                    probs = discriminator.predict([mem_samp_d['preobs'], mem_samp_d['actions']])

                    R.set_priority(samp_idx,probs[:,0])

                    rewards = -np.log(discriminator.predict([mem_samp_g['preobs'], mem_samp_g['actions']])[:,1])

                # Q-update
                vv = V(mem_samp_g['postobs']) # the v comes from the target network
                y = rewards + args.gamma * np.squeeze(vv)

                loss_g = model.train_on_batch([mem_samp_g['preobs'], mem_samp_g['actions']], y) # and we update the actual network.
                loss_generator.append(loss_g)
                # copy weights to target model, averaged by tau
                weights = model.get_weights()
                target_weights = target_model.get_weights()
                for i in xrange(len(weights)):
                    target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
                    target_model.set_weights(target_weights)
                #print "average loss:", loss/k

        if done:
            break

    loss_d_avg = np.mean(loss_discriminator[-args.max_timesteps*args.train_repeat+1:]) if len(loss_discriminator)>0 else False
    loss_g_avg = np.mean(loss_generator[-args.max_timesteps * args.train_repeat+1:]) if len(loss_discriminator)>0 else False
    print "LOSS D AVG", loss_d_avg
    print "LOSS G AVG", loss_g_avg

    print "Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward
    Rw.append(episode_reward/args.max_timesteps)

print "Average reward per episode {}".format(total_reward / args.episodes)


#### BOOK KEEPING #####

save = True
if save:
    datestr_suff = time.strftime("%d_%b_%Y_%H_%M_%S", time.gmtime())
    basedir = args.outf
    if basedir[-1] != '/':
        basedir += '/'
    outdir = basedir+ datestr_suff+"/"
    print outdir
    print os.path.isdir(outdir)
    os.makedirs(outdir)


    json_str = model.to_json()
    with open(outdir + '/model.json', 'w+') as outf:
        outf.write(json_str)

    arg_alt = args
    with open(outdir + '/params.pkl', 'w+') as outf:
        pickle.dump(args, outf)

    print 'Training complete. Saving results.'
    model.save_weights(outdir + '/weights.h5')


metrics = {"per_episode_r":Rw}

output_file(outdir + "/learning_curve.html")
p = figure(plot_width=620, plot_height=500)

p.legend.label_text_font_size = "18pt"
p.xaxis.axis_label = 'Average Reward per episode'
p.xaxis.axis_label_text_font_size = '20pt'
p.xaxis.axis_label_text_font_style = 'normal'
p.yaxis.axis_label = 'Episodes'
p.yaxis.axis_label_text_font_size = '20pt'
p.yaxis.axis_label_text_font_style = 'normal'
p.line(range(len(Rw)),Rw)
show(p)
#save(p)


if args.expert_model!=None:
    metrics["generator_loss"] = loss_generator
    metrics["discriminator_loss"] = loss_discriminator
    plt.plot(loss_generator)
    plt.savefig(outdir + '/generator_loss.png')
    plt.figure()
    plt.plot(loss_discriminator)
    plt.savefig(outdir + '/discriminator_loss.png')
    #plt.show()

with open(outdir + '/metrics.pkl', 'w+') as outf:
    pickle.dump(metrics, outf)


if args.notes is not None:
    with open(outdir + '/notes.txt', 'w+') as outf:
        outf.write(args.notes)


if args.gym_record:
  env.monitor.close()
