import numpy as np
import tensorflow as tf
import gym
import time
# from spinup.algos.tf1.sac import core

# import algo_mameta.core as core
import algo_mameta.core_alternative as core
from spinup.algos.tf1.sac.core import get_vars
from spinup.utils.logx import EpochLogger

import envs

# from abstractGameLP.createGraph import *
# from abstractGameLP.createGraph_v2 import *

from abstractGameLP.createGraph_v3 import *

# must be in the same directory

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class CentralizeReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, otheract_dim):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.otheracts_buf = np.zeros([size, otheract_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, otheract):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.otheracts_buf[self.ptr] = otheract
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    otheracts=self.otheracts_buf[idxs])


"""
alternate: take std from core_alternative
"""

class DefSacAlternativeMeta():

    def __init__(self, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.01,
                 batch_size=100, start_steps=10000, update_after=1000, update_every=50,
                 num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(), save_freq=1, centralizeQ=False):

        
        # flag of whether using meta stratey 0: False; 1: True
        self.m = 1
        
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.centralizeQ = centralizeQ

        env, self.test_env = env_fn(), env_fn()
        obs_dim = env.def_observation_space.shape[0]
        act_dim = env.def_action_space.shape[0]

        # used for centralized q function
        otheract_dim = env.att_action_space.shape[0]

        act_limit = env.def_action_space.high[0]

        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph, self.oa_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None, otheract_dim)

        ac_kwargs['action_space'] = env.def_action_space
        ac_kwargs['centralizeQ'] = centralizeQ
        ac_kwargs['oa'] = self.oa_ph

        # Main outputs from computation graph
        with tf.variable_scope('def'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.log_std = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

            # ty: placeholder to hold meta strategy param TODO: check meta_log_std dimension
            self.meta_mu = core.placeholder(act_dim)
            # self.meta_log_std = core.placeholder(act_dim)

            self.meta_mu_next = core.placeholder(act_dim)
            # self.meta_log_std_next = core.placeholder(act_dim)

            # ty: logp_phi
            self.logp_phi = core.gaussian_likelihood(self.a_ph, self.meta_mu, self.log_std)
            _, _, self.logp_phi = core.apply_squashing_func(self.meta_mu, self.a_ph, self.logp_phi)

        with tf.variable_scope('def', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, self.q1_pi, self.q2_pi, _ = actor_critic(self.x_ph, self.pi, **ac_kwargs)

            # get actions and log probs of actions for next states, for Q-learning
            _, self.pi_next, self.logp_pi_next, _, _, self.log_std_next = actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)
            
            # ty: logp_phi_next, make sure the action is from the current policy
            self.logp_phi_next = core.gaussian_likelihood(self.pi_next, self.meta_mu_next, self.log_std_next)
            _, _, self.logp_phi_next = core.apply_squashing_func(self.meta_mu_next, self.pi_next, self.logp_phi_next)


         # Target value network
        with tf.variable_scope('def_target'):
            # target q values, using actions from *current* policy
            _, _, _, self.q1_targ, self.q2_targ, _  = actor_critic(self.x2_ph, self.pi_next, **ac_kwargs)


        # if not centralizeQ:
        #     self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # else:
        self.replay_buffer = CentralizeReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, otheract_dim=otheract_dim)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['def/pi', 'def/q1', 'def/q2', 'def'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)


        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)
        min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)

        # original
        q_backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*(min_q_targ - alpha * self.logp_pi_next + alpha * self.logp_phi_next * self.m))

        # with abs
        # q_backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*(min_q_targ + tf.math.abs(alpha * self.logp_pi_next - alpha * self.logp_phi_next * self.m)))

        # Soft actor-critic losses
        # original
        pi_loss = tf.reduce_mean(alpha * self.logp_pi - alpha * self.logp_phi * self.m - min_q_pi)

        # with abs
        # pi_loss = tf.reduce_mean(-tf.math.abs(alpha * self.logp_pi - alpha * self.logp_phi * self.m) - min_q_pi)

        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        value_loss = q1_loss + q2_loss

        # Policy train op 
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('def/pi'))


        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('def/q')
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

            # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([self.train_value_op]):
            self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                      for v_main, v_targ in zip(get_vars('def'), get_vars('def_target'))])


        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, self.q1, self.q2, self.logp_pi, 
                    self.train_pi_op, self.train_value_op, self.target_update]

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('def'), get_vars('def_target'))])


    def set_session(self, sess):
        self.sess = sess


    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1,-1)})[0]

    def act(self, o, t, deterministic=False):
        if t > self.start_steps:
            return self.get_action(o,deterministic)
        else:
            # sketchy code
            return self.test_env.def_action_space.sample()
        

    # note: oa will always be provided with values
    # the decentralized version simply does not use it
    def train(self, o, a, r, o2, d, t, oa):

        
        self.replay_buffer.store(o, a, r, o2, d, oa)
        
        if t >= self.update_after and t % self.update_every == 0:

            for j in range(self.update_every):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                feed_dict = {self.x_ph: batch['obs1'],
                             self.x2_ph: batch['obs2'],
                             self.a_ph: batch['acts'],
                             self.r_ph: batch['rews'],
                             self.d_ph: batch['done'],
                             # ty: fill in correct values
                             self.meta_mu: np.apply_along_axis(obs2mu, 1, batch['obs1']),
                             self.meta_mu_next: np.apply_along_axis(obs2mu, 1, batch['obs2']),
                             # ty for centralized Q
                             self.oa_ph: batch['otheracts']
                            }

                outs = self.sess.run(self.step_ops, feed_dict)
                # ty todo: add logging here


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    
    




