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

# from abstractGameLP.createGraph_v3 import *

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

    
    


"""

"""


# def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#         polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
#         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#         logger_kwargs=dict(), save_freq=1):
#     """
#     Soft Actor-Critic (SAC)


#     Args:
#         env_fn : A function which creates a copy of the environment.
#             The environment must satisfy the OpenAI Gym API.

#         actor_critic: A function which takes in placeholder symbols 
#             for state, ``x_ph``, and action, ``a_ph``, and returns the main 
#             outputs from the agent's Tensorflow computation graph:

#             ===========  ================  ======================================
#             Symbol       Shape             Description
#             ===========  ================  ======================================
#             ``mu``       (batch, act_dim)  | Computes mean actions from policy
#                                            | given states.
#             ``pi``       (batch, act_dim)  | Samples actions from policy given 
#                                            | states.
#             ``logp_pi``  (batch,)          | Gives log probability, according to
#                                            | the policy, of the action sampled by
#                                            | ``pi``. Critical: must be differentiable
#                                            | with respect to policy parameters all
#                                            | the way through action sampling.
#             ``q1``       (batch,)          | Gives one estimate of Q* for 
#                                            | states in ``x_ph`` and actions in
#                                            | ``a_ph``.
#             ``q2``       (batch,)          | Gives another estimate of Q* for 
#                                            | states in ``x_ph`` and actions in
#                                            | ``a_ph``.
#             ===========  ================  ======================================

#         ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
#             function you provided to SAC.

#         seed (int): Seed for random number generators.

#         steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
#             for the agent and the environment in each epoch.

#         epochs (int): Number of epochs to run and train agent.

#         replay_size (int): Maximum length of replay buffer.

#         gamma (float): Discount factor. (Always between 0 and 1.)

#         polyak (float): Interpolation factor in polyak averaging for target 
#             networks. Target networks are updated towards main networks 
#             according to:

#             .. math:: \\theta_{\\text{targ}} \\leftarrow 
#                 \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

#             where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
#             close to 1.)

#         lr (float): Learning rate (used for both policy and value learning).

#         alpha (float): Entropy regularization coefficient. (Equivalent to 
#             inverse of reward scale in the original SAC paper.)

#         batch_size (int): Minibatch size for SGD.

#         start_steps (int): Number of steps for uniform-random action selection,
#             before running real policy. Helps exploration.

#         update_after (int): Number of env interactions to collect before
#             starting to do gradient descent updates. Ensures replay buffer
#             is full enough for useful updates.

#         update_every (int): Number of env interactions that should elapse
#             between gradient descent updates. Note: Regardless of how long 
#             you wait between updates, the ratio of env steps to gradient steps 
#             is locked to 1.

#         num_test_episodes (int): Number of episodes to test the deterministic
#             policy at the end of each epoch.

#         max_ep_len (int): Maximum length of trajectory / episode / rollout.

#         logger_kwargs (dict): Keyword args for EpochLogger.

#         save_freq (int): How often (in terms of gap between epochs) to save
#             the current policy and value function.

#     """

#     logger = EpochLogger(**logger_kwargs)
#     logger.save_config(locals())

#     tf.set_random_seed(seed)
#     np.random.seed(seed)

#     env, test_env = env_fn(), env_fn()
#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]

#     # Action limit for clamping: critically, assumes all dimensions share the same bound!
#     act_limit = env.action_space.high[0]

#     # Share information about action space with policy architecture
#     ac_kwargs['action_space'] = env.action_space

#     print("---")
#     print("obs_dim:", obs_dim)
#     print("act_dim:", act_dim)
#     print("act_limit:", act_limit)
#     print("env.action_space", env.action_space)
    

#     # Inputs to computation graph
#     x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

#     # Main outputs from computation graph
#     with tf.variable_scope('main'):
#         mu, pi, logp_pi, q1, q2 = actor_critic(x_ph, a_ph, **ac_kwargs)

#         # ty: placeholder to hold meta strategy param TODO: check meta_log_std dimension
#         meta_mu = core.placeholder(act_dim)
#         meta_log_std = core.placeholder(act_dim)

#         meta_mu_next = core.placeholder(act_dim)
#         meta_log_std_next = core.placeholder(act_dim)

#         # ty: logp_phi
#         logp_phi = core.gaussian_likelihood(a_ph, meta_mu, meta_log_std)
#         _, _, logp_phi = core.apply_squashing_func(meta_mu, a_ph, logp_phi)


#     with tf.variable_scope('main', reuse=True):
#         # compose q with pi, for pi-learning
#         _, _, _, q1_pi, q2_pi = actor_critic(x_ph, pi, **ac_kwargs)

#         # get actions and log probs of actions for next states, for Q-learning
#         _, pi_next, logp_pi_next, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)
        
#         # ty: logp_phi_next, make sure the action is from the current policy
#         logp_phi_next = core.gaussian_likelihood(pi_next, meta_mu_next, meta_log_std_next)
#         _, _, logp_phi_next = core.apply_squashing_func(meta_mu_next, pi_next, logp_phi_next)

        

#     # Target value network
#     with tf.variable_scope('target'):
#         # target q values, using actions from *current* policy
#         _, _, _, q1_targ, q2_targ  = actor_critic(x2_ph, pi_next, **ac_kwargs)

#     # Experience buffer
#     replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

#     # Count variables
#     var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
#     print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

#     # Min Double-Q:
#     min_q_pi = tf.minimum(q1_pi, q2_pi)
#     min_q_targ = tf.minimum(q1_targ, q2_targ)

#     # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
#     q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_targ - alpha * logp_pi_next + alpha * logp_phi_next))

#     # Soft actor-critic losses
#     pi_loss = tf.reduce_mean(alpha * logp_pi - alpha * logp_phi - min_q_pi)
#     q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
#     q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
#     value_loss = q1_loss + q2_loss

#     # Policy train op 
#     # (has to be separate from value train op, because q1_pi appears in pi_loss)
#     pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#     train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

#     # Value train op
#     # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
#     value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#     value_params = get_vars('main/q')
#     with tf.control_dependencies([train_pi_op]):
#         train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

#     # Polyak averaging for target variables
#     # (control flow because sess.run otherwise evaluates in nondeterministic order)
#     with tf.control_dependencies([train_value_op]):
#         target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
#                                   for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

#     # All ops to call during one training step
#     step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, 
#                 train_pi_op, train_value_op, target_update]

#     # Initializing targets to match main variables
#     target_init = tf.group([tf.assign(v_targ, v_main)
#                               for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     sess.run(target_init)

#     # Setup model saving
#     logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
#                                 outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

#     def get_action(o, deterministic=False):
#         act_op = mu if deterministic else pi
#         return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    
#     def test_agent():
#         for j in range(num_test_episodes):
#             o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
#             while not(d or (ep_len == max_ep_len)):
#                 # Take deterministic actions at test time 
#                 o, r, d, _ = test_env.step(get_action(o, True))
#                 ep_ret += r
#                 ep_len += 1
#             logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

#     start_time = time.time()
#     o, ep_ret, ep_len = env.reset(), 0, 0
#     total_steps = steps_per_epoch * epochs

#     # Main loop: collect experience in env and update/log each epoch
#     for t in range(total_steps):

#         # Until start_steps have elapsed, randomly sample actions
#         # from a uniform distribution for better exploration. Afterwards, 
#         # use the learned policy.
#         if t > start_steps:
#             a = get_action(o)
#         else:
#             a = env.action_space.sample()

#         # Step the env
#         o2, r, d, _ = env.step(a)
#         ep_ret += r
#         ep_len += 1

#         # Ignore the "done" signal if it comes from hitting the time
#         # horizon (that is, when it's an artificial terminal signal
#         # that isn't based on the agent's state)
#         d = False if ep_len==max_ep_len else d

       
#         # Store experience to replay buffer
#         replay_buffer.store(o, a, r, o2, d)

#         # Super critical, easy to overlook step: make sure to update 
#         # most recent observation!
#         o = o2

#         # End of trajectory handling
#         if d or (ep_len == max_ep_len):
#             logger.store(EpRet=ep_ret, EpLen=ep_len)
#             o, ep_ret, ep_len = env.reset(), 0, 0

#         # ty: temporary values for meta_mu, ...
#         # temp0s = np.ones((100,4)) * (-10)
#         # ty: temporary variance for meta strategy
#         temp1s = np.ones((100,4))
#         # Update handling
#         if t >= update_after and t % update_every == 0:
#             for j in range(update_every):
#                 batch = replay_buffer.sample_batch(batch_size)
#                 feed_dict = {x_ph: batch['obs1'],
#                              x2_ph: batch['obs2'],
#                              a_ph: batch['acts'],
#                              r_ph: batch['rews'],
#                              d_ph: batch['done'],
#                              # ty: fill in correct values
#                              meta_mu: np.apply_along_axis(obs2mu, 1, batch['obs1']),
#                              meta_log_std: temp1s,
#                              meta_mu_next: np.apply_along_axis(obs2mu, 1, batch['obs2']),
#                              meta_log_std_next: temp1s
#                             }
#                 outs = sess.run(step_ops, feed_dict)
#                 logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
#                              Q1Vals=outs[3], Q2Vals=outs[4], LogPi=outs[5])

#         # End of epoch wrap-up
#         if (t+1) % steps_per_epoch == 0:
#             epoch = (t+1) // steps_per_epoch

#             # Save model
#             if (epoch % save_freq == 0) or (epoch == epochs):
#                 logger.save_state({'env': env}, None)

#             # Test the performance of the deterministic version of the agent.
#             test_agent()

#             # Log info about epoch
#             logger.log_tabular('Epoch', epoch)
#             logger.log_tabular('EpRet', with_min_and_max=True)
#             logger.log_tabular('TestEpRet', with_min_and_max=True)
#             logger.log_tabular('EpLen', average_only=True)
#             logger.log_tabular('TestEpLen', average_only=True)
#             logger.log_tabular('TotalEnvInteracts', t)
#             logger.log_tabular('Q1Vals', with_min_and_max=True) 
#             logger.log_tabular('Q2Vals', with_min_and_max=True) 
#             logger.log_tabular('LogPi', with_min_and_max=True)
#             logger.log_tabular('LossPi', average_only=True)
#             logger.log_tabular('LossQ1', average_only=True)
#             logger.log_tabular('LossQ2', average_only=True)
#             logger.log_tabular('Time', time.time()-start_time)
#             logger.dump_tabular()