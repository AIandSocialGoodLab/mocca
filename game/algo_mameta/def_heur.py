# import numpy as np
# import tensorflow as tf
# import gym
# import time
# # from spinup.algos.tf1.sac import core

# import algo_mameta.core as core
# from spinup.algos.tf1.sac.core import get_vars
# from spinup.utils.logx import EpochLogger

# import envs

# # note: currently can only use v1
# from abstractGameLP.createGraph import *

# # must be in the same directory




# class DefHeur():

#     def __init__(self, env_fn):

#         env = env_fn()


#     def set_session(self, sess):
#         pass


#     def act(self, o, t, deterministic=False):

#         defX = o[0]
#         defY = o[1]

#         defPos = np.array([defX, defY])
#         a = obs2pi(o)

#         # making sure defender not getting out boundary
#         newPos = defPos + a

#         if (newPos[0] <= 0 or newPos[0] >= 500 or
#             newPos[1] <= 0 or newPos[1] >= 500):
#             return np.array([0,0])
#         else:
#             return a
        
#     def train(self, o, a, r, o2, d, t, oa):
#         pass










