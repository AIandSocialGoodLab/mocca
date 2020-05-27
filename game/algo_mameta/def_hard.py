import numpy as np
import tensorflow as tf
import gym
import time
# from spinup.algos.tf1.sac import core

import algo_mameta.core as core
from spinup.algos.tf1.sac.core import get_vars
from spinup.utils.logx import EpochLogger

import envs

# note: currently can only use v1
# from abstractGameLP.createGraph import *

# must be in the same directory




class DefHard():

    def __init__(self, env_fn):

        env = env_fn()

        self.direcMatrix = np.zeros((10,10,2))

                
        up = [0,20]
        down = [0,-20]
        right = [20,0]
        left = [-20, 0]

        self.direcMatrix[5][5] = down
        self.direcMatrix[4][5] = left
        self.direcMatrix[4][4] = down 
        self.direcMatrix[3][4] = left
        self.direcMatrix[3][3] = left
        self.direcMatrix[3][2] = left
        self.direcMatrix[3][1] = down
        self.direcMatrix[2][1] = right
        self.direcMatrix[2][2] = right
        self.direcMatrix[2][3] = right
        self.direcMatrix[2][4] = down
        self.direcMatrix[1][4] = right
        self.direcMatrix[1][5] = down
        self.direcMatrix[0][5] = right
        self.direcMatrix[0][6] = up
        self.direcMatrix[1][6] = right
        self.direcMatrix[1][7] = up
        self.direcMatrix[2][7] = right
        self.direcMatrix[2][8] = up
        self.direcMatrix[3][8] = up
        self.direcMatrix[4][8] = up
        self.direcMatrix[5][8] = up
        self.direcMatrix[6][8] = up
        self.direcMatrix[7][8] = up
        self.direcMatrix[8][8] = left
        self.direcMatrix[8][7] = down
        self.direcMatrix[7][7] = left
        self.direcMatrix[7][6] = down
        self.direcMatrix[6][6] = left
        self.direcMatrix[6][5] = down


    def reset(self):
        pass

    def set_session(self, sess):
        pass


    def act(self, o, t, deterministic=False):

        curX = o[0]
        curY = o[1]

        r = int(curY // 50)
        c = int(curX // 50)

        # print(r,c)
        return np.copy(self.direcMatrix[r][c])
        
        
    def train(self, o, a, r, o2, d, t, oa):
        pass










