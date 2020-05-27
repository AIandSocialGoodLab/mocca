"""
different setting - final paper setting

attacker stealthy when not attacking
initial location of defender & attacker randomized
no uav
multiple targets allowed
penetration time
no uav
"""

"""
same setting as v3, but absolutely no uav here
"""

import gym
import numpy as np

MAX_STEPS = 60



class CatcherEnv(gym.Env):

	def __init__(self):

		self.obs_limit = 500.0
		self.act_limit = 20.0
		
		# ty: this is the place to modify reward
		self.tarPoss = [(400, 400), (200,100), (100,200)]
		self.tarPossNdarray = np.array(self.tarPoss)
		self.tarRs = [20, 15, 10]
		self.num_target = len(self.tarPoss)
		assert(len(self.tarPoss)==len(self.tarRs))

		# ty: this is the observation for defender
		# defender x, y, uav x, y, find or not, target 1 x, y
		self.def_observation_space = gym.spaces.Box(**self._initDefObsSpace())

		# ty: this is the observation for attacker [todo: include followed by uav flag]
		# self.att_observation_space = gym.spaces.Box(low=np.array([0,0], dtype=np.float32) ,high=np.array([500,500], dtype=np.float32))

		self.att_observation_space = gym.spaces.Box(**self._initAttObsSpace())

		# now this contains all agent's actions ty: the actions dimension should remain unchanged
		# defender delta x, delta y, uav delta x, delta y, attacker delta x, delta y
		self.action_space = gym.spaces.Box(low=np.array([-20,-20,-20,-20, -20, -20],dtype=np.float32), high=np.array([20, 20, 20, 20, 20, 20],dtype=np.float32))

		# defender: def delta x, delta y, uav delta x, delta y
		self.def_action_space = gym.spaces.Box(low=np.array([-20,-20],dtype=np.float32), high=np.array([20, 20],dtype=np.float32))

		# attacker: att delta x, delta y
		self.att_action_space = gym.spaces.Box(low=np.array([-20,-20],dtype=np.float32), high=np.array([20,20],dtype=np.float32)) 

		# attacker x, y
		# ty: att_state is not updated!!
		# self.att_state = np.array([100,100], dtype=np.float32)

		self.state = None

		self.penetrationTime = 5

	def _initDefObsSpace(self):
		
		lb = np.zeros(2 + 2 * self.num_target, dtype=np.float32)
		ub = np.ones(2 + 2 * self.num_target, dtype=np.float32) * self.obs_limit
		
		return {"low":lb, "high":ub}

	def _initAttObsSpace(self):
		# ty: attacker also knows the loaction of the targets
		lb = np.zeros(2 + 2 * self.num_target, dtype=np.float32)
		ub = np.ones(2 + 2 * self.num_target, dtype=np.float32) * self.obs_limit

		return {"low":lb, "high":ub}


	def _initDefState(self):

		# modify this to be stochastic later
		# 2 3 4th argument was for uav, no uav now, just set to 1
		# return np.array([250,250,1,1,1])

		defState = self.def_observation_space.sample()[:2]
		
		return defState


	def _initTarget(self):
		self.tarPos2rDict = dict()
		for i in range(0, self.num_target):
			self.tarPos2rDict[self.tarPoss[i]] = self.tarRs[i]
		return np.ravel(self.tarPoss)


	def _initAttState(self):
		# modify this to be stochastic later
		# return np.array([100, 100])

		attState = self.att_observation_space.sample()[:2]
		
		return attState


	def getDefObs(self, obs_all):
		# ty: IMPORTANT -> this should be consistent with _initDefObsSpace
		return obs_all[:2+self.num_target*2]


	def getAttObs(self, obs_all):
		# ty: IMPORTANT -> this should be consistent with _initAttObsSpace
		# return obs_all[5+self.num_target*2:]
		# ty: attacker also knows the location of the target
		return obs_all[2: ]


	def reset(self):
		
		# [ty: this is hardcoded, should be enhanced later]
		# defender x, y, uav x, y, find or not, target 1 x, y + attacker x, y
		# self.state = np.array([250,250,250,250,0,400,400,100,100],dtype=np.float32)
		self.state = np.concatenate([self._initDefState(), self._initTarget(), self._initAttState()])

		self.num_steps = 0

		self.penetrationCtrs = np.zeros(self.num_target, dtype=np.int)

		
		return self.state.copy()

	
	def _compDist(self, x1, y1, x2, y2):
		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def _isDefAtBoundary(self):

		if (self.state[0] == 0 or self.state[0] == 500 or
			self.state[1] == 0 or self.state[1] == 500 or 
			self.state[2] == 0 or self.state[2] == 500 or 
			self.state[3] == 0 or self.state[3] == 500):
			return True
		else:
			return False

	def _isAttAtBoundary(self):

		if (self.state[-2] == 0 or self.state[-2] == 500 or
			self.state[-1] == 0 or self.state[-1] == 500):
			return True
		else:
			return False

	def _calcReward(self):

		
		# if attacker at target & defender at target -> attacker caught
		# if attacker at target & penetration complete -> target attacked
		# if attacker at target & defender not at target -> penetration + 1
		# else -> discounted factor reward

		shapingFlag = 0

		for i in range(self.num_target):
			tarX, tarY = self.tarPoss[i]
			tarDefDist = self._compDist(self.state[0], self.state[1], tarX, tarY)

			if (tarDefDist <= 15.0):
				# ty: added reward shaping for defender
				shapingFlag = 1


		for i in range(self.num_target):
			tarX, tarY = self.tarPoss[i]
			tarR = self.tarRs[i]

			tarAttDist = self._compDist(self.state[-2], self.state[-1], tarX, tarY)
			attDefDist = self._compDist(self.state[-2], self.state[-1], self.state[0], self.state[1])

			# ty: change attDefDist threshold to 30.0 to help defender
			if (tarAttDist <= 15.0) and (attDefDist <= 30.0):
				# attacker caught when attacking
				return (tarR, -tarR), True, {"done": "attacker caught"}
			elif (tarAttDist <= 15.0) and (self.penetrationCtrs[i] >= self.penetrationTime):
				# attacker completes attack
				return (-tarR, tarR), True, {"done": "target attacked"}
			elif (tarAttDist <= 15.0) and (self.penetrationCtrs[i] < self.penetrationTime):
				# attacker ongoing attack
				self.penetrationCtrs[i] += 1
				return (-0.5 + shapingFlag * 2, 1.0), False, {"done": None}
			else:
				# other cases
				self.penetrationCtrs[i] = 0
		
		return (-0.5 + shapingFlag * 2, -0.5), False, {"done": None}



	def def_step(self, action):

		action[0] = np.clip(action[0], -20.0, 20.0)
		action[1] = np.clip(action[1], -20.0, 20.0)
		# action[2] = np.clip(action[2], -20.0, 20.0)
		# action[3] = np.clip(action[3], -20.0, 20.0)

		defDeltax = action[0]
		defDeltay = action[1]

		# uavDeltax = action[2]
		# uavDeltay = action[3]

		self.state[0] = np.clip(self.state[0] + defDeltax, 0.0, 500.0)
		self.state[1] = np.clip(self.state[1] + defDeltay, 0.0, 500.0)

		# ty: no uav
		# self.state[2] = np.clip(self.state[2] + uavDeltax, 0, 500)
		# self.state[3] = np.clip(self.state[3] + uavDeltay, 0, 500)


	def att_step(self, action):

		action[0] = np.clip(action[0], -20.0, 20.0)
		action[1] = np.clip(action[1], -20.0, 20.0)

		attDeltax = action[0]
		attDeltay = action[1]

		self.state[-2] = np.clip(self.state[-2] + attDeltax, 0.0, 500.0)
		self.state[-1] = np.clip(self.state[-1] + attDeltay, 0.0, 500.0)



	# ty: r1: defender reward, r2: attacker reward
	def step(self, action):
		
		self.num_steps += 1

		# ty: this is hard coded, change this
		self.def_step(action[:2])
		self.att_step(action[2:4])		
		
		# if maximum steps reached
		if self.num_steps >= MAX_STEPS:
			rs = (-10.0, -10.0)
			done = True
			info = {"done": "max steps reached"}
		elif self._isDefAtBoundary():
			rs = (-200.0, 0.0)
			done = True
			info = {"done": "def out of boundary"}
		elif self._isAttAtBoundary():
			rs = (0.0, -200.0)
			done = True
			info = {"done": "att out of boundary"}
		else:
			rs, done, info = self._calcReward()

		# ty: make changes to r, make it able to return flexible number player reward
		return self.state.copy(), rs, done, info


"""
observation:
defender position [x,y]
uav position [x,y]
find or not [0/1]
(attacker position encoded in uav position)
target position(s)


action:
defender direction, speed, uav direction speed

attacker currently encoded in environment
"""

"""
current problems:
not adjusting
"""

"""
thoughts:
when goal not sure:
use supervised learning to decide the goal?
"""

"""
change to x & y displacement
"""


"""
game abstraction

abstract strategy beforehand

tumas sandholm

regularizer

concrete algo

multiple target + attacker type

attacker have different payoff on different target

ARMOR
"""










