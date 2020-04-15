import gym
import numpy as np

MAX_STEPS = 50


class CatcherEnv(gym.Env):

	def __init__(self):
		# todo: change this to obs for all
		# ty: this is the observation for defender
		# defender x, y, uav x, y, find or not, target 1 x, y
		self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0],dtype=np.float32),high=np.array([500,500,500,500,1,500,500],dtype=np.float32))

		self.def_observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0],dtype=np.float32),high=np.array([500,500,500,500,1,500,500],dtype=np.float32))

		# ty: this is the observation for attacker [todo: include followed by uav flag]
		self.att_observation_space = gym.spaces.Box(low=np.array([0,0], dtype=np.float32) ,high=np.array([500,500], dtype=np.float32))

		# now this contains all agent's actions
		# defender delta x, delta y, uav delta x, delta y, attacker delta x, delta y
		self.action_space = gym.spaces.Box(low=np.array([-20,-20,-20,-20, -20, -20],dtype=np.float32), high=np.array([20, 20, 20, 20, 20, 20],dtype=np.float32))

		# defender: def delta x, delta y, uav delta x, delta y
		self.def_action_space = gym.spaces.Box(low=np.array([-20,-20,-20,-20],dtype=np.float32), high=np.array([20, 20, 20, 20],dtype=np.float32))

		# attacker: att delta x, delta y
		self.att_action_space = gym.spaces.Box(low=np.array([-20,-20],dtype=np.float32), high=np.array([20,20],dtype=np.float32)) 

		# attacker x, y
		# ty: att_state is not updated!!
		self.att_state = np.array([100,100], dtype=np.float32)

		self.state = None

	def reset(self):
		
		# [ty: this is hardcoded, should be enhanced later]
		# defender x, y, uav x, y, find or not, target 1 x, y + attacker x, y
		self.state = np.array([250,250,250,250,0,400,400,100,100],dtype=np.float32)

		# defender x, y, uav x, y, find or not, target 1 x, y
		self.defender_state = np.array([250,250,250,250,0,400,400],dtype=np.float32)

		# attacker x, y
		self.att_state = np.array([100,100], dtype=np.float32)
		self.num_steps = 0

		self.lastDefTarDist = self._compDist(self.state[5], self.state[6], self.state[0], self.state[1])

		
		return self.state.copy()

	
	def _compDist(self, x1, y1, x2, y2):
		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def getAttState(self):
		return self.state[-2:].copy()

	def _tanhOut2distance(self, out):

		# TODO: hard code the new range, fix this
		# https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
		return (out + 1) * 40 / 2 + (-20)

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

	def _attMove_straight(self):
		self.att_state[0] = np.clip(self.att_state[0] + 5, 0, 500)
		self.att_state[1] = np.clip(self.att_state[1] + 5, 0, 500)

	def _attMove_zigzag(self):

		if self.att_up_steps == 15:
			self.att_state[0] = np.clip(self.att_state[0] + 20, 0, 500)
			self.att_right_steps += 1
		elif self.att_right_steps == 15:
			self.att_state[1] = np.clip(self.att_state[1] + 20, 0, 500)
			self.att_up_steps += 1
		else:
			if np.random.randint(0,2) == 0:
				self.att_state[0] = np.clip(self.att_state[0] + 20, 0, 500)
				self.att_right_steps += 1
			else:
				self.att_state[1] = np.clip(self.att_state[1] + 20, 0, 500)
				self.att_up_steps += 1


	def step(self, actions):

		self.num_steps += 1




	def def_step(self, action):

		action[0] = np.clip(self._tanhOut2distance(action[0]), -20, 20)
		action[1] = np.clip(self._tanhOut2distance(action[1]), -20, 20)
		action[2] = np.clip(self._tanhOut2distance(action[2]), -20, 20)
		action[3] = np.clip(self._tanhOut2distance(action[3]), -20, 20)

		defDeltax = action[0]
		defDeltay = action[1]

		uavDeltax = action[2]
		uavDeltay = action[3]

		self.state[0] = np.clip(self.state[0] + defDeltax, 0, 500)
		self.state[1] = np.clip(self.state[1] + defDeltay, 0, 500)

		self.state[2] = np.clip(self.state[2] + uavDeltax, 0, 500)
		self.state[3] = np.clip(self.state[3] + uavDeltay, 0, 500)


	def att_step(self, action):

		action[0] = np.clip(self._tanhOut2distance(action[0]), -20, 20)
		action[1] = np.clip(self._tanhOut2distance(action[1]), -20, 20)

		attDeltax = action[0]
		attDeltay = action[1]

		self.state[-2] = np.clip(self.state[-2] + attDeltax, 0, 500)
		self.state[-1] = np.clip(self.state[-1] + attDeltay, 0, 500)




	def step(self, action):
		
		self.num_steps += 1

		# ty: this is hard coded, change this
		self.def_step(action[:4])
		self.att_step(action[4:6])		
		
		
		# compute distance metrics
		defAttDist = self._compDist(self.state[0], self.state[1], self.state[-2], self.state[-1])
		uavAttDist = self._compDist(self.state[2], self.state[3], self.state[-2], self.state[-1])
		tarAttDist = self._compDist(self.state[5], self.state[6], self.state[-2], self.state[-1])
		# reward shaping
		# tarDefDist = self._compDist(self.state[5], self.state[6], self.state[0], self.state[1])


		# if uav found attacker
		if uavAttDist < 10:
			r = 0.0 # should not set this to be greater
			self.state[2] = self.state[-2]
			self.state[3] = self.state[-1]
			self.state[4] = 1
			

		# if defender caught attacker
		if defAttDist < 15:
			r = 10.0
			done = True
			info = {"done": "attacker caught"}
		# if attacker completes the attack
		elif tarAttDist < 10:
			r = -10.0
			done = True
			info = {"done": "target attacked"}
		# if maximum steps reached
		elif self.num_steps >= 50:
			r = -10.0
			done = True
			info = {"done": "max steps reached"}
		elif self._isDefAtBoundary():
			r = -10.0
			done = True
			info = {"done": "def out of boundary"}
		elif self._isAttAtBoundary():
			r = 10.0
			done = True
			info = {"done": "att out of boundary"}
		# elif tarDefDist < 10:
		# 	r = 2.0
		# 	done = False
		# 	info = {"done":"attacker caught : reached target"}
		# elif tarDefDist < self.lastDefTarDist:
		# 	self.lastDefTarDist = tarDefDist
		# 	r = 1.0
		# 	done = False
		# 	info = {"done":None}
		else:
			# ty: this means a positive reward for attacker every time step, may not be reasonable
			r = -0.5
			done = False
			info = {"done": None}

		# ty: make changes to r, make it able to return flexible number player reward
		return self.state.copy(), r, done, info


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










