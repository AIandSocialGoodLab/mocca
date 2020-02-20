import gym
import numpy as np

MAX_STEPS = 50


class CatcherEnv(gym.Env):

	def __init__(self):
		# defender x, y, uav x, y, find or not, target 1 x, y
		self.observation_space = gym.spaces.Box(low=np.array([0,0,0,0,0,0,0],dtype=np.float32),high=np.array([500,500,500,500,1,500,500],dtype=np.float32))
		# defender direction, speed, uav direction, speed
		self.action_space = gym.spaces.Box(low=np.array([-20,-20,-20,-20],dtype=np.float32), high=np.array([20, 20, 20, 20],dtype=np.float32))
		# attacker x, y
		self.attacker_state = np.array([100,100], dtype=np.float32)

		self.state = None

	def reset(self):
		
		self.state = np.array([250,250,250,250,0,400,400],dtype=np.float32)
		self.attacker_state = np.array([100,100], dtype=np.float32)
		self.num_steps = 0

		self.lastDefTarDist = self._compDist(self.state[5], self.state[6], self.state[0], self.state[1])

		self.att_up_steps = 0
		self.att_right_steps = 0

		return self.state.copy()

	
	def _compDist(self, x1, y1, x2, y2):
		return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

	def getAttState(self):
		return self.attacker_state.copy()

	def _tanhOut2distance(self, out):

		# TODO: hard code the new range, fix this
		# https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
		return (out + 1) * 40 / 2 + (-20)

	def _isAtBoundary(self):

		if (self.state[0] == 0 or self.state[0] == 500 or
			self.state[1] == 0 or self.state[1] == 500 or 
			self.state[2] == 0 or self.state[2] == 500 or 
			self.state[3] == 0 or self.state[3] == 500):
			return True
		else:
			return False

	def _attMove_straight(self):
		self.attacker_state[0] = np.clip(self.attacker_state[0] + 5, 0, 500)
		self.attacker_state[1] = np.clip(self.attacker_state[1] + 5, 0, 500)

	def _attMove_zigzag(self):

		if self.att_up_steps == 15:
			self.attacker_state[0] = np.clip(self.attacker_state[0] + 20, 0, 500)
			self.att_right_steps += 1
		elif self.att_right_steps == 15:
			self.attacker_state[1] = np.clip(self.attacker_state[1] + 20, 0, 500)
			self.att_up_steps += 1
		else:
			if np.random.randint(0,2) == 0:
				self.attacker_state[0] = np.clip(self.attacker_state[0] + 20, 0, 500)
				self.att_right_steps += 1
			else:
				self.attacker_state[1] = np.clip(self.attacker_state[1] + 20, 0, 500)
				self.att_up_steps += 1


	def step(self, action):
		
		self.num_steps += 1

		oldTarDefDist = self._compDist(self.state[5], self.state[6], self.state[0], self.state[1])
		
		# make scale conversions tanh to our case
		# TODO: vectorize this
		action[0] = np.clip(self._tanhOut2distance(action[0]), -20, 20)
		action[1] = np.clip(self._tanhOut2distance(action[1]), -20, 20)
		action[2] = np.clip(self._tanhOut2distance(action[2]), -20, 20)
		action[3] = np.clip(self._tanhOut2distance(action[3]), -20, 20)

		
		# TODO: vectorize this
		defDeltax = action[0]
		defDeltay = action[1]

		uavDeltax = action[2]
		uavDeltay = action[3]

		self.state[0] = np.clip(self.state[0] + defDeltax, 0, 500)
		self.state[1] = np.clip(self.state[1] + defDeltay, 0, 500)

		self.state[2] = np.clip(self.state[2] + uavDeltax, 0, 500)
		self.state[3] = np.clip(self.state[3] + uavDeltay, 0, 500)

		# attacker make move
		# self._attMove_straight()
		self._attMove_zigzag()
		


		# compute distance metrics
		defAttDist = self._compDist(self.state[0], self.state[1], self.attacker_state[0], self.attacker_state[1])
		uavAttDist = self._compDist(self.state[2], self.state[3], self.attacker_state[0], self.attacker_state[1])
		tarAttDist = self._compDist(self.state[5], self.state[6], self.attacker_state[0], self.attacker_state[1])
		# reward shaping
		tarDefDist = self._compDist(self.state[5], self.state[6], self.state[0], self.state[1])


		# if uav found attacker
		if uavAttDist < 10:
			r = 0.0 # should not set this to be greater
			self.state[2] = self.attacker_state[0]
			self.state[3] = self.attacker_state[1]
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
		elif self._isAtBoundary():
			r = -10.0
			done = True
			info = {"done": "out of boundary"}
		elif tarDefDist < 10:
			r = 2.0
			done = False
			info = {"done":"attacker caught : reached target"}
		elif tarDefDist < self.lastDefTarDist:
			self.lastDefTarDist = tarDefDist
			r = 1.0
			done = False
			info = {"done":None}
		else:
			r = -0.5
			done = False
			info = {"done": None}

		return self.state.copy(), r, done, info












