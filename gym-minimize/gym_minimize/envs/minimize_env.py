import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class MinimizeEnv(gym.Env):
	
	metadata = {'render.modes': ['human']}

	def __init__(self):

		self.state = [[ 0 for j in range(4)] for i in range(4)] #this is our grid, 0 denotes empty cell
		self.NO_OF_SQUARES = 16
		self.total_x = 0
		self.function_val = 0
		self.prev_function_val = 0
		self.return_pos = 0
		self.position = 0
		self.done = 0
		self.selected = []
		self.reward = 0
		
	def function(self,x):

		return (x**4/4) - (10*x**3/3) + (27*x**2/2) - (18*x) + 20


	def substitute(self):

		#calculate total sum here. this is the x, to be substituded to our function
		self.total_x = sum(sum(np.array(self.state)))
		self.function_val = self.function(self.total_x)

			# return self.function_val



	def step(self, action):
		
		# if (self.done == 1):
		# 	print 'Grid filling completed!'
		# 	return [self.state, self.reward, self.done, self.function_val]

		# elif self.state[int(self.position/3)][self.position%3] != "_":
		# 	print 'This cell is already filled!'
		# 	return [self.state, self.reward, self.done, self.function_val]

		# print self.position
		self.state[int(self.position/4)][self.position%4] = action
		self.prev_function_val = self.function_val
		self.substitute() #now function_val should have the total sum of the grid

		#we don't need a reward until the whole grid is filled
		if (any(0 in x for x in self.state)):
			self.reward = 0
			self.position += 1
			self.return_pos = self.position

			if (self.position == self.NO_OF_SQUARES):
				self.position = 0
				self.return_pos = self.position

			return [self.state, self.return_pos, self.reward, self.function_val]


		else:
			#if the function value is high, our reward would be low and vice versa
			#Hence reward=1/function_val
			if (self.function_val > self.prev_function_val*4):
				self.reward = -100
				# print self.prev_function_val, self.function_val, self.reward
			else:
				self.reward = 1.0*100/self.function_val
				# print self.prev_function_val, self.function_val, self.reward
			self.position += 1
			self.return_pos = self.position

			if (self.position == self.NO_OF_SQUARES):
				self.position = 0
				self.return_pos = self.position

			return [self.state, self.return_pos, self.reward, self.function_val]
		

		self.render()



	def reset(self):
		
		self.state = [[ 0 for j in range(4)] for i in range(4)] #this is our grid, '_' denotes empty cell
		self.NO_OF_SQUARES = 16
		self.total_x = 0
		self.function_val = 0
		self.prev_function_val = 0
		self.return_pos = 0
		self.position = 0
		self.done = 0
		self.selected = []
		self.reward = 0

		return self.state, self.return_pos

	def render(self):
		print ''
		print ''
		for i in range(4):
			for j in range(4):
				print self.state[i][j], ' | ',
			print ""







































# print Minimize_Function.__init__()
# print state

# Math_env().__init__()

# ins.__init__()