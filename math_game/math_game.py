
'''
###########################################################################################
This math game is developed by Avin to test the performance of Reinforcement Learning(RL).
Q-table is used in this project.

Game Description:

	Select numbers from a pool of numbers and fill a 4x4 grid such that when the SUM of the
	numbers inside the grid is substituded to the following function, it is minimized!

	f(x) = (x**4/4) - (10*x**3/3) + (27*x**2/2) - (18*x) + 20
###########################################################################################
'''

import numpy as np
import gym
import gym_minimize
import random
import matplotlib.pyplot as plt


env = gym.make("minimize-v0")

action_space = np.array([0.11, 0.17, 0.26, 0.31, 0.39, 0.46, 0.54, 0.75, 0.82, 0.91])

action_size = len(action_space)
state_size = env.NO_OF_SQUARES


qtable = np.zeros((state_size, action_size))

total_episodes = 200000       # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob



# List of rewards
rewards = []

iteration_arr = []

# for i in range(100):
# 	print env.step(np.random.choice(action_space))

for i in range(100):
	print 'iteration: ', i
	state, state_pos = env.reset()

	for episode in range(total_episodes):

		#first we reset the environment
		# state, state_pos = env.reset()
		# print 'state_pos', state_pos
		step = 0
		done = 0
		total_rewards = 0

		# for step in range(max_steps):

		#randomly choose an action
		tradeoff = random.uniform(0,1)


		## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
		if tradeoff > epsilon:
			action = action_space[np.argmax(qtable[state_pos,:])]

		# Else doing a random choice --> exploration
		else:
			action = np.random.choice(action_space)
			# print 'random action'

		# Take the action and observe the outcome state(s') and reward (r)
		new_state, new_pos, reward, function_val = env.step(action)
		# print new_state, new_pos, reward, function_val

		# Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
		# qtable[new_state,:] : all the actions we can take from new state
		action_pos = np.argwhere(action_space==action).item()
		qtable[state_pos, action_pos] = qtable[state_pos, action_pos] + learning_rate * (reward + gamma * np.max(qtable[new_pos,:]) - qtable[state_pos, action_pos])
		
		total_rewards += reward

		state = new_state
		state_pos = new_pos

		# Reduce epsilon (because we need less and less exploration)
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
		rewards.append(total_rewards)
		# env.render()

	print ("Score over time: " +  str(sum(rewards)/total_episodes))
	print(qtable)


	state, state_pos = env.reset()


	for episode in range(20):

		# step = 0

		print ('*********************************************')
		print 'EPISODE', episode

		# for step in range(max_steps):

		action = action_space[np.argmax(qtable[state_pos,:])]

		new_state, new_pos, reward, function_val=env.step(action)

		env.render()
		state = new_state
		state_pos = new_pos

	print ''
	print ''
	print '################## FINAL STATE ##############'

	env.render()

	print '#################### RESULTS ################'

	print 'Global Minimum (y)    = ', 2.0, 'at x = ', 6.0
	print 'Predicted minimum (y) = ', function_val, 'at x = ', sum(sum(np.array(state)))

	iteration_arr.append(function_val)

iteration = [i for i in range(100)]
plt.plot(iteration, [2.0 for i in range(100)], label='Global Minimum')
plt.plot(iteration, iteration_arr, label='Minimum Found By RL Agent')
plt.title('Reinforcement Learning Agent Performance')
plt.xlabel('Iteration No.')
plt.ylabel('Predicted/Global Minimum')
plt.legend()
plt.savefig('RL_agent_performance')
plt.show()