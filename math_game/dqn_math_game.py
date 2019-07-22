
'''
###########################################################################################
This math game is developed by Avin to test the performance of Reinforcement Learning(RL).
A DQN is used in this project

Game Description:

	Select numbers from a pool of numbers and fill a 3x3 grid such that when the SUM of the
	numbers inside the grid is substituded to the following function, it is minimized!

	f(x) = (x**4/4) - (10*x**3/3) + (27*x**2/2) - (18*x) + 20
###########################################################################################
'''

import numpy as np
import gym
import gym_minimize
import random
import tensorflow as tf
import tensorflow.contrib.layers as lays
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import model_from_json
import matplotlib.pyplot as plt
from collections import deque

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


env = gym.make("minimize-v0")

action_space = np.array([0.01, 0.11, 0.17, 0.26, 0.31, 0.39, 0.46, 0.54, 0.75, 20.84, 20.89])

action_size = len(action_space)
state_size = env.NO_OF_SQUARES


total_episodes = 64000      # Total episodes
learning_rate = 0.008           # Learning rate
max_steps = 99                # Max steps per episode
batch_size = 64
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.001            # Exponential decay rate for exploration prob

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False




# List of rewards
rewards = []


# model = Sequential()

# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(48))
# model.add(Activation('relu'))
# model.add(Dense(action_size))
# model.add(Activation('linear'))

# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
# model.summary()

class DQNetwork:
	def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		
		with tf.variable_scope(name):
			# We create the placeholders
			# *state_size means that we take each elements of state_size in tuple hence is like if we wrote
			# [None, 84, 84, 4]
			self.inputs_ = tf.placeholder(tf.float32, [None, 16], name="inputs")
			self.actions_ = tf.placeholder(tf.float32, [None, 11], name="actions_")
			
			# Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
			self.target_Q = tf.placeholder(tf.float32, [None], name="target")

			self.dense1 = tf.layers.dense(self.inputs_, 128, activation=tf.nn.relu)
			self.dense2 = tf.layers.dense(self.dense1, 256, activation=tf.nn.relu)
			self.output = tf.layers.dense(self.dense2, action_size)

			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

			self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
					
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)
# memory = SequentialMemory(limit=1000000, window_length=4)

# memory.append([1,2,3,4], 0.23, 0.004, 1)

# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
# nb_steps=1000000)


# dqn = DQNAgent(model=DQN_model(), nb_actions=action_size, policy=policy, memory=memory,
#                processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
#                train_interval=4, delta_clip=1.)
# dqn.compile(Adam(lr=.00025), metrics=['mae'])


class Memory():

	def __init__(self, max_memory_size):
		self.buffer = deque(maxlen=max_memory_size)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		buffer_size = len(self.buffer)
		index = np.random.choice(np.arange(buffer_size),
								size = batch_size,
								replace = False)
		
		return [self.buffer[i] for i in index]


# Following snippet is to deal with the empty memory problem
# Randomly take an action and store the experience tuples
state, state_pos = env.reset()
# Instantiate memory
memory = Memory(max_memory_size = memory_size)
# for j in range(10):
for i in range(pretrain_length):
	
	#select a random action
	action=np.random.choice(action_space)
	#observe the reward and new state
	new_state, new_pos, reward, function_val = env.step(action)
	# reward_arr = np.zeros(11)
	# reward_arr[np.argwhere(action_space==action).item()] = reward
	#add the experience to memory
	memory.add((np.array(state).flatten(), action, reward, np.array(new_state).flatten(), function_val))

	state = np.array(new_state).flatten()
	state_pos = new_pos


#####Training the Agent########

# Function to predict an action
# It either randomly selects an action or select the action with the max Q value
# Depending on the epsilon
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
	## EPSILON GREEDY STRATEGY
	# Choose action a from state s using epsilon greedy.
	## First we randomize a number
	exp_exp_tradeoff = np.random.rand()

	# Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
	explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
	
	if (explore_probability > exp_exp_tradeoff):
		# Make a random action (exploration)
		action = random.choice(action_space)
		# print explore_probability, exp_exp_tradeoff
		print 'exploring#########'
		
	else:
		# Get action from Q-network (exploitation)
		# Estimate the Q values for current state
		# Qvalues = model.predict(np.array(state_pos).reshape(1,1))
		Qvalues = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: np.array(np.array(state).flatten()).reshape(1,16)})
		# print state_pos
		# print Qvalues
		# Take the biggest Q value (= the best action)
		# print Qvalues
		choice = np.argmax(Qvalues)
		print 'choice',choice
		action = action_space[int(choice)]
				
	return action, explore_probability


##Now train the Agent
saver = tf.train.Saver()

f = open('outfile','a+')

for i in range(100):
	if training == True:

		with tf.Session() as sess:
			# Initialize the variables
			sess.run(tf.global_variables_initializer())

			# Initialize the decay rate (that will use to reduce epsilon) 
			decay_step = 0

			for episode in range(total_episodes):

				print episode

				episode_rewards = []
				# Predict the action to take and take it
				action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, np.array(state).flatten(), action_space)
				new_state, new_pos, reward, function_val = env.step(action)
				# reward_arr = np.zeros(11)
				# reward_arr[np.argwhere(action_space==action).item()] = reward
				memory.add((np.array(state).flatten(), action, reward, np.array(new_state).flatten(), function_val))

				state = np.array(new_state).flatten()
				state_pos = new_pos

				decay_step +=1


				### LEARNING PART            
				# Obtain random mini-batch from memory
				batch = memory.sample(batch_size)
				# print 'Memory Batch -----',batch
				# print 'memory#################'
				# print batch
				states_mb = np.array([each[0] for each in batch])
				actions_mb = np.array([each[1] for each in batch])
				# print actions_mb
				actions_mb_arr = np.zeros((len(actions_mb),action_size))
				for i in range(len(actions_mb)):
					action_position = np.argwhere(action_space==actions_mb[i]).item()
					actions_mb_arr[i][action_position] = 1
				# print actions_mb_arr
				rewards_mb = np.array([each[2] for each in batch])
				# print rewards_mb 
				next_states_mb = np.array([each[3] for each in batch])
				function_vals_mb = np.array([each[4] for each in batch])

				target_Qvalues_batch = []
				
				# Get Q values for next_state 
				# Qvalues_next_state = model.predict(np.array(next_states_mb).reshape(len(states_mb),1))
				Qvalues_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: np.array(next_states_mb).reshape(len(states_mb),16)})
				# print Qvalues_next_state
				# print np.array(Qvalues_next_state).shape
				# Set Q_target = r + gamma*maxQ(s', a')
				for i in range(0, len(batch)):

					target = rewards_mb[i] + gamma * np.max(Qvalues_next_state[i])
					# print target
					target_Qvalues_batch.append(target)

					targets_mb = np.array([each for each in target_Qvalues_batch])
					# print targets_mb.shape
					# break

				# history = model.fit(states_mb, targets_mb, epochs=1)

				loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
											feed_dict={DQNetwork.inputs_: np.array(states_mb).reshape(len(states_mb),16),
													   DQNetwork.target_Q: targets_mb,
													   DQNetwork.actions_: actions_mb_arr})
				print 'Loss......', loss


				# Save model every 5 episodes
				if episode % 5 == 0:
					save_path = saver.save(sess, "./models/model.ckpt")
					print("Model Saved")
				# if (episode % 5 == 0):
				# 	# serialize model to JSON
				# 	model_json = model.to_json()
				# 	with open("model.json", "w") as json_file:
				# 		json_file.write(model_json)
				# 	# serialize weights to HDF5
				# 	model.save_weights("model.h5")
				# 	print("Saved model to disk")



	# Watch the agent solving math problem

	# # load json and create model
	# json_file = open('model.json', 'r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# # load weights into new model
	# loaded_model.load_weights("model.h5")
	# print("Loaded model from disk")


	# # evaluate loaded model on test data
	# loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

	# print 'prediction', loaded_model.predict(np.array([[1]])).shape
	# # score = loaded_model.evaluate(X, Y, verbose=0)
	# # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


	state, state_pos = env.reset()
	#for random initialization of state
	for i in range(16):
		action=np.random.choice(action_space)
		new_state, new_pos, reward, function_val = env.step(action)

	with tf.Session() as sess:
		# Load the model
		saver.restore(sess, "./models/model.ckpt")

		for episode in range(100):

			# step = 0

			print ('*********************************************')
			print 'EPISODE', episode

			# for step in range(max_steps):
			Qvalues = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: np.array(state).flatten().reshape(1,16)})
			print 'Final Qvalues', Qvalues
			action = action_space[np.argmax(Qvalues)]

			new_state, new_pos, reward, function_val=env.step(action)
			print new_pos, reward

			env.render()
			state = np.array(new_state).flatten()
			state_pos = new_pos

	print ''
	print ''
	print '################## FINAL STATE ##############'

	env.render()

	print '#################### RESULTS ################'

	print 'Global Minimum (y)    = ', 2.0, 'at x = ', 6.0 
	f.write('Predicted minimum (y) = '+ str(function_val) + 'at x = '+ str(np.sum(np.sum(state)))+'\n')
	print 'Predicted minimum (y) = ', function_val, 'at x = ', np.sum(np.sum(state))












# # later...
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")