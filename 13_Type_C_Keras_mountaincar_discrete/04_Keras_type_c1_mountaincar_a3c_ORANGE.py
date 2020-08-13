import tensorflow as tf
import random
import os
import sys
import gym
import numpy as np
import time
from collections import deque

from keras.layers import *
from keras import backend as K
from keras.models import *
import multiprocessing
import threading

env_name = 'MountainCar-v0'
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
# np.random.seed(123)
# tf.set_random_seed(456)  # reproducible
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
N_WORKERS = multiprocessing.cpu_count()
# N_WORKERS = 8

RUN_TIME = 20*60
n_optimizers = 2

class A3CAgent:
    """
    A shared brain.
    """
    def __init__(self):
        
        self.sess = tf.Session()
        self.c_loss_weight = 0.5
        self.entropy_panelty_weight = 0.01
        
        K.set_session(self.sess)
        K.manual_variable_initialization(True)
        
        self.learning_rate = 5e-3
        self.batch_size = 32
        
        self.epsilon = 0.5

        self.model = self.build_model()
        self._init_op = self._init_op(self.model)
        
        self.training_queue = deque()
        self.training_queue_lock = threading.Lock()

    def build_model(self):

        state = Input( batch_shape=(None, state_size) )
        shared = Dense(units=16, activation='relu')(state)
        policy = Dense(units=action_size, activation='softmax')(shared)
        value = Dense(units=1, activation='linear')(shared)

        model = Model(inputs=[state], outputs=[policy, value])
        
        # have to initialize before threading
        model._make_predict_function()
        
        return model

    def _init_op(self, model):
        state = tf.placeholder(tf.float32, shape=(None, state_size))
        action = tf.placeholder(tf.float32, shape=(None, action_size))
        # not immediate, but discounted n step reward
        q_target = tf.placeholder(tf.float32, shape=(None, 1)) 

        policy, value = model(state)

        # with tf.variable_scope('td_error'):
        # A_t = R_t - V(S_t)
        # td_error = tf.subtract(q_target, value, name='td_error')
        td_error = q_target - value

        # with tf.variable_scope('actor_loss'):
        # Policy loss
        log_p = action * tf.log(tf.clip_by_value(policy,1e-10,1.))
        log_lik = log_p * tf.stop_gradient(td_error)
        actor_loss = -tf.reduce_mean(tf.reduce_sum(log_lik, axis=1))

        # with tf.variable_scope('local_gradients'):
        # entropy(for more exploration)
        entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(tf.clip_by_value(policy,1e-10,1.)), axis=1))

        # with tf.variable_scope('critic_loss'):
        # Value loss
        # critic_loss = tf.reduce_mean(tf.square(td_error))
        critic_loss = tf.reduce_mean(tf.square(value - q_target), axis=1)

        # Total loss
        loss_total = actor_loss + critic_loss - entropy * 0.01

        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
        # train_op = optimizer.minimize(loss_total)
        train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99).minimize(loss_total)

        self.sess.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()  # To avoid future modifications
        
        return state, action, q_target, train_op
    
    def train_model(self):
        if len(self.training_queue) < self.batch_size:
            return False
        with self.training_queue_lock:
            states, actions, rewards, n_discounts, next_states, \
                dones = zip(*self.training_queue)
            self.training_queue.clear()
        states      = np.vstack(states)
        actions     = np.vstack(actions)
        rewards     = np.vstack(rewards)
        next_states = np.vstack(next_states)
        n_discounts = np.vstack(n_discounts)
        dones  = np.vstack(dones)
        
        values = self.predict_value(next_states)
        # set v to 0 where next_states is terminal state
        returns = rewards + n_discounts * values * dones
        
        self.state, self.action, self.q_target, self.train_op = self._init_op
        feed_dict={self.state: states, self.action: actions, self.q_target: returns}
        self.sess.run(self.train_op,feed_dict = feed_dict)
        return True

    def train_push(self, state, action_array, reward, n_discount, next_state):
        if next_state is None:
            self.training_queue.append((state, action_array, reward, n_discount, np.zeros((state_size,)), 0.0))
        else:
            self.training_queue.append((state, action_array, reward, n_discount, next_state, 1.0))

    def predict_policy(self, state):
        with self.default_graph.as_default():
            policy, value = self.model.predict(state)
            return policy

    def predict_value(self, state):
        with self.default_graph.as_default():
            policy, value = self.model.predict(state)
            return value

    def get_action(self, state):
        action_array = np.zeros((action_size,))
        
        if self.epsilon > 0.01:
            self.epsilon -= 0.00002
            
        if random.random() < self.epsilon:
            action = random.randint(0, action_size-1)
            
        else:
            prob_weights = self.predict_policy(state.reshape(1, -1)).ravel()
            action = np.random.choice(range(action_size), p=prob_weights)
        action_array[action] = 1
        return action, action_array

# A global brain
global_agent = A3CAgent()

class Optimizer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.started = False

    def run(self):
        self.started = True
        while self.started:
            if not global_agent.train_model():
                # If list is empty, don't keep trying until timeout, just yield already
                time.sleep(0)

    def stop(self):
        self.started = False

        
episode = 0

class Agent(threading.Thread):
    def __init__(self, render=False, discount_factor=0.99, max_td_steps=8):
        super().__init__()
        self.THREAD_DELAY = 0.001
        self.started = False
        self.render = render
        self.episode_rewards = []
        
        self.discount_factor = discount_factor
        self.max_td_steps = max_td_steps
        self.env = gym.make(env_name).unwrapped
        self.memory = deque()  # used for n_step return
        self.stop_signal = False

    def train_agent(self, state, action_array, reward, next_state):
        self.memory.append((state, action_array, reward, next_state))
        
        if next_state is None:
            # We have reached the end of the episode, flush the memory
            while self.memory:
                state, action_array, _, _ = self.memory[0]
                reward, n_discount, next_state = self.compute_return(self.memory)
                global_agent.train_push(state, action_array, reward, n_discount, next_state)
                self.memory.popleft()
        else:
            if len(self.memory) >= self.max_td_steps:
                # We can compute at least one full n-step return
                state, action_array, _, _ = self.memory[0]
                reward, n_discount, next_state = \
                    self.compute_return([self.memory[i] for i in range(self.max_td_steps)])
                global_agent.train_push(state, action_array, reward, n_discount, next_state)
                self.memory.popleft()

    def compute_return(self, trajectories):
        reward = 0.0
        for k in range(len(trajectories)):
            reward += self.discount_factor**k * trajectories[k][2]
        n_discount = self.discount_factor**len(trajectories)
        next_state = trajectories[-1][3]
        return reward, n_discount, next_state
    
    def work(self):
        global episode
        state = self.env.reset()

        scores = 0
        episodes = []
        
        while True and scores < 10000:
            time.sleep(self.THREAD_DELAY)
            
            if self.render:
                self.env.render()

            # action = self.agent.get_action(state)
            action, action_array = global_agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if done or scores == 10000: # terminal state
                next_state = None

            self.train_agent(state, action_array, reward, next_state)

            state = next_state
            scores += 1

            if done or scores == 10000:
                episode += 1
                break

        print("Episode :",episode,"/ Total Reward :",scores)
        
    def run(self):
        self.started = True
        if self.render:
            self.figure = plt.figure()
            self.plot = self.figure.add_subplot(1, 1, 1)
        while self.started:
            self.work()

    def stop(self):
        self.started = False

agents = [Agent() for i in range(N_WORKERS)]
optimizers = [Optimizer() for i in range(n_optimizers)]

# agents[0].render = True
for agent in agents:
    agent.start()
for optimizer in optimizers:
    optimizer.start()

time.sleep(RUN_TIME)

for agent in agents:
    agent.stop()
for agent in agents:
    agent.join()  # Let the agents finish their episode

for optimizer in optimizers:
    optimizer.stop()
for optimizer in optimizers:
    optimizer.join()

print("Training finished")
# env_test.run()