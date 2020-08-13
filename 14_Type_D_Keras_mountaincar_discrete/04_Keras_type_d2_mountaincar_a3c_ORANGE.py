import tensorflow as tf
import random
import os
import sys
import gym
import pylab
import numpy as np
import time
from collections import deque
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, Sequential
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

# This is A3C(Asychronous Advantage Actor-Critic) agent for the Cartpole
class A3CAgent:
    def __init__(self, state_size, action_size, env_name):

        # get gym environment name
        self.env_name = env_name
        
        # if you want to see Cartpole learning, then change to True
        self.load_model = False
        
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        self.value_size = 1

        # these are hyper parameters for the A2C
        
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        
        self.hidden1, self.hidden2 = 30, 30
        self.ep_trial_step = 2000
        
        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        
        if self.load_model:
            self.actor.load_weights("./save_model/Cartpole_Actor.h5")
            self.critic.load_weights("./save_model/Cartpole_Critic.h5")
            
    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        # actor.add(Dense(30, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        actor._make_predict_function()

        actor.summary()

        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        critic.add(Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='glorot_uniform'))
        critic._make_predict_function()
        critic.summary()
        return critic
    # make loss function for Policy Gradient
    # [log(action probability) * discounted_rewards] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Policy Gradient 의 핵심
        # log(정책) * return 의 gradient 를 구해서 최대화시킴
        policy = self.actor.output
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * K.stop_gradient(discounted_rewards)
        loss = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        # create training function
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, discounted_rewards], [], updates=updates)
        return train

    # create error function and training function to update value network
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    # make agents(local) and start training
    def train(self):
        # self.load_model('./save_model/Mountaincar_a3c.h5')
        agents = [Worker(i, self.actor, self.critic, self.optimizer, self.env_name, 
                        self.action_size, self.state_size) for i in range(N_WORKERS)]

        for agent in agents:
            time.sleep(1)
            agent.start()
            
# This is Worker(local) class for threading
class Worker(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, action_size, state_size):
        threading.Thread.__init__(self)
        self.discount_factor = 0.99         # decay rate
        self.states, self.actions, self.rewards = [], [], []
        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.action_size = action_size
        self.render = False        

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_and_norm_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy network and value network every episode
    def train_model(self):
        discounted_rewards = self.discount_and_norm_rewards(self.rewards)
        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        discounted_rewards = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, discounted_rewards])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, state_size]))[0]
        
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    # Thread interactive with environment
    def run(self):
        global episode
        env = gym.make(self.env_name).unwrapped
        scores, episodes = [], []
        episode = 0
        avg_score = 10000
        start_time = time.time()
        
        while time.time() - start_time < 20*60 and avg_score > 200:
            
            done = False
            score = 0
            state = env.reset()
            
            while not done and score < 10000:
                # fresh env
                if self.render:
                    env.render()

                # get action for the current state and go one step in environment
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # save the sample <s, a, r> to the memory
                self.append_sample(state, action, reward)
                
                # every time step we do train from the replay memory
                score += 1
                # swap observation
                state = next_state

                if done or score == 10000:
                    episode += 1
                    env.reset()
                    self.train_model()
                    
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                    print("episode :", episode, "/ score :", score, "last 30 gme avg :", avg_score)

        pylab.plot(episodes, scores, 'b')
        pylab.savefig("./save_graph/Mountaincar_PG.png")
        self.actor.save_weights("./save_model/Mountaincar_Actor.h5")
        self.critic.save_weights("./save_model/Mountaincar_Critic.h5")
            
if __name__ == "__main__":
    global_agent = A3CAgent(state_size, action_size, env_name)
    global_agent.train()
