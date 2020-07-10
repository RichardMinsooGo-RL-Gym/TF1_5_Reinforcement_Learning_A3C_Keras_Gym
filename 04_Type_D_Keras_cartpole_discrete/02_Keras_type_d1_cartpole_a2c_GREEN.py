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
from keras.models import Model

# In case of CartPole-v1, maximum length of episode is 500
env_name = 'CartPole-v1'
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
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
    
# This is A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        self.value_size = 1

        # these are hyper parameters for the A2C
        self.discount_factor = 0.99         # decay rate
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        
        self.hidden1, self.hidden2 = 30, 30
        self.ep_trial_step = 500

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()
        
        self.states, self.actions, self.rewards = [], [], []

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
                
        if self.load_model:
            self.actor.load_weights("./save_model/Cartpole_Actor.h5")
            self.critic.load_weights("./save_model/Cartpole_Critic.h5")
            
    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        state_value = Dense(self.value_size, activation='linear', kernel_initializer='glorot_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

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
        self.states.append(state[0])
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
        policy = self.actor.predict(state, batch_size=1).flatten()
        # 각 액션의 확률로 액션을 결정
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        action_array = np.zeros(self.action_size)
        action_array[action] = 1
        return action

def main():
    # make A2C agent
    agent = A2CAgent(state_size, action_size)
    scores, episodes = [], []
    
    episode = 0
    avg_score = 0
    start_time = time.time()
    
    while time.time() - start_time < 10*60 and avg_score < 495:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        while not done and score < agent.ep_trial_step:
            # fresh env
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            # every time step we do train from the replay memory
            score += 1
            # swap observation
            state = next_state

            if done or score == agent.ep_trial_step:
                # env.reset()
                episode += 1
                agent.train_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(episode)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print("episode :{:>5d}".format(episode), "/ score :{:>5d}".format(score), \
                      "/ last 30 game avg :{:>4.1f}".format(avg_score))

    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/Cartpole_ActorCritc.png")
       
    e = int(time.time() - start_time)
    print('Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    
    agent.actor.save_weights("./save_model/Cartpole_Actor.h5")
    agent.critic.save_weights("./save_model/Cartpole_Critic.h5")
    sys.exit()        

if __name__ == "__main__":
    main()
