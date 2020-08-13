import os
import sys
import gym
import pylab
import numpy as np
import time
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

env_name = 'MountainCar-v0'
env = gym.make(env_name)
# env.seed(1)     # reproducible, general Policy gradient has high variance
# np.random.seed(123)
# tf.set_random_seed(456)  # reproducible
env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model_path = os.path.join(os.getcwd(), 'save_model')
graph_path = os.path.join(os.getcwd(), 'save_graph')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
# This is PolicyGradient agent for the Cartpole
class PolicyGradient:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        
        # get size of state and action
        self.action_size = action_size
        self.state_size = state_size
        # these is hyper parameters for the PolicyGradient
        self.discount_factor = 0.99         # decay rate
        self.actor_lr = 0.005
        self.ep_trial_step = 10000

        self.hidden1, self.hidden2 = 24, 24
        
        # create model for policy network
        self.actor = self.build_model()
        self.optimizer = self.optimizer()
        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.actor.load_weights('./save_model/Cartpole_PG.h5')
            
    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        actor = Sequential()
        actor.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        actor.summary()

        return actor

    # create error function and training function to update policy network
    def optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Policy Gradient 의 핵심
        # log(정책) * return 의 gradient 를 구해서 최대화시킴
        policy = self.actor.output
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, discounted_rewards], [], updates=updates)
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
        action_array = np.zeros(self.action_size)
        action_array[action] = 1
        self.actions.append(action_array)

    # update policy network and value network every episode
    def train_model(self):
        discounted_rewards = self.discount_and_norm_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        # 각 액션의 확률로 액션을 결정
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

def main():
    # make A2C agent
    agent = PolicyGradient(state_size, action_size)
    scores, episodes = [], []
    
    episode = 0
    avg_score = agent.ep_trial_step
    start_time = time.time()
    
    # while episode < 200:
    while time.time() - start_time < 20*60 and avg_score > 200:
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
    pylab.savefig("./save_graph/Cartpole_PG.png")
       
    e = int(time.time() - start_time)
    print('Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    
    agent.actor.save_weights("./save_model/Cartpole_Actor.h5")
    sys.exit()        

if __name__ == "__main__":
    main()
