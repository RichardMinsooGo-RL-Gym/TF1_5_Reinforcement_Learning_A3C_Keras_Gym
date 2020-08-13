import tensorflow as tf
import random
import os
import sys
import gym
import pylab
import numpy as np
import time
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model

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
        self.model_lr = 0.005
        
        self.hidden1, self.hidden2 = 128, 128
        self.ep_trial_step = 15000

        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []

        # create model for actor and critic network
        self.model = self.build_model()
        self.optimizer = self.model_optimizer()
        
    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        self.state = Input(batch_shape=[None, self.state_size])
        self.action = K.placeholder(shape=[None, self.action_size])
        self.q_target = K.placeholder(shape=[None, ])

        actor_hidden = Dense(self.hidden1, activation='tanh', kernel_initializer='glorot_uniform')(self.state)
        actor_predict = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        critic_hidden = Dense(self.hidden1, activation='tanh', kernel_initializer='glorot_uniform')(self.state)
        critic_predict = Dense(self.value_size, activation='linear', kernel_initializer='glorot_uniform')(critic_hidden)

        model = Model(inputs=self.state, outputs=[actor_predict, critic_predict])

        model._make_predict_function()
        model.summary()

        return model

    # make loss function for Policy Gradient
    # [log(action probability) * q_target] will be input for the back prop
    # we add entropy of action probability to loss
    
    def model_optimizer(self):
        self.policy, self.value = self.model.output
        # Core of Actor Critic
        # A_t = R_t - V(S_t)
        self.td_error = self.q_target - self.value
        # Value loss
        self.critic_loss = K.mean(K.square(self.td_error))
        
        # Policy loss
        actor_predict = K.sum(self.action * self.policy, axis=1)
        cross_entropy = K.log(actor_predict + 1e-10) * K.stop_gradient(self.q_target)
        self.actor_loss = -K.sum(cross_entropy)

        # entropy(for more exploration)
        self.entropy = K.sum(self.policy * K.log(self.policy + 1e-10), axis=1)

        # Total loss
        self.total_loss = self.actor_loss + self.critic_loss - self.entropy * 0.01

        optimizer = Adam(lr=self.model_lr)
        updates = optimizer.get_updates(self.model.trainable_weights, [], self.total_loss)
        train_op = K.function([self.model.input, self.action, self.q_target], [], updates=updates)

        return train_op    
        
    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_and_norm_rewards(self, buffer_reward):
        discounted_rewards = np.zeros_like(buffer_reward)
        running_add = 0
        for index in reversed(range(0, len(buffer_reward))):
            running_add = running_add * self.discount_factor + buffer_reward[index]
            discounted_rewards[index] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def append_sample(self, state, action, reward):
        
        action_array = np.zeros(self.action_size)
        action_array[action] = 1
        self.buffer_state.append(state[0])
        self.buffer_action.append(action_array)
        self.buffer_reward.append(reward)

    # update policy network and value network every episode
    def train_model(self):
        discounted_rewards = self.discount_and_norm_rewards(self.buffer_reward)
        values = self.model.predict(np.array(self.buffer_state))[1]
        values = np.reshape(values, len(values))

        discounted_rewards = discounted_rewards - values

        self.optimizer([self.buffer_state, self.buffer_action, discounted_rewards])
        
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        # Run forward propagation to get softmax probabilities
        policy = self.model.predict(state, batch_size=1)[0].flatten()
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

def main():
    # make A2C agent
    agent = A2CAgent(state_size, action_size)
    scores, episodes = [], []
    
    episode = 0
    avg_score = agent.ep_trial_step
    start_time = time.time()
    
    while time.time() - start_time < 30*60 and avg_score > 200:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        while not done and score < agent.ep_trial_step:
            # every time step we do train from the replay memory
            score += 1

            # fresh env
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            # make step in environment
            next_state, reward, done, _ = env.step(action) 
            next_state = np.reshape(next_state, [1, agent.state_size])

            # save the sample <state, action, reward> to the memory
            agent.append_sample(state, action, reward)

            # if train_steps % 10 == 0 or done:   # update global and assign to local net
            #     agent.train_model()

            # swap observation
            state = next_state

            # train when epsisode finished
            if done or score == agent.ep_trial_step:
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
    sys.exit()

if __name__ == "__main__":
    main()
