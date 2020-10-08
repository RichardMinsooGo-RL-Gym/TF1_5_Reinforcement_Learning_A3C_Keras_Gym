import numpy as np
import time, datetime
import gym
import pylab
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from collections import deque
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model

import tensorflow as tf

env_name = "CartPole-v1"
# set environment
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

game_name =  sys.argv[0][:-3]

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)
    
# This is RL_Agent agent for the Cartpole
class RL_Agent:
    def __init__(self, state_size, action_size, env_name):

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # these is hyper parameters for the RL_agent
        self.model_lr = 0.005
        
        self.hidden1, self.hidden2 = 64, 64
        
        # create model for policy network
        self.actor, self.critic, self.model = self.build_model()
        
        # method for training actor and critic network
        self.loss_and_train = self.loss_and_train()

        self.discount_factor = 0.99         # decay rate
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []
        self.ep_trial_step = 500
        self.render = False
        self.training_time = 5*60

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=[None, self.state_size])

        actor_hidden = Dense(self.hidden1, activation='tanh', kernel_initializer='glorot_uniform')(state)
        actor_predict = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        actor = Model(inputs=state, outputs=actor_predict)
        # actor._make_predict_function()
        actor.summary()

        critic_hidden = Dense(self.hidden1, activation='tanh', kernel_initializer='glorot_uniform')(state)
        critic_predict = Dense(self.value_size, activation='linear', kernel_initializer='glorot_uniform')(critic_hidden)
        critic = Model(inputs=state, outputs=critic_predict)
        # critic._make_predict_function()
        critic.summary()
        
        model = Model(inputs=state, outputs=[actor_predict, critic_predict])
        model.summary()
        
        return actor, critic, model

    # make loss function for Policy Gradient
    # [log(action probability) * q_target] will be input for the back prop
    # we add entropy of action probability to loss
    
    def loss_and_train(self):
        action = K.placeholder(shape=[None, self.action_size])
        q_target = K.placeholder(shape=[None, ])

        policy, value = self.model.output
        # Core of Actor Critic
        # A_t = R_t - V(S_t)
        # td_error = tf.subtract(q_target, value, name='td_error')
        td_error = q_target - value

        # Policy loss
        log_p = K.sum(action * policy, axis=1)
        log_lik = K.log(log_p + 1e-10) * K.stop_gradient(q_target)
        actor_loss = -K.sum(log_lik)
        
        # entropy(for more exploration)
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        # Value loss
        critic_loss = K.mean(K.square(td_error))
        # Total loss
        loss_total = actor_loss + critic_loss - entropy * 0.01

        # create training function
        optimizer = Adam(lr=self.model_lr)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss_total)
        train_op = K.function([self.model.input, action, q_target], [], updates=updates)

        return train_op

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
        # discounted_rewards = self.discount_and_norm_rewards(self.buffer_reward)
        
        discounted_rewards = np.zeros_like(self.buffer_reward)
        running_add = 0
        for index in reversed(range(0, len(self.buffer_reward))):
            running_add = running_add * self.discount_factor + self.buffer_reward[index]
            discounted_rewards[index] = running_add
        
        values = self.model.predict(np.array(self.buffer_state))[1]
        values = np.reshape(values, len(values))

        discounted_rewards = discounted_rewards - values

        self.loss_and_train([self.buffer_state, self.buffer_action, discounted_rewards])
        self.buffer_state, self.buffer_action, self.buffer_reward = [], [], []

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        # Run forward propagation to get softmax probabilities
        policy = self.model.predict(state, batch_size=1)[0].flatten()
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def run(self):
        scores, episodes = [], []
        avg_score = 0

        episode = 0
        time_step = 0
        # start training    
        # Step 3.2: run the game
        start_time = time.time()

        while time.time() - start_time < self.training_time and avg_score < 490:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done and score < self.ep_trial_step:
                # every time step we do train from the replay memory
                score += 1
                time_step += 1

                # fresh env
                if self.render:
                    env.render()
                
                # Select action_arr
                action = self.get_action(state)

                # run the selected action_arr and observe next state and reward
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                
                # It is specific for cartpole.
                if done:
                    reward = -100

                # save the sample <state, action, reward> to the memory
                self.append_sample(state, action, reward)

                # update the old values
                state = next_state

                # train when epsisode finished
                if done or score == self.ep_trial_step:
                    episode += 1
                    self.train_model()

                    # every episode, plot the play time
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                    print("episode :{:>5d}".format(episode), "/ score :{:>5.0f}".format(score), \
                          "/ last 30 game avg :{:>4.1f}".format(avg_score))

        # pylab.plot(episodes, scores, 'b')
        # pylab.savefig("./save_graph/Cartpole_ActorCritc.png")

        e = int(time.time() - start_time)
        print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        # sys.exit()

if __name__ == "__main__":
    global_agent = RL_Agent(state_size, action_size, "network")
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    
    # if actor exists, restore actor
    if os.path.isfile("./save_model/Cartpole_Actor.h5"):
        global_agent.actor.load_weights("./save_model/Cartpole_Actor.h5")
        print(" Actor restored!!")
    if os.path.isfile("./save_model/Cartpole_Critic.h5"):
        global_agent.critic.load_weights("./save_model/Cartpole_Critic.h5")
        print(" Critic restored!!")
        
    global_agent.run()
    
    # Save the trained results
    global_agent.actor.save_weights("./save_model/Cartpole_Actor.h5")
    global_agent.critic.save_weights("./save_model/Cartpole_Critic.h5")
