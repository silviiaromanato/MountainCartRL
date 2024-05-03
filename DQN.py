import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import gymnasium as gym
from time import time
from matplotlib import pyplot as plt

class DQNnetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_sizes):
        super(DQNnetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(state_size, hidden_layer_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)     

class DQNAgent:
    def __init__(self, env, state_size, action_size,
                 gamma = 0.99, min_epsilon = 0.05, max_epsilon = 0.9, 
                 decay_epsilon = 0.995, replay_buffer_max = 10000, batch_size = 64, 
                 learning_rate=0.001, hidden_layer_sizes = [64, 64]):
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0 # Not too sure about the approach

        # Q-Network
        self.Q = DQNnetwork(state_size, action_size, hidden_layer_sizes)
        self.Q_target = DQNnetwork(state_size, action_size, hidden_layer_sizes)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_epsilon = decay_epsilon
    
        self.replay_buffer = deque(maxlen=replay_buffer_max)

        self.loss_history = []

    def observe(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.Q(state).detach().numpy()
            return np.argmax(q_values)

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            if type(state) is tuple:
                state = state[0]
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            q_values = self.Q(state).detach().numpy()
            if done:
                q_values[action] = reward
            else:
                next_q_values = self.Q_target(next_state).detach().numpy()
                q_values[action] = reward + (self.gamma * np.amax(next_q_values))
            states.append(state)
            targets.append(torch.FloatTensor(q_values))
        states = torch.stack(states)
        targets = torch.stack(targets)
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(self.Q(states), targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_epsilon

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.loss_history.append(loss.item())

    def train(self, env, agent, num_episodes, reward_function = "-1"):
        rewards = []
        durations=[]
        dones = []
        aux_rewards = []
        for _ in tqdm(range(num_episodes)):
            t0 = time()
            seed = np.random.randint(0, 100000)
            state = env.reset(seed=seed)[0]
            done, truncated = False, False
            total_reward, total_aux_reward = 0, 0

            while not (done | truncated):
                action = agent.select_action(state)
                next_state, reward, done, truncated, _, aux_reward = env.step(action, reward_function = reward_function)
                if reward_function != "-1":
                    agent.observe(state, action, next_state, aux_reward, done)
                else:   
                    agent.observe(state, action, next_state, reward, done)
                agent.update()
                total_reward += reward
                total_aux_reward += aux_reward
                state = next_state

            aux_rewards.append(total_aux_reward)
            rewards.append(total_reward)
            durations.append(time() - t0)
            dones.append(done)

        self.rewards = rewards
        self.durations = durations
        self.dones = dones
        self.aux_rewards = aux_rewards
        
    def save_agent(self,path):
        current_directory = os.getcwd()
        path = current_directory + "/agents_saved/" + path + "/"
        if os.path.exists(path) == False:
            os.makedirs(path)
            print("Directory created: ", path)
        
        torch.save(self.Q.state_dict(), path + "Q_values.pt")
        np.save(path + "rewards.npy", self.rewards)
        np.save(path + "durations.npy", self.durations)
        print("Agent saved on path: ", path)


    def load_agent(self,path):
        current_directory = os.getcwd()
        path = current_directory + "/agents_saved/" + path + "/"
        self.Q.load_state_dict(torch.load(path + "Q_values.pt"))
        self.rewards = np.load(path + "rewards.npy")
        self.durations = np.load(path + "durations.npy")
        print("Agent loaded from path: ", path)

    def plots(self):
        fig, axs = plt.subplots(3, 2, figsize=(10, 8))
        fig.suptitle('Training Results for DQN', fontsize=16)

        axs[0, 0].plot(self.rewards, color='purple')
        axs[0, 0].set_title('Rewards')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].set_xlabel('Episode')

        axs[1, 0].plot(self.aux_rewards, color='purple')
        axs[1, 0].set_title('Auxiliary Rewards')
        axs[1, 0].set_ylabel('Auxiliary Reward')
        axs[1, 0].set_xlabel('Episode')

        axs[2, 0].plot(self.durations)
        axs[2, 0].set_title('Durations')
        axs[2, 0].set_ylabel('Duration (seconds)')
        axs[2, 0].set_xlabel('Episode')

        axs[0, 1].plot(self.dones)
        axs[0, 1].set_title('Dones')
        axs[0, 1].set_ylabel('Done')
        axs[0, 1].set_xlabel('Episode')

        axs[1, 1].plot(self.loss_history)
        axs[1, 1].set_title('Loss')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].set_xlabel('Training Step')

        # Hide empty subplot
        axs[2, 1].axis('off')

        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.show()

