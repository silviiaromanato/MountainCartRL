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
    
class RandomNetwork(nn.Module):
    def __init__(self, state_size, output_size=1, hidden_sizes=[64, 64]):
        super(RandomNetwork, self).__init__()
        layers = [nn.Linear(state_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, env, state_size, action_size,
                 gamma = 0.99, min_epsilon = 0.05, max_epsilon = 0.9, 
                 decay_epsilon = 0.995, replay_buffer_max = 10000, batch_size = 64, 
                 learning_rate=0.001, hidden_layer_sizes = [64, 64],
                 target_update_frequency = 500, reward_factor = 0.1):
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network (target and policy network)
        self.Q = DQNnetwork(state_size, action_size, hidden_layer_sizes)
        self.Q_target = DQNnetwork(state_size, action_size, hidden_layer_sizes)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.loss_history = []

        self.rewards_history = []
        self.aux_rewards_history = []
        self.durations_history = []
        self.dones_history = []

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_epsilon = decay_epsilon
        self.reward_factor = reward_factor

        # Update target network frequency
        self.target_update_frequency = target_update_frequency
        self.update_count = -1

        # RND networks
        self.target = RandomNetwork(state_size)
        self.predictor = RandomNetwork(state_size)
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        # Normalization parameters
        self.state_mean = 0
        self.state_std = 1
        self.reward_mean = 0
        self.reward_std = 1

        # Initial steps to skip reward calculation
        self.initial_steps = 1000
        self.step_count = 0

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_max)


    def observe(self, state, action, next_state, reward, done, rnd = False):

        if rnd == True:
            intrinsic_reward = self.update_rnd(state)
            reward += intrinsic_reward
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.step_count += 1
        self.update()

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
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q_values = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.Q_target(next_states).detach().max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
    
        self.ep_loss += loss.item()
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_epsilon

    def update_rnd(self, state):
        # Normalize state
        normalized_state = (state - self.state_mean) / self.state_std
        normalized_state = torch.FloatTensor(normalized_state).unsqueeze(0)

        target_output = self.target(normalized_state).detach()
        predictor_output = self.predictor(normalized_state)

        # Calculate loss and update predictor
        loss = nn.MSELoss()(predictor_output, target_output)
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        # Calculate intrinsic reward
        if self.step_count > self.initial_steps:
            reward_diff = (predictor_output - target_output).pow(2)
            reward_diff = reward_diff.detach().numpy()
            normalized_reward_diff = (reward_diff - self.reward_mean) / self.reward_std
            normalized_reward_diff = np.clip(normalized_reward_diff, -5, 5)
            return normalized_reward_diff.item() * self.reward_factor
        else:
            return 0

    def train(self, env, agent, num_episodes, reward_function = "-1", rnd = False):
        for ep in tqdm(range(num_episodes)):
            t0 = time()
            seed = np.random.randint(0, 100000)
            state = env.reset(seed=seed)[0]
            done, truncated = False, False
            ep_reward, total_aux_reward = 0, 0
            self.ep_loss = 0

            while not (done | truncated):
                action = agent.select_action(state)
                
                if reward_function != "-1":
                    next_state, reward, done, truncated, _, aux_reward = env.step(action, reward_function = reward_function)
                    agent.observe(state, action, next_state, aux_reward, done)
                    total_aux_reward += aux_reward
                else:   
                    next_state, reward, done, truncated, _ = env.step(action)
                    agent.observe(state, action, next_state, reward, done, rnd = rnd)

                ep_reward += reward
                state = next_state

                if rnd:
                    self.state_mean = 0.9 * self.state_mean + 0.1 * np.mean(next_state)
                    self.state_std = 0.9 * self.state_std + 0.1 * np.std(next_state)

            if reward_function != "-1":
                self.aux_rewards_history.append(total_aux_reward)
            self.rewards_history.append(ep_reward)
            self.durations_history.append(time() - t0)
            self.dones_history.append(done)
            self.loss_history.append(self.ep_loss)
        
    def save_agent(self,path):
        current_directory = os.getcwd()
        path = current_directory + "/agents_saved/" + path + "/"
        if os.path.exists(path) == False:
            os.makedirs(path)
            print("Directory created: ", path)
        
        torch.save(self.Q.state_dict(), path + "Q_values.pt")
        np.save(path + "rewards.npy", self.rewards_history)
        np.save(path + "durations.npy", self.durations_history)
        np.save(path + "dones.npy", self.dones_history)
        np.save(path + "aux_rewards.npy", self.aux_rewards_history)
        print("Agent saved on path: ", path)


    def load_agent(self,path):
        current_directory = os.getcwd()
        path = current_directory + "/agents_saved/" + path + "/"
        self.Q.load_state_dict(torch.load(path + "Q_values.pt"))
        self.rewards_history = np.load(path + "rewards.npy").tolist()
        self.durations_history = np.load(path + "durations.npy").tolist()
        self.dones_history = np.load(path + "dones.npy").tolist()
        self.aux_rewards_history = np.load(path + "aux_rewards.npy").tolist()
        print("Agent loaded from path: ", path)

    def plots(self, reward_function = "-1"):

        y = np.linspace(0, len(self.rewards_history), len(self.rewards_history))
        if reward_function != "-1":
            fig, axs = plt.subplots(3, 2, figsize=(10, 8))
            fig.suptitle('Training Results for DQN', fontsize=16)

            axs[0, 0].plot(self.rewards_history, color='purple')
            axs[0, 0].set_title('Rewards')
            axs[0, 0].set_ylabel('Reward')
            axs[0, 0].set_xlabel('Episode')

            axs[1, 0].plot(self.aux_rewards_history, color='purple')
            axs[1, 0].set_title('Auxiliary Rewards')
            axs[1, 0].set_ylabel('Auxiliary Reward')
            axs[1, 0].set_xlabel('Episode')

            axs[2, 0].scatter(y, self.durations_history, s = 3)
            axs[2, 0].set_title('Durations')
            axs[2, 0].set_ylabel('Duration (seconds)')
            axs[2, 0].set_xlabel('Episode')

            axs[0, 1].scatter(y, self.dones_history, s = 3)
            axs[0, 1].set_title('Dones')
            axs[0, 1].set_ylabel('Done')
            axs[0, 1].set_xlabel('Episode')

            axs[1, 1].plot(self.loss_history)
            axs[1, 1].set_title('Loss')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].set_xlabel('Episode')

            # Hide empty subplot
            axs[2, 1].axis('off')

            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.show()

        else:

            fig, axs = plt.subplots(3, 2, figsize=(10, 8))
            fig.suptitle('Training Results for DQN', fontsize=16)

            axs[0, 0].plot(self.rewards_history, color='purple')
            axs[0, 0].set_title('Rewards')
            axs[0, 0].set_ylabel('Reward')
            axs[0, 0].set_xlabel('Episode')

            axs[2, 0].scatter(y, self.durations_history, s = 3)
            axs[2, 0].set_title('Durations')
            axs[2, 0].set_ylabel('Duration (seconds)')
            axs[2, 0].set_xlabel('Episode')

            axs[0, 1].scatter(y, self.dones_history, s = 3)
            axs[0, 1].set_title('Dones')
            axs[0, 1].set_ylabel('Done')
            axs[0, 1].set_xlabel('Episode')

            axs[1, 1].plot(self.loss_history)
            axs[1, 1].set_title('Loss')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].set_xlabel('Episode')

            # Hide empty subplot
            axs[2, 1].axis('off')

            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.show()

