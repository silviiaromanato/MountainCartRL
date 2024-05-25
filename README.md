# MountainCartRL
## Authors
Silvia Romano
Alexi Semiz

## Introduction
In this project, we explore the application of Reinforcement Learning (RL) algorithms to the Mountain Car environment, a classic benchmark problem implemented in Gymnasium. The Mountain Car environment presents a scenario where a car is placed at a random location within a valley. The objective is to drive the car to the top of the right hill. However, the car is underpowered, requiring it to build momentum by oscillating back and forth between the hills before it can reach the goal.

The agent in this environment has three possible actions: accelerate to the left, accelerate to the right, or do nothing. The reward system is straightforward yet challenging: the agent receives a reward of -1 for each time step until it reaches the goal, at which point the reward is 0. This sparse reward function poses a significant challenge because the agent receives no positive feedback until it achieves the goal, making it difficult for the agent to learn effective strategies.

To address these challenges, this project implements and compares two RL algorithms: a model-free algorithm known as Deep Q-Network (DQN) and a model-based algorithm called Dyna. The DQN algorithm learns directly from interactions with the environment, updating its policy based on the rewards received. In contrast, the Dyna algorithm combines direct learning from interactions with simulated learning from a model of the environment, potentially improving efficiency and performance.

## Code
The DQN agents lcass are coded in the DQN.py file while the DYNA agent class is in DYNA.py. The main notebook is MountainCart_notebook.ipynb. All the agents training, plots, results are there.