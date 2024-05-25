
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import os
import pickle


class DynaAgent():
    def __init__(self,env,size_bin_x=0.025,size_bin_y=0.005,epsilon_start=0.9,epsilon_end=0.05,alpha=0.1,gamma=0.99,k=5):
        self.env = env
        
        self.n_bins_x = int((env.observation_space.high[0]-env.observation_space.low[0])/size_bin_x)
        
        assert (env.observation_space.high[0]-env.observation_space.low[0])/size_bin_x-self.n_bins_x<1e-5 , "size_bin_x is not a divisor of the range of the x axis"
        self.n_bins_y= int((env.observation_space.high[1]-env.observation_space.low[1])/size_bin_y)
        assert (env.observation_space.high[1]-env.observation_space.low[1])/size_bin_y-self.n_bins_y<1e-5 , "size_bin_y is not a divisor of the range of the y axis"
        self.disc_step=np.array([size_bin_x,size_bin_y])
        self.x_bins = np.linspace(env.observation_space.low[0],env.observation_space.high[0],self.n_bins_x+1)
        self.y_bins= np.linspace(env.observation_space.low[1],env.observation_space.high[1],self.n_bins_y+1)
        
    
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.alpha = alpha
        self.gamma = gamma
        self.k=k
        self.n_states= (self.n_bins_x+1)*(self.n_bins_y+1)
        self.counts_total=np.zeros((self.n_states,3))
        self.counts_transition=np.zeros((self.n_states,3,self.n_states))
        self.P=np.ones((self.n_states,3,self.n_states))*1/self.n_states
        self.R=np.zeros((self.n_states,3))
        self.Q=np.zeros((self.n_states,3))
        self.model = {}
        
        self.cumulative_reward_per_episode=[]
        self.episodes_success=[]#0 or 1
        self.last_reward_step=0
        self.timesteps_per_reward=[]
        
        self.state_action_pairs_encontered = []
        
        self.state=self.discretize_position(self.env.reset()[0])
        self.nb_training_episodes=0
        self.q_value_updates=[]
        self.stored_max_q_every_250_episodes=[]
        self.stored_trajectories_every_250_episodes=[]
        self.episodes_duration=[]
    def discretize_position(self,position):
        
        return np.digitize(position[0],self.x_bins)+np.digitize(position[1],self.y_bins)*(self.n_bins_x+1)
    
    def invert_discretization(self,discretized_position):
        x_idx=discretized_position%(self.n_bins_x+1)
        y_idx=discretized_position//(self.n_bins_x+1)
        return np.array([self.x_bins[x_idx],self.y_bins[y_idx]])
    
    def update_P(self,state,action,next_state):
        self.counts_total[state,action]+=1
        self.counts_transition[state,action,next_state]+=1
        self.P[state,action,next_state]+=self.counts_transition[state,action,next_state]/(1+self.counts_total[state,action])
        self.P[state,action]/=np.sum(self.P[state,action])
        pass
    
    def update_R(self,state,action,reward):
        self.R[state,action] = (self.R[state,action]*(self.counts_total[state,action]-1)+reward)/self.counts_total[state,action]
        pass
    
    def update_Q(self,state,action):
        new_q=self.R[state,action]+self.gamma*np.sum(self.P[state,action]*np.max(self.Q,axis=1))
        self.q_value_updates.append(new_q-self.Q[state][action])
        self.Q[state][action]=new_q
        pass
    
    def select_action(self,state,epsilon):
        if np.random.rand()<epsilon:
            action = np.random.choice([0,1,2])
        else:
            action = np.argmax(self.Q[state])
        return action
    
    
    def observe(self,state,action,next_state,reward):
        self.update_P(state,action,next_state)
        self.update_R(state,action,reward)
        self.update_Q(state,action)
        
        self.state=next_state
        #max_q=np.max(self.Q,axis=1)
        for planning_step in range(self.k):
            
            state,action=self.state_action_pairs_encontered[np.random.randint(len(self.state_action_pairs_encontered))]
            reward = self.R[state,action]
            
            # self.Q[state][action] =reward+self.gamma*np.sum(self.P[state,action]*max_q)
            self.Q[state][action] =reward+self.gamma*np.sum(self.P[state,action]*np.max(self.Q,axis=1))
    
    def update_step(self,epsilon):
        state=self.state
        action=self.select_action(state,epsilon)
        self.state_action_pairs_encontered.append((state,action))
            
        next_state, reward_step, done, truncated ,_= self.env.step(action)
        next_state=self.discretize_position(next_state)
        
        self.observe(state,action,next_state,reward_step)
        
        return done,truncated,reward_step
    
    def update(self,epsilon,render=False):
        if self.nb_training_episodes%100==0:#store max q values and trajectories every 250 episodes
                self.stored_max_q_every_250_episodes.append(np.max(self.Q,axis=-1))
                self.stored_trajectories_every_250_episodes.append([self.state])
                store_traj=True
        else:
            store_traj=False
            
        done=False
        truncated=False
        cum_reward_ep=0
        t0=time()
        while not(done|truncated):
            done,truncated,reward_step=self.update_step(epsilon)
            cum_reward_ep+=reward_step
            if store_traj:
                self.stored_trajectories_every_250_episodes[-1].append(self.state)
                
            if done:
                if render:
                    print(f"Episode finished after {self.nb_training_episodes-self.last_reward_step} timesteps")
                self.timesteps_per_reward.append((self.nb_training_episodes,self.nb_training_episodes-self.last_reward_step))
                self.last_reward_step=self.nb_training_episodes
                self.episodes_success.append(1)
                self.state=self.discretize_position(self.env.reset()[0])
            elif truncated:
                
                self.state=self.discretize_position(self.env.reset()[0])
                self.episodes_success.append(0)
                
        self.episodes_duration.append(time()-t0)
        self.nb_training_episodes+=1
        self.cumulative_reward_per_episode.append(cum_reward_ep)  
                    
        pass
    
    
    
    def training(self,n_steps,reset=True,render=False):
        if render:
            self.env=gym.make('MountainCar-v0', render_mode="human")
        else:
            self.env=gym.make('MountainCar-v0')
        print("Training...")
        if reset:
            self.state=self.discretize_position(self.env.reset()[0])
        else:
            pass
        epsilon = self.epsilon_start
        
        
        store_traj=False
        
        for i in tqdm(range(n_steps)):#number or episodes
            self.update(epsilon,render)
                       
                
            #epsilon exponential decay:
            epsilon = self.epsilon_end+(self.epsilon_start-self.epsilon_end)*np.exp(-i/1000)
           
            
        pass
    
    
    def inference(self,max_steps,render=True):
        if render:
            self.env=gym.make('MountainCar-v0', render_mode="human")
        else:
            self.env=gym.make('MountainCar-v0')
        state=self.discretize_position(self.env.reset()[0])
        tot_reward=0
        
        for i in range(max_steps):
            action = np.argmax(self.Q[state])
            next_state, reward, done, _ ,_= self.env.step(action)
            next_state=self.discretize_position(next_state)
            tot_reward+=reward
            state=next_state
            if done:
                if render:
                    print(f"Episode finished after {i} timesteps")
                break
        if render:
            print(f'total reward: {tot_reward}')
        return tot_reward
    
    def save_agent(self,path):
        if not os.path.exists("agents_saved/"+path+"/"):
            os.makedirs("agents_saved/"+path+"/")
        path="agents_saved/"+path+"/"
        np.save(path+"_P.npy",self.P)
        np.save(path+"_Q.npy",self.Q)
        np.save(path+"_R.npy",self.R)
        #store it as pickle
        pickle.dump(self.state_action_pairs_encontered, open(path+"_state_action_pairs_encontered.pkl", 'wb'))
        np.save(path+"_counts_total.npy",self.counts_total)
        np.save(path+"_counts_transition.npy",self.counts_transition)
        pass
    
    def load_agent(self,path):
        path="agents_saved/"+path+"/"
        self.P=np.load(path+"_P.npy")
        self.Q=np.load(path+"_Q.npy")
        self.R=np.load(path+"_R.npy")
        #self.state_action_pairs_encontered=np.load(path+"_state_action_pairs_encontered.npy")
        with open(path+"_state_action_pairs_encontered.pkl", 'rb') as f:
            self.state_action_pairs_encontered = pickle.load(f)
        self.counts_total=np.load(path+"_counts_total.npy")
        self.counts_transition=np.load(path+"_counts_transition.npy")
        pass
    
    
    
#plots functions 

def plot_episodes_duration(agent,filename=None):
    plt.figure()
    plt.scatter(range(len(agent.episodes_duration)),agent.episodes_duration,s=1,alpha=0.5)
    #plt.yscale("log")
    plt.title("Duration of each episode")
    plt.xlabel("Episode number")
    plt.ylabel("Duration (s)")
    if filename is not None:
        filename="plots/dyna/"+filename
        plt.savefig(filename)
    plt.show()
    
    
    
def plot_accumulated_reward(agent,window=50,filename=None):
    """ Plot the accumulated reward per episode (using average window to smoooth) and on a separate plot the cuùulative reward over episodes"""
    cum_reward_per_ep=np.array(agent.cumulative_reward_per_episode)
    window=window
    moving_average = np.convolve(cum_reward_per_ep, np.ones(window), 'valid') / window
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(moving_average)
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Accumulated reward")
    ax[0].set_title("Accumulated reward per episode")
    ax[1].plot(np.cumsum(cum_reward_per_ep))
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Cumulative reward")
    ax[1].set_title("Cumulative reward over episodes")
    if filename is not None:
        filename="plots/dyna/"+filename
        plt.savefig(filename)
    plt.show()
    pass

def plot_cumulatiuve_number_of_sucesses(agent,filename=None):
    """ Plot the cumulative number of successes over episodes"""
    cum_success=np.cumsum(agent.episodes_success)
    plt.plot(cum_success)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative number of successes")
    plt.title("Cumulative number of successes over episodes")
    if filename is not None:
        filename="plots/dyna/"+filename
        plt.savefig(filename)
    
    plt.show()
    pass

def show_q_updates(agent,step=10):
    y_vals=agent.q_value_updates[::step]
    x_vals=range(len(y_vals))
    plt.scatter(x_vals,y_vals,s=3,alpha=0.2)
    plt.hlines(1,1,len(y_vals),linestyles='dashed',colors='r')
    #plt.yscale("log")
    plt.xlabel(f"Step number (every {step} steps)")
    if step==1:
        plt.xlabel("Step number")
    plt.ylabel("Q value update")
    plt.title(f"Q value updates every {step} steps")
    if step==1:
        plt.title("Q value updates")
    plt.show()

def inference_test(agent,n_tests,return_rewards=False):
    nb_done_under_200_steps=0
    all_rewards=[]
    for i in tqdm(range(n_tests)):
        reward=agent.inference(300,render=False)
        all_rewards.append(reward)
        if reward>=-200:
            nb_done_under_200_steps+=1
    print(f"Proportion of successful episodes: {nb_done_under_200_steps/n_tests*100} %")
    if return_rewards:
        return nb_done_under_200_steps/n_tests,all_rewards
    return nb_done_under_200_steps/n_tests

    
def show_max_Q(agent,filename=None):
    """Plot max Q value for each state
    Args:
        agent: An agent object with Q-values and state bins attributes
    """
    plt.figure(figsize=(5, 5))
    
    # Calculate max Q value per state and set zeros to NaN
    max_Q = agent.Q.max(axis=-1).copy()
    #max_Q[2000]=100
    max_Q=max_Q.reshape((agent.n_bins_y+1), (agent.n_bins_x+1))
    max_Q[max_Q == 0] = np.nan  # Replace zeros with NaN
    
    # Use imshow to plot, NaN values will not be displayed
    plt.imshow(max_Q, origin='lower', aspect='auto', cmap='viridis',extent=[-1.2, 0.6, -0.07, 0.07])  # Added cmap for better color handling
    plt.colorbar()
    
    #add trajectories
    position_250=np.array([agent.invert_discretization(state) for state in agent.stored_trajectories_every_250_episodes[0]])
    position_500=np.array([agent.invert_discretization(state) for state in agent.stored_trajectories_every_250_episodes[1]])
    position_1000=np.array([agent.invert_discretization(state) for state in agent.stored_trajectories_every_250_episodes[3]])
    position_3000=np.array([agent.invert_discretization(state) for state in agent.stored_trajectories_every_250_episodes[-1]])
    
    plt.plot(position_250[:,0],position_250[:,1],color='r',label="250",alpha=0.5)
    plt.plot(position_500[:,0],position_500[:,1],color='w',label="500",alpha=0.5)
    plt.plot(position_1000[:,0],position_1000[:,1],color='b',label="1000",alpha=0.5)
    plt.plot(position_3000[:,0],position_3000[:,1],color='black',label="3000",alpha=0.5)
    plt.legend(title="Trajectory episode n°")
        
    plt.title(f"Max Q values (after {agent.nb_training_episodes} episodes)")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    if filename is not None:
        filename="plots/dyna/"+filename
        plt.savefig(filename)
    plt.show()
    
def plot_all(agent,step=1,window=50):
    plot_episodes_duration(agent)
    plot_accumulated_reward(agent,window)
    plot_cumulatiuve_number_of_sucesses(agent)
    show_q_updates(agent,step=step)
    show_max_Q(agent)
    pass
def plot_all_on_same_figure(agent):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    #plot_episodes_duration(agent)
    ax[0,0].scatter(range(len(agent.episodes_duration)),agent.episodes_duration,s=1,alpha=0.5)
    ax[0,0].set_title("Duration of each episode")
    ax[0,0].set_xlabel("Episode number")
    ax[0,0].set_ylabel("Duration (s)")
    
    #plot_accumulated_reward(agent)
    cum_reward_per_ep=np.array(agent.cumulative_reward_per_episode)
    window=50
    moving_average = np.convolve(cum_reward_per_ep, np.ones(window), 'valid') / window
    ax[0,1].plot(moving_average)
    ax[0,1].set_xlabel("Episode")
    ax[0,1].set_ylabel("Accumulated reward")
    ax[0,1].set_title("Accumulated reward per episode")
    
    #plot_cumulatiuve_number_of_sucesses(agent)
    cum_success=np.cumsum(agent.episodes_success)
    ax[1,0].plot(cum_success)
    ax[1,0].set_xlabel("Episode")
    ax[1,0].set_ylabel("Cumulative number of successes")
    ax[1,0].set_title("Cumulative number of successes over episodes")
    
    #show_q_updates(agent)
    y_vals=agent.q_value_updates[::]
    x_vals=range(len(y_vals))
    ax[1,1].scatter(x_vals,y_vals,s=3,alpha=0.2)
    ax[1,1].hlines(1,1,len(y_vals),linestyles='dashed',colors='r')
    #ax[1,1].set_yscale("log")
    ax[1,1].set_xlabel("Step number")
    ax[1,1].set_ylabel("Q value update")
    ax[1,1].set_title("Q value updates")
    plt.show()
    
    #show_max_Q(agent)
    show_max_Q(agent)
    pass

def plot_all_on_same_column(agent):
    fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    #plot_episodes_duration(agent)
    ax[0].scatter(range(len(agent.episodes_duration)),agent.episodes_duration,s=1,alpha=0.5)
    ax[0].set_title("Duration of each episode")
    ax[0].set_xlabel("Episode number")
    ax[0].set_ylabel("Duration (s)")
    
    #plot_accumulated_reward(agent)
    cum_reward_per_ep=np.array(agent.cumulative_reward_per_episode)
    window=50
    moving_average = np.convolve(cum_reward_per_ep, np.ones(window), 'valid') / window
    ax[1].plot(moving_average)
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Accumulated reward")
    ax[1].set_title("Accumulated reward per episode")
    
    #plot_cumulatiuve_number_of_sucesses(agent)
    cum_success=np.cumsum(agent.episodes_success)
    ax[2].plot(cum_success)
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Cumulative number of successes")
    ax[2].set_title("Cumulative number of successes over episodes")
    
    #show_q_updates(agent)
    y_vals=agent.q_value_updates[::]
    x_vals=range(len(y_vals))
    ax[3].scatter(x_vals,y_vals,s=3,alpha=0.2)
    ax[3].hlines(1,1,len(y_vals),linestyles='dashed',colors='r')
    #ax[3].set_yscale("log")
    ax[3].set_xlabel("Step number")
    ax[3].set_ylabel("Q value update")
    ax[3].set_title("Q value updates")
    plt.show()
    
    #show_max_Q(agent)
    show_max_Q(agent)
    pass

def plot_max_q_at_different_episodes(agent):
    """ Plot the max Q value for different episodes, on the same figure with 3 columns"""
    nb_lines=(len(agent.stored_max_q_every_250_episodes)-1)//3+1
    fig,ax=plt.subplots(nb_lines,3,figsize=(15,5*nb_lines))
    for i in range(len(agent.stored_max_q_every_250_episodes)):
        max_Q = agent.stored_max_q_every_250_episodes[i].copy()
        max_Q=max_Q.reshape((agent.n_bins_y+1), (agent.n_bins_x+1))
        max_Q[max_Q == 0] = np.nan  # Replace zeros with NaN
        ax[i//3,i%3].imshow(max_Q, origin='lower', aspect='auto', cmap='viridis',extent=[-1.2, 0.6, -0.07, 0.07])  # Added cmap for better color handling
        ax[i//3,i%3].set_title(f"Max Q values episode {i*250}")
        ax[i//3,i%3].set_xlabel("Position")
        ax[i//3,i%3].set_ylabel("Velocity")
    #plt.colorbar()
    plt.show()
    pass
    