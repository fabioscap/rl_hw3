import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, random, collections

class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self, n_frames = 4, 
                       device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous)

        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        self.n_frames = n_frames # store n_frames consequent observations as the state
                                 # (restore Markov)
        self.q_frames = collections.deque(maxlen=n_frames)

        self.replay_buffer = UniformReplayBuffer(100,(2,3,2))

        self.q_network = Q_network(self.n_frames,5)

    def forward(self, x):
        return self.q_network(torch.from_numpy(x))
    
    def act(self, state):
        out = self.q_network(torch.from_numpy(state) / 255.0)
        return torch.argmax(out) # Q-greedy

    def train(self, T, K, k, beta):
        # TODO
        # 1-to-1 implementation of pseudocode
        # as described in the paper
        obs, *_ = self.env.reset()
        self.reset_frames(obs)
        state = self.get_state()

        action = self.act(state)

        for t in range(T):
            obs, rew, term, trunc, info = self.env.step(action)

            new_state = self.process_obs(obs)

            # store with maximum probability
            self.replay_buffer.store(state,action,rew,term,new_state)

            if t % K == 0: # sample k transitions from replay buffer
                self.batch_update(self.replay_buffer.sample(k))
                    
        return
    
    def batch_update(self,transitions):
        self.network.optimizer.zero_grad()

        # stack transitions
        
        # compute Q values for each state using fast network

        # compute Q values using slow network

        # compute the max of next state Q values
        return

    def append_frame(self,obs):
        processed_obs = self.process_obs(obs)
        self.q_frames.append(processed_obs)

    def reset_frames(self,init_obs):
        processed_obs = self.process_obs(init_obs)
        self.q_frames = collections.deque(self.n_frames * [processed_obs], maxlen=self.n_frames)
    
    def process_obs(self,obs:np.ndarray):
        # process the observation
        # resize, color, crop(?) ...

        # to float
        obs = obs.astype(np.float32) / 255.0
        # grayscale
        graymap = np.array([0.2989, 0.5870, 0.1140])
        obs = np.dot(obs[...,:], graymap)
        return obs.reshape(1,*obs.shape)

    def get_state(self):
        frames = np.array(self.q_frames,dtype=np.float32)
        # stack all frames in the channel dimension

        # stack along channels
        state = frames.reshape(1,-1, *frames.shape[2:]) # shape (1,N*C,W,H)

        # stack along width
        #state = frames.reshape(1,frames.shape[1],-1,frames.shape[3] ) # shape (1,C,N*W,H)
        
        return state 

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'), map_location=self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

class Q_network(nn.Module):

    def __init__(self, in_channels, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=15, kernel_size=3, stride=1)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        o1 = self.act(self.conv1(x))
        o2 = self.pool(self.act(self.conv2(o1)))
        
        print(o2.shape)
        return o2
        

class UniformReplayBuffer():
    # uniform sampling
    
    # the buffer is a collection of vectors representing flattened transition
    def __init__(self, capacity: int,   # the capacity of the buffer
                       s_shape:  tuple, # the shape of a state
                       a_shape=(1,)):   # the shape of an action
        
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.s_size = math.prod(s_shape) # how many numbers to store a state
        self.a_size = math.prod(a_shape) # how many numbers to store an action

        # the size of a transition: s+a+r+d+s' 
        self.t_size = self.s_size + self.a_size + 1 + 1 + self.s_size 

        self.size = capacity
        self.idx = -1
        self.buffer = np.zeros((capacity, self.t_size), dtype=np.float32)
        self.full = False # wether the buffer is full or it contains empty spots

        # pre compute where to put the data and how to fetch it back
        self.s_slice = slice(0,self.s_size)
        self.a_slice = slice(self.s_slice.stop,self.s_slice.stop+self.a_size)
        self.r_slice = slice(self.a_slice.stop,self.a_slice.stop+1)
        self.d_slice = slice(self.r_slice.stop,self.r_slice.stop+1)
        self.sp_slice = slice(self.d_slice.stop,self.d_slice.stop+self.s_size)
        
    def store(self, s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    sp: np.ndarray):
        
        self.idx += 1
        if (self.idx >= self.size):
            self.full = True # the buffer is full
            self.idx = 0     # reset the index (start to overwrite old experiences)

        self.buffer[self.idx,self.s_slice] = s.flatten()
        self.buffer[self.idx,self.a_slice] = a if type(a) == int else a.flatten()
        self.buffer[self.idx,self.r_slice] = r
        self.buffer[self.idx,self.d_slice] = d
        self.buffer[self.idx,self.sp_slice] = sp.flatten()
            
    def sample(self, n: int):
        if self.full: # sample from whole buffer
            high = self.size
        else: # do not consider empty spots
            high = self.idx

            
        # random.sample performs sampling without replacement
        idxes = random.sample(range(high), n) 
        # idxes = random.choice(range(high), n) # with replacement, might be faster
        
        transitions = self.buffer[idxes]

        # model wants states to be in shape [n, s_shape]
        states =     transitions[:,self.s_slice].reshape(n,*self.s_shape)
        actions =    transitions[:,self.a_slice]
        rewards =    transitions[:,self.r_slice]
        dones =      transitions[:,self.d_slice]
        new_states = transitions[:,self.sp_slice].reshape(n,*self.s_shape)

        return (states,actions,rewards,dones,new_states)


p = Policy()
p.train(T=10, K = 5, k=2, beta = 0)   


## Resources ##
# https://ai.stackexchange.com/questions/12144/which-kind-of-prioritized-experience-replay-should-i-use