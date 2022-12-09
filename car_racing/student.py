import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, random, collections
from functools import reduce
import operator

class Policy():
    continuous = False # you can change this

    def __init__(self, n_frames = 3, 
                       lr = 5e-7,
                       gamma = 0.99,
                       epsilon = 0.5,
                       epsilon_decay = .995,
                       epsilon_bound = 0.1,
                       max_steps_per_episode = 1000,
                       render=False,
                       device=torch.device('cuda')):

        self.device = device

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_bound = epsilon_bound

        self.render = render

        if self.render:
            self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        
        self.max_steps_per_episode = max_steps_per_episode

        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        self.n_frames = n_frames # store n_frames consequent observations as the state
                                 # (restore Markov)
        self.q_frames = collections.deque(maxlen=n_frames)
        self.state_shape = (self.n_frames,83,96)

        self.q_network = Q_network(self.n_frames,self.env.action_space.n).to(device)
        # numel = sum([p.numel() for p in self.q_network.parameters()])

        self.lr = lr
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self._eval_reset = True # flag to prepare the frames in evaluation mode
        self.last_action = None

    def act_training(self,state,epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        r = random.random()
        if r > epsilon:
            with torch.no_grad():
                out = self.q_network(torch.from_numpy(state).to(self.device))
                action = int(torch.argmax(out).item()) # Q-greedy
        else:
            # with some probability random, with some the previous
            if self.last_action != None and r > epsilon / 2: 
                action = self.last_action
            else:
                action = self.env.action_space.sample()
        
        self.last_action = action
        return action

    def act(self, obs): # to be used in evaluation settings
        if self._eval_reset:
            state = self.reset_frames(obs)
        else:
            self._eval_reset = False
            state = self.append_frame(obs)
        state = self.get_state()
        
        return self.act_training(state, epsilon=0.1)

    def train(self, n_episodes = 6000,
                    update = 5,
                    batch = 64,
                    sync = 20000, #steps
                    buffer_capacity = 100000,
                    save = 10, #episodes
                    eval = 10, #episodes
                    finetune = True,
                    forward_start = False,
                    log = True
                    ):

        if finetune:
            self.load()
        
        # initialize the target network
        self.t_network = Q_network(self.n_frames,self.env.action_space.n).to(self.device)
        for param in self.t_network.parameters(): 
            param.requires_grad = False # disable gradient computation for target network
        self.sync_networks()

        # initialize the replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity,self.state_shape, alpha=0.8)

        # total timestep variable
        t = 0

        if log:
            open("log.txt","w").close()
        
        for episode in range(n_episodes):
            done = False
            cause = "term"
            state = self.reset_env()

            # start the track a little bit further (not really useful)
            if forward_start:
                for _ in range(random.randint(10,60)):
                    obs, *_ = self.env.step(3)
                state = self.reset_frames(obs)

            step = 0
            episode_reward = 0.0
            touch_grass = 0
            neg_rewards = 0

            while not done:
                rew = 0
                action = self.act_training(state)

                obs, rew, term, trunc, _ = self.env.step(action)
                done = term or trunc
                if self.render: self.env.render()

                episode_reward += rew # the actual (not modified) reward.
                
                # some reward modifications to help convergence
                #   check for grass in front of the car
                if obs[64,48,1] > 200: 
                    rew += -0.05
                    touch_grass += 1
                else:
                    touch_grass = 0
                if touch_grass >= 90:
                    rew += -5.0
                    done = True # most likely off track
                    cause = "grass"

                if action == 0: # do nothing
                    rew += -1.0
                elif action == 1: # right
                    rew += 0
                elif action == 2: # left
                    rew += 0 
                elif action == 3: # gas
                    rew += 0.05
                else: # brake 
                    rew += 0

                if rew < 0:
                    neg_rewards += 1
                else:
                    neg_rewards = 0
                if neg_rewards > 180:
                    done = True
                    cause = "neg rewards"

                # fetch the new state
                new_state = self.append_frame(obs)

                # we have everything that's necessary to store the transition
                # in the replay buffer
                self.replay_buffer.store(state, action, rew, done, new_state)
                #rb.store(state, action, rew, done, new_state)
                
                # next iteration act upon the new state
                state = new_state

                step += 1
                t += 1
                if step > self.max_steps_per_episode:
                    done = True
                    cause = "max_steps"
                
                if t > 0:
                    if t > batch and t % update == 0:
                        self.batch_update(*self.replay_buffer.sample(batch))
                    if t % sync == 0:
                        self.sync_networks()
        
            self.update_epsilon()


            print(f"episode {episode}:\n"
                    f"-steps:  {step}\n" + 
                    f"-reward: {episode_reward}\n" +
                    f"-cause:  {cause}\n" +
                    f"-eps:    {self.epsilon}")


            if log:
                with open("log.txt","a") as log_file:
                    log_file.write(f"{episode} {step} {episode_reward} {self.epsilon}\n")

            if episode > 0 and episode % save == 0:
                print("saving network ...")
                self.save()
                print("done.")
            if episode > 0 and episode % eval == 0:
                self.eval(render=False, n_episodes=1)

    def reset_env(self, t = 0):
        if self.render:
            self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        # reset the environment
        obs, *_ = self.env.reset()
        for _ in range(t): # do nothing (skip the zoom in phase)
            obs, *_ = self.env.step(0)
        
        # reset the frame buffer with the first useful state
        state = self.reset_frames(obs) 
        return state

    def eval(self, n_episodes=5, render=True):
        if render:
            env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode="human")
        else:
            env = gym.make('CarRacing-v2', continuous=self.continuous)
        total_reward = 0
        for episode in range(n_episodes):
            episode_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(self.max_steps_per_episode):
                action = self.act(s)
                s, reward, done, truncated, info = env.step(action)
                if render: env.render()
                episode_reward += reward
                if done or truncated: break
            total_reward += episode_reward
        env.close()
        total_reward /= n_episodes
        if total_reward > 300:
            self.save(f'model{int(total_reward)}')

        print(f"EVALUATION REWARD {total_reward}")

    def batch_update(self,states,actions,rewards,dones,new_states,idxes,weights):
        
        s = torch.from_numpy(states).to(self.device)
        a = torch.from_numpy(actions).to(self.device).reshape(-1,1)
        r = torch.from_numpy(rewards).to(self.device).reshape(-1,1)
        d = torch.from_numpy(dones).to(self.device).reshape(-1,1)
        sp = torch.from_numpy(new_states).to(self.device)
        w = torch.from_numpy(weights).float().to(self.device).reshape(-1,1)

        self.optimizer.zero_grad()
        
        # compute Q values for current state
        Q_s = self.q_network(s) # (k,a)

        Q_s = torch.gather(Q_s,1,a.long()) # (k) 
        
        with torch.no_grad():
            Q_sp_ = self.q_network(sp) # (k,a)
            amax = torch.argmax(Q_sp_, dim=1).unsqueeze(-1) # (k)

        Q_sp = self.t_network(sp) # (k,a)
        Q_sp = torch.gather(Q_sp,1,amax.long()) # (k) 

        """
        # compute Q values for each state using q_network
        Q_s = self.q_network(s) # (k,a)

        Q_s = torch.gather(Q_s,1,a.long()) # (k) 

        # compute Q values using t_network
        Q_sp = self.t_network(sp) # (k,a)
        #Q_sp = self.q_network(sp) # (k,a)
        # compute the max of next state Q values
        Q_sp = torch.max(Q_sp,dim=1).values.reshape(-1,1) # (k)
        """
        target = r + (1-d)*self.gamma*Q_sp
        
        # TD = torch.clamp(target - Q_s,-1,1)
        TD = target - Q_s

        loss = self.loss(TD.mul(w),torch.zeros_like(TD))
        #loss = self.loss(TD,torch.zeros_like(TD))
        
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(idxes,TD.detach().to("cpu").reshape(-1))
        
        return loss

    def append_frame(self,obs):
        processed_obs = self.process_obs(obs)
        self.q_frames.append(processed_obs)
        return self.get_state()

    def reset_frames(self,init_obs):
        processed_obs = self.process_obs(init_obs)
        self.q_frames = collections.deque(self.n_frames * [processed_obs], maxlen=self.n_frames)
        return self.get_state()
    
    def process_obs(self,obs:np.ndarray):
        # process the observation

        # crop
        obs = obs[0:83,:,:] # cut out the bottom bar

        # get rid of boxes
        obs[:,:,1][obs[:,:,1] > 200] = 255

        # to float
        obs = obs.astype(np.float32) / 255.0

        # grayscale
        graymap = np.array([0.3, 0.3, 0.4])
        obs = np.dot(obs[...,:], graymap)

        return obs.reshape(1,*obs.shape)

    def get_state(self):
        frames = np.array(self.q_frames,dtype=np.float32)
        # stack all frames in the channel dimension

        # stack along channels
        state = frames.reshape(1,-1, *frames.shape[2:]) # shape (1,N*C,W,H)

        return state 

    def sync_networks(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay if self.epsilon > self.epsilon_bound else 1

    def save(self, name='model.pt'):
        torch.save(self.q_network.state_dict(), name)

    def load(self):
        self.q_network.load_state_dict(torch.load('model.pt', map_location=self.device))

class Q_network(nn.Module):

    def __init__(self, in_channels, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.act = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=1)

        self.linear1 = nn.Linear(64,96)
        self.linear2 = nn.Linear(96,out_dim)

    def forward(self, x):
        o1 = self.act(self.conv1(x))
        o2 = self.pool1(self.act(self.conv2(o1)))
        o3 = self.pool1(self.act(self.conv3(o2)))

        # coincidentally o3 already has the right shape with these cnn parameters
        o3_linear = o3.flatten(start_dim=1)
        
        o4 = self.act(self.linear1(o3_linear))
        out = self.linear2(o4)
        return out
        
class UniformReplayBuffer():
    # uniform sampling
    
    # the buffer is a collection of vectors representing flattened transition
    def __init__(self, capacity: int,   # the capacity of the buffer
                       s_shape:  tuple, # the shape of a state
                       a_shape=(1,)):   # the shape of an action
        
        self.s_shape = s_shape
        self.a_shape = a_shape

        self.s_size = reduce(operator.mul, s_shape, 1) # how many numbers to store a state
        self.a_size = reduce(operator.mul, a_shape, 1) # how many numbers to store an action

        # the size of a transition: s+a+r+d+s' 
        self.t_size = self.s_size + self.a_size + 1 + 1 + self.s_size 

        self.capacity = capacity
        self.idx = -1

        self.buffer = {"state":      np.zeros((capacity,*s_shape),dtype=np.float32),
                       "action":     np.zeros((capacity,*a_shape),dtype=np.float32),
                       "reward":     np.zeros(capacity,dtype=np.float32),
                       "done":       np.zeros(capacity,dtype=np.float32),
                       "next_state": np.zeros((capacity,*s_shape),dtype=np.float32)}

        self.full = False # wether the buffer is full or it contains empty spots

    def size(self):
        if self.full:
            return self.capacity
        else:
            return self.idx+1

    def store(self, s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    sp: np.ndarray):
        
        self.idx += 1
        if (self.idx >= self.capacity):
            if not self.full:
                self.full = True # the buffer is full
                print("buffer full")
            self.idx = 0     # reset the index (start to overwrite old experiences)

        self.buffer["state"][self.idx,...] = s.copy()
        self.buffer["action"][self.idx] = a if type(a) == int else a.copy()
        self.buffer["reward"][self.idx] = r
        self.buffer["done"][self.idx] = d
        self.buffer["next_state"][self.idx,...] = sp.copy()

    def sample_idxes_weights(self,n):
        high = self.size()
        return random.choices(population=range(high), k=n), None     

    def sample(self, n: int):
        # random.sample performs sampling without replacement
        idxes, w = self.sample_idxes_weights(n)


        # model wants states to be in shape [n, s_shape]
        states =     self.buffer["state"][idxes]
        actions =    self.buffer["action"][idxes]
        rewards =    self.buffer["action"][idxes]
        dones =      self.buffer["done"][idxes]
        new_states = self.buffer["next_state"][idxes]

        return (states,actions,rewards,dones,new_states,idxes,w)

class PrioritizedReplayBuffer(UniformReplayBuffer):

    def __init__(self, capacity: int, s_shape: tuple, a_shape=(1, ), alpha=0.6, beta_0=0.4, beta_inc=1.001):
        super().__init__(capacity, s_shape, a_shape)
        if math.ceil(math.log2(capacity)) != math.floor(math.log2(capacity)):
            capacity = 2**math.ceil(math.log2(capacity))
            print(f"rescaling buffer to the next power of two: {capacity}.")
        
        # store the priorities in a tree
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0

        self.alpha = alpha
        self.beta = beta_0
        self.beta_inc = beta_inc

    def sample_idxes_weights(self, n):
        high = self.size()

        (idxes, Ps) = self.priorities.sample_batch(n)

        w = (high*Ps)**-self.beta

        w /= w.max()
        if self.beta < 1: # beta annealing
            self.beta*= self.beta_inc 

        return idxes, w

    def store(self, s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    sp: np.ndarray):
        super().store(s,a,r,d,sp)
        self.priorities.set_priority(self.idx,self.max_priority)

    def update_priorities(self, idxes, td_errors, eps=1e-6):
        updated_priorities = np.abs(td_errors)**self.alpha + eps

        _m = updated_priorities.max()
        if _m > self.max_priority: # update the maximum priority
            self.max_priority = _m

        for i in range(len(idxes)):
            self.priorities.set_priority(idxes[i],updated_priorities[i])

class SumTree():
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.size = 2*n_bins - 1
        self.data = np.zeros(self.size)
        self.height = math.log2(n_bins)

    def _left(self, i):
        return 2*i+1

    def _right(self, i):
        return 2*i+2

    def _parent(self, i):
        return (i-1) // 2

    def _update_cumulative(self, i):
        value_left = self.data[self._left(i)]
        value_right = self.data[self._right(i)]
        self.data[i] = value_left + value_right

        if i == 0: # the root of the tree
            return
        else: # update the parent
            self._update_cumulative(self._parent(i)) 

    def _is_leaf(self, i):
        # it is a leaf if it's stored in the last self.n_bins positions
        return i >= self.size - self.n_bins 

    def _importance_sampling(self, priority, i=0):
        # https://adventuresinmachinelearning.com/sumtree-introduction-python/
        if self._is_leaf(i):
            # return transition to which i corresponds
            return i - (self.size - self.n_bins), self.data[i] 
        else:
            value_left = self.data[self._left(i)]
            # value_right = self.data[self._right(i)]
            
            if priority < value_left:
                return self._importance_sampling(priority, self._left(i))
            else: # priority >= value_left
                return self._importance_sampling(priority-value_left, self._right(i))

    def get_sum(self):
        return self.data[0]        

    def set_priority(self, idx, priority):
        # where is the leaf stored on the array
        pos = self.size - self.n_bins + idx

        self.data[pos] = priority
        self._update_cumulative(self._parent(pos))

    def sample_batch(self, k):
        rng = self.get_sum() / k
        # low variance sampling like in particle filter
        unif = np.random.uniform() * rng
        
        idxes = np.zeros(k, dtype=np.uint32)
        Ps = np.zeros(k)

        for i in range(k):
            idxes[i], Ps[i]  = self._importance_sampling(unif)
            unif += rng
        return idxes, Ps

