from abc import abstractmethod
import gym
import numpy as np
import torch

class Environment:
    def __init__(self):
        self.n_actions = -1
        self.state = None # Could be cont., discrete, or combination? Presumably if combination, then a tuple of (discrete_dims, cont_dims)
        # self.reward_features = None # Same as above; for traditional IRL, as well as I believe the new approach, we assume we have access to this
        # self.reward_weights = np.zeros(len(self.reward_features), dtype = float)
        self.gamma = -1
        self.max_steps = None
        self.current_step = 0
        return
    def step(self, action):
        self.current_step += 1
        self.update_state(action)
        observation = self.get_observation()
        reward_features = self.get_reward_features()
        reward = np.dot(self.reward_weights, reward_features)
        done = self.is_terminal() or self.current_step == self.max_steps
        return observation, reward, done
    @abstractmethod
    def get_observation(self):
        pass
    @abstractmethod
    def is_terminal(self):
        pass
    @abstractmethod
    def get_reward_features(self):
        pass
    @abstractmethod
    def update_state(self, action):
        pass
    @abstractmethod
    def reset(self):
        # Note: this needs to reset current step at some point
        pass
    def generate_trajectories(self, policy, n_trajectories):
        pass #TODO
    def evaluate_policy(self, policy):
        pass #TODO

class Gridworld(Environment):
    def __init__(self):
        super(Gridworld, self).__init__()
        self.obs_dim = (5, 5)
        # self.obs_bounds = np.reshape(np.repeat(np.array([0, 1], dtype=float), 25), (5, 5, 2))
        self.obs_bounds = None
        # wrong dimensions right now, needs to be other way around (dimension 2 first)
        self.num_actions = 4
        self.state = (0, 0)
        self.reward_weights = np.array([-10, 10], dtype=float)
        self.water = (4, 2)
        self.goal = (4, 4)
        self.obstacles = [(2, 2), (3, 2)]
        self.gamma = 0.9
        self.p = self.__get_transitions__()
        return
    
    def get_reward_features(self):
        return np.array([self.state == self.water, self.state == self.goal], dtype=bool)
    
    def update_state(self, action):
        transition_probs = self.p[self.state][action]
        # ??? What is this [0] at the end?
        move = np.nonzero(np.random.multinomial(1, transition_probs))[0]
        if move == 0: 
            self.state = (self.state[0], self.state[1] + 1)
        elif move == 1: 
            self.state = (self.state[0] + 1, self.state[1])
        elif move == 2: 
            self.state = (self.state[0], self.state[1] - 1)
        elif move == 3: 
            self.state = (self.state[0] - 1, self.state[1])
        return
            
    def step(self, action):
        self.update_state(action)
        r_features = self.get_reward_features()
        r = np.dot(r_features, self.reward_weights)
        return self.get_observation(), r, self.is_terminal()
            
    def reset(self):
        self.state = (0, 0)
        return
    
    def __get_transitions__(self):
        p = np.zeros([5, 5, 4, 5])
        # Normally go right with 80% probability
        p[:, :, 0, 0] = 0.8
        # Down
        p[:, :, 1, 1] = 0.8
        # Left
        p[:, :, 2, 2] = 0.8
        # Up
        p[:, :, 3, 3] = 0.8
        # Add unintentional sideways movement
        p[:, :, 0, 1] = p[:, :, 0, 3] = 0.05
        # Down
        p[:, :, 1, 0] = p[:, :, 1, 2] = 0.05
        # Left
        p[:, :, 2, 1] = p[:, :, 2, 3] = 0.05
        # Up
        p[:, :, 3, 2] = p[:, :, 3, 0] = 0.05
        # Add walls and obstacles
        # Right
        p[:, 4, :, 0] = p[2:4, 1, :, 0] = 0
        # Down
        p[4, :, :, 1] = p[1, 2, :, 1] = 0
        # Left
        p[:, 0, :, 2] = p[2:4, 3, :, 2] = 0
        # Up
        p[0, :, :, 3] = p[4, 2, :, 3] = 0
        # Always stay still in the terminal state
        p[4, 4, :, :] = 0
        # Update stationary probs
        p[:, :, :, 4] = 1-np.sum(p, axis=3)
        
        return p
    
    def is_terminal(self):
        return self.state == self.goal
    
    def get_observation(self):
        return self.state
    
    def display_policy(self, policy):
        unicodes = ['\u2192', '\u2193', '\u2190', '\u2191']
        policy_actions = torch.argmax(policy, -1)
        for (row, actions) in enumerate(policy_actions):
            for (col, action) in enumerate(actions):
                char = unicodes[action] if (row, col) not in self.obstacles else ' '
                if row == col == 4 : char = 'G' 
                print(char, end='\t')
            print()

class MountainCar(Environment):
    def __init__(self):
        super().__init__()
        self._initial_state = np.array([-0.5, 0], dtype=float)
        self.obs_dim = 2
        self.obs_bounds = np.array([[-1.2, -0.07], [0.5, 0.07]], dtype=float)
        self.num_actions = 3
        self.actions = np.array([-1, 0, 1], dtype=float)
        self.state = np.copy(self._initial_state)
        self.reward_weights = np.array([-1], dtype=float)
        self.goal = 0.5
        self.x_min = -1.2
        self.x_max = self.goal
        self.v_min = -0.07
        self.v_max = 0.07
        self.gamma = 1.
        self.max_steps = 1000
        return
    
    def get_reward_features(self):
        return np.array([self.state[0] != self.goal], dtype=bool)
    
    def update_state(self, action):
        action_effect = self.actions[action]
        self.state[1] += 0.001*action_effect - 0.0025*np.cos(3.0*self.state[0])
        self.state[1] = np.clip(self.state[1], self.v_min, self.v_max)
        self.state[0] += self.state[1]
        self.state[0] = np.clip(self.state[0], self.x_min, self.x_max)
        return
            
    def reset(self):
        self.state = np.copy(self._initial_state)
        self.current_step = 0
        return
    
    def is_terminal(self):
        return self.state[0] >= self.goal
    
    def get_observation(self):
        return self.state
    
    def display_policy(self, policy):
        return

class CartPole(Environment):
    def __init__(self):
        super().__init__()
        self._initial_state = np.array([0, 0, 0, 0], dtype=float) # x, v, theta, omega
        self.obs_dim = 4
        self.obs_bounds = np.array([[-2.4, -10, -np.pi/12, -np.pi], [2.4, 10, np.pi/12, np.pi]], dtype=float)
        self.num_actions = 2
        self.actions = np.array([-1, 1], dtype=float)
        self.state = np.copy(self._initial_state)
        self.reward_weights = np.array([1], dtype=float)
        self.gamma = 1.
        self.dt_base = 0.02
        self.sim_steps_per_t = 1
        self.dt = 0.02/self.sim_steps_per_t
        self.max_steps = int(20/self.dt_base) + 10 # Not sure why the 10 here; just do 20 secs?
        self.l = 0.5
        self.g = 9.8
        self.muc = 0.0005
        self.mup = 0.000002
        self.m = 0.1
        self.mc = 1
        
    def get_reward_features(self):
        return np.array([True], dtype=bool)
    
    def update_state(self, action):
        action_effect = self.actions[action]
        x = self.state[0]
        v = self.state[1]
        theta = self.state[2]
        omega = self.state[3]
        force = action_effect * 10. # I'm not quite sure why this is 10
        for sim_step in range(self.sim_steps_per_t):
            domega = (self.g*np.sin(theta) + np.cos(theta)*(self.muc*np.sign(v) - force - self.m*self.l*(omega**2)*np.sin(theta)) / (self.m + self.mc) - self.mup*omega / (self.m*self.l)) / (self.l*(4.0 / 3.0 - self.m / (self.m + self.mc)*(np.cos(theta)**2)))
            a = (force + self.m*self.l*((omega**2)*np.sin(theta) - domega*np.cos(theta)) - self.muc*np.sign(v)) / (self.m + self.mc)
            omega += self.dt*domega
            v += self.dt*a
            theta += self.dt*omega
            x += self.dt*v
        self.state[:] = [x, v, theta, omega]
        np.clip(self.state, self.obs_bounds[0], self.obs_bounds[1])
        return
            
    def reset(self):
        self.state = np.copy(self._initial_state)
        self.current_step = 0
        return
    
    def is_terminal(self):
        return self.state[2] <= self.obs_bounds[0][2] or self.state[2] >= self.obs_bounds[1][2]
    
    def get_observation(self):
        return self.state
    
    def display_policy(self, policy):
        return

class Acrobot(Environment):
    pass

class InfiniteGridworld(Environment):
    def __init__(self, reward_weights):
        pass
        

class GymEnv(gym.Env):
    def __init__(self, base_env):
        super(GymEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5) # up, right, down, left, noop
        self.state_space = gym.spaces.a # TODO
        return
    

def new_env(env_name):
    if env_name.lower() == "gridworld":
        return Gridworld()
    elif (''.join([i for i in env_name if i.isalpha()])).lower() == 'mountaincar':
        return MountainCar()
    elif (''.join([i for i in env_name if i.isalpha()])).lower() == 'cartpole':
        return CartPole()
    else:
        raise NotImplementedError('Not yet implemented')
        
        


# class CustomEnv(gym.Env):
#   """Custom Environment that follows gym interface"""
#   metadata = {'render.modes': ['human']}

#   def __init__(self, arg1, arg2, ...):
#     super(CustomEnv, self).__init__()
#     # Define action and observation space
#     # They must be gym.spaces objects
#     # Example when using discrete actions:
#     self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#     # Example for using image as input:
#     self.observation_space = spaces.Box(low=0, high=255, shape=
#                     (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

#   def step(self, action):
#     # Execute one time step within the environment
#     ...
#   def reset(self):
#     # Reset the state of the environment to an initial state
#     ...
#   def render(self, mode='human', close=False):
#     # Render the environment to the screen
#     ...