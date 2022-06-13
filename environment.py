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
        self.update_state(self, action)
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
    pass

class CartPole(Environment):
    pass

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