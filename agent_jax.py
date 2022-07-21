import jax.numpy as jnp
from jax import grad
from jax.nn import softmax
# Doing different classes for now because it seems hard to unify the exact class variables they'll need otherwise (e.g. lambda, gradients)
# TODO add random initialization later

class Agent():
    def __init__(self, theta, h, pi, gamma, lr):
        self.theta = theta
        self.h = h
        self.pi = pi
        self.lr = lr
    
class TD(Agent):
    def __init__(self, theta, h, pi, gamma, lr, on_policy, n, lambduh):
        super(TD, self).__init__(theta, h, pi, gamma, lr)
        # on_policy: indicates q vs sarsa
        self.on_policy = on_policy
        self.n = n
        self.lambduh = lambduh
        # self.grads = 
    # get estimate of gradient of MSE wrt params
    
class REINFORCE(Agent):
    def __init__(self, theta, h, pi, gamma, lr, discount_updates):
        super(TD, self).__init__(theta, h, pi, gamma, lr)
        self.discount_updates = discount_updates
        self.trace = jnp.zeros_like(theta)
        
    def get_action(self, obs):
        action_probs = __get_action_probs__(obs, self.theta)
        action = 
        self.trace += grad(__get_action_probs_log__, argnums=1)(obs, self.theta, action)
        pass
    
    def __get_action_probs__(self, obs, theta):
        return pi(h(theta, obs))
    
    def __get_action_probs_log__(self, obs, theta):
        return jnp.log(__get_action_probs__(self, obs, theta))
    
    def __get_action_prob_log__(self, obs, theta, action):
        return jnp.log(__get_action_probs__(self, obs, theta))[action]
    
    def update(self, r):
        
    
def new_agent(policy_name, pi_name, alg_name, theta_dims, pi_hyperparams, alg_hyperparams):
    if policy_name.lower() == 'tabular':
        h = h_tabular
        theta = jnp.zeros((*theta_dims['input'], theta_dims['output']), dtype=float)
    elif policy_name.lower() == 'linear':
        h = h_linear
        theta = jnp.zeros((*theta_dims['input'], theta_dims['output']), dtype=float)
    else:
        raise NotImplementedError
    
    if (''.join([i for i in pi_name if i.isalpha()])).lower() == 'epsilongreedy':
        pi = pi_epsilon_greedy
    elif pi_name == 'softmax':
        pi = pi_softmax
    else:
        raise NotImplementedError
    
    if alg_name.lower() == 'sarsa':
        return TD(theta, h, pi, True, alg_hyperparams['n'], alg_hyperparams['lambda'])
    elif (''.join([i for i in alg_name if i.isalpha()])).lower() == 'qlearning':
        return TD(theta, h, pi, False, alg_hyperparams['n'], alg_hyperparams['lambda'])
    elif alg_name.lower() == 'reinforce':
        raise NotImplementedError
        
def h_tabular(theta, obs):
    return theta[obs]

def h_linear(theta, obs):
    # theta: (rep_size, |A|)
    # obs: (rep_size)
    return jnp.squeeze(jnp.expand_dims(obs, 0) @ theta) # (|A|)

def pi_epsilon_greedy(preferences, epsilon):
    def pi_epsilon_greedy_curried(preferences):
        probs = jnp.zeros_like(preferences) + epsilon/len(preferences)
        probs[jnp.argmax(preferences)] += 1-epsilon+epsilon/len(preferences)
        return probs
    return pi_epsilon_greedy_curried

def pi_softmax(preferences, temperature):
    def pi_softmax_curried(preferences):
        preferences *= temperature
        return softmax(preferences)
    return pi_softmax_curried