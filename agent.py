import torch
# Doing different classes for now because it seems hard to unify the exact class variables they'll need otherwise (e.g. lambda, gradients)
# TODO add random initialization later

class Agent():
    def __init__(self, theta, h, pi, gamma, optim):
        self.theta = theta
        self.h = h
        self.pi = pi
        self.gamma = gamma
        self.optim = optim
        self.step = 0
    
    def __get_action_probs__(self, obs, theta):
        return self.pi(self.h(theta, obs))
    
    
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
    def __init__(self, theta, h, pi, gamma, optim, discount_updates, online):
        super(REINFORCE, self).__init__(theta, h, pi, gamma, optim)
        self.discount_updates = discount_updates
        self.trace = torch.zeros_like(theta)
        self.theta_grad_accumulant = torch.zeros_like(theta)
        self.online = online
        
    def get_action(self, obs):
        if self.theta.grad is not None:
            self.optim.zero_grad()
        action_probs = self.__get_action_probs__(obs, self.theta)
        action = torch.multinomial(action_probs, 1).squeeze()
        log_action_prob = torch.log(action_probs[action])
        log_action_prob.backward()
        self.trace += self.theta.grad
        return action
    
#     def __get_action_probs_log__(self, obs, theta):
#         return torch.log(__get_action_probs__(self, obs, theta))
    
#     def __get_action_prob_log__(self, obs, theta, action):
#         return torch.log(__get_action_probs__(self, obs, theta))[action]
    
    def update(self, r, done):
        with torch.no_grad():
            if not self.discount_updates:
                self.theta_grad_accumulant = -r*self.trace
                self.trace *= self.gamma
            else:
                self.theta_grad_accumulant = -r*(self.gamma**self.step)*self.trace

            if self.online or done:
                self.theta.grad = torch.copy(self.theta_grad_accumulant)
                self.optim.step()
                self.theta_grad_accumulant *= 0
        self.step += 1
        return
    
    def reset(self):
        self.trace = torch.zeros_like(self.theta)
        self.step = 0
    
def new_agent(policy_name, pi_name, alg_name, optim_name, gamma, theta_dims, pi_hyperparams, alg_hyperparams):
    if policy_name.lower() == 'tabular':
        h = h_tabular
        theta = torch.zeros((*theta_dims['input'], theta_dims['output']), dtype=float, requires_grad=True)
    elif policy_name.lower() == 'linear':
        h = h_linear
        theta = torch.zeros((*theta_dims['input'], theta_dims['output']), dtype=float, requires_grad=True)
    else:
        raise NotImplementedError
    
    if (''.join([i for i in pi_name if i.isalpha()])).lower() == 'epsilongreedy':
        pi = pi_epsilon_greedy(pi_hyperparams['epsilon'])
    elif pi_name.lower() == 'softmax':
        pi = pi_softmax(pi_hyperparams['temperature'])
    else:
        raise NotImplementedError
        
    if optim_name.lower() == 'sgd':
        optim = torch.optim.SGD([theta], lr=alg_hyperparams['lr'])
    elif optim_name.lower() == 'adam':
        optim = torch.optim.Adam([theta], lr=alg_hyperparams['lr'])
    
    if alg_name.lower() == 'sarsa':
        return TD(theta, h, pi, True, alg_hyperparams['n'], alg_hyperparams['lambda'])
    elif (''.join([i for i in alg_name if i.isalpha()])).lower() == 'qlearning':
        return TD(theta, h, pi, False, alg_hyperparams['n'], alg_hyperparams['lambda'])
    elif alg_name.lower() == 'reinforce':
        return REINFORCE(theta, h, pi, gamma, optim, alg_hyperparams['discount_updates'], alg_hyperparams['online'])
    else:
        raise NotImplementedError
        
def h_tabular(theta, obs):
    return theta[obs]

def h_linear(theta, obs):
    # theta: (rep_size, |A|)
    # obs: (rep_size)
    return torch.squeeze(torch.unsqueeze(obs, 0) @ theta) # (|A|)

def pi_epsilon_greedy(epsilon):
    def pi_epsilon_greedy_curried(preferences):
        probs = torch.zeros_like(preferences) + epsilon/len(preferences)
        probs[torch.argmax(preferences)] += 1-epsilon+epsilon/len(preferences)
        return probs
    return pi_epsilon_greedy_curried

def pi_softmax(temperature):
    def pi_softmax_curried(preferences):
        preferences_with_temp = preferences*temperature
        return torch.nn.Softmax()(preferences_with_temp)
    return pi_softmax_curried
