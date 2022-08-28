import torch
# Doing different classes for now because it seems hard to unify the exact class variables they'll need otherwise (e.g. lambduh, gradients)
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
    
    
class SARSAQ(Agent):
    def __init__(self, theta, h, pi, gamma, lr, on_policy, expected, n, lambduh):
        super(TD, self).__init__(theta, h, pi, gamma, lr)
        # on_policy: indicates q vs sarsa
        self.on_policy = on_policy
        self.expected = expected
        self.n = n
        self.lambduh = lambduh
        self.rt = None
        self.st = None
        self.at = None
        self.theta_grad = torch.zeros_like(self.theta)
        
        if self.on_policy and self.expected:
            raise Exception("That doesn't make sense.")
        # self.grads = 
    # get estimate of gradient of MSE wrt params
    
    def get_action(self, obs):
        # When updating, obs = s_{t+1}, so we need to observe qs[action], i.e. q(s_{t+1}, a_{t+1})
        qs = self.h(self.theta, obs)
        # Make sure epsilon greedy doesn't throw off autograd
        action_probs = self.pi(qs)
        action = torch.multinomial(action_probs, 1).squeeze()
        if self.on_policy and not self.expected and self.step > 0:
            self.__update__(qs[action])
        qs[action].backward()
        self.theta_grad += self.theta.grad
        self.st = obs
        self.at = action
        return action
    
    def update(self, obs, r, done):
        # obs = s_{t+1}, self.st = s_t, self.at = a_t, self.rt = r_{t-1}
        # r = r_t
        self.rt = r
        if not self.on_policy:
            qs = self.h(self.theta, obs)
            qp = torch.max(qs)
            self.__update__(qp)
            pass
        elif self.expected:
            qs = self.h(self.theta, obs)
            action_probs = self.pi(qs)
            qp = torch.dot(qs, action_probs)
            self.__update__(qp)
        elif done:
            qp = 0
            self.__update__(qp)
        self.step += 1
    
    def __update__(self, qp):
        with torch.no_grad():
            delta = self.rt + self.gamma*qp - self.h(self.theta, self.st)[self.at]
            print(delta)
            self.theta.grad = -delta*self.theta_grad
            self.optim.step()
            self.optim.zero_grad()
            self.theta_grad *= self.lambduh*self.gamma
            
    def reset(self):
        print("reset")
        self.theta_grad = torch.zeros_like(self.theta)
        self.step = 0
        
class _TD():
    def __init__(self, theta, h, gamma, optim, lambduh):
        self.theta = theta
        self.h = h
        self.gamma = gamma
        self.optim = optim
        self.lambduh = lambduh
        self.last_v = None
        self.theta_grad = torch.zeros_like(theta) # As in other lambduh methods,
        # we can't accumulate in the actual tensor gradient because 
        # we need to multiply the actual gradient by the td error
        
    def get_error_and_update(self, obs, r, done):
        # what do we need for TD? Just reward and state, I think, and we store the previous state's value
        # We need to get gradients, though, and right now we're calling it inside of a torch.no_grad()
        # I think we can move that to just the optim step at the end
        # I don't think we use done
        assert self.theta.grad is not None
        with torch.no_grad():
            # We're currently calculating h of each obs twice. Can fix later
            # breakpoint()
            td_error = r + self.gamma*self.h(self.theta, obs) - self.last_v
            # if torch.any(torch.abs(self.theta_grad) > 1):
            #     breakpoint()
            self.theta.grad = -td_error * self.theta_grad
            self.optim.step()
            self.theta_grad *= self.lambduh*self.gamma
        return td_error
    
    def get_value(self, obs):
        # assert bool(self.lambduh) == torch.any(self.theta_grad), f"{self.lambduh} {self.theta_grad}"
        self.optim.zero_grad()
        value = self.h(self.theta, obs)
        value.backward()
        self.theta_grad += self.theta.grad
        self.last_v = value
        return value
    
    def reset(self):
        self.theta_grad *= 0
        self.last_v = None
        
    
class REINFORCE(Agent):
    def __init__(self, theta, h, pi, gamma, optim, critic, discount_updates, online):
        super(REINFORCE, self).__init__(theta, h, pi, gamma, optim)
        self.discount_updates = discount_updates
        self.theta_grad = torch.zeros_like(theta)
        self.theta_grad_accumulant = torch.zeros_like(theta)
        self.online = online
        self.critic = critic
        self.critic_value = None # Stores value of last state
        
    def get_action(self, obs):
        if self.theta.grad is not None:
            self.optim.zero_grad()
        action_probs = self.__get_action_probs__(obs, self.theta)
        action = torch.multinomial(action_probs, 1).squeeze()
        log_action_prob = torch.log(action_probs[action])
        log_action_prob.backward()
        self.theta_grad += self.theta.grad
        if self.critic is not None:
            self.critic_value = self.critic.get_value(obs)
        return action
    
#     def __get_action_probs_log__(self, obs, theta):
#         return torch.log(__get_action_probs__(self, obs, theta))
    
#     def __get_action_prob_log__(self, obs, theta, action):
#         return torch.log(__get_action_probs__(self, obs, theta))[action]
    
    def update(self, obs, r, done):
        if not self.discount_updates:
            self.theta_grad_accumulant += -r*self.theta_grad
            if self.critic is not None:
                self.theta_grad_accumulant += 0*self.critic_value*self.theta_grad
                self.critic.get_error_and_update(obs, r, done)
            self.theta_grad *= self.gamma
        else:
            if self.critic is not None:
                raise NotImplementedError("need to implement baseline for this")
            self.theta_grad_accumulant += -r*(self.gamma**self.step)*self.theta_grad
        if self.online or done:
            with torch.no_grad():
                self.theta.grad = torch.clone(self.theta_grad_accumulant)
                self.optim.step()
                # print(self.theta, self.theta.grad)
                self.theta_grad_accumulant *= 0
        self.step += 1
        return
    
    def reset(self):
        # print(self.theta)
        self.theta_grad = torch.zeros_like(self.theta)
        self.step = 0
        if self.critic is not None:
            self.critic.reset()
        

class AC(Agent):
    def __init__(self, theta, h, pi, gamma, optim, critic, lambduh):
        super().__init__(theta, h, pi, gamma, optim)
        self.critic = critic
        self.critic_value = None
        self.lambduh = lambduh
        self.theta_grad = torch.zeros_like(theta)
        
    def get_action(self, obs):
        # if self.theta.grad is not None:
        self.optim.zero_grad()
        action_probs = self.__get_action_probs__(obs, self.theta)
        action = torch.multinomial(action_probs, 1).squeeze()
        log_action_prob = torch.log(action_probs[action])
        log_action_prob.backward()
        self.theta_grad += self.theta.grad
        if self.critic is not None:
            self.critic_value = self.critic.get_value(obs)
        return action
    
    def update(self, obs, r, done):
        with torch.no_grad():
            td_error = self.critic.get_error_and_update(obs, r, done)
            self.theta.grad = -td_error*self.theta_grad
            self.optim.step()
            self.theta_grad *= self.lambduh*self.gamma
        self.step += 1
        return
    
    def reset(self):
        self.theta_grad *= 0
        self.step = 0
        self.critic.reset()
            
        
def new_agent(policy_name, pi_name, alg_name, optim_name, gamma, theta_dims, pi_hyperparams, alg_hyperparams):
    if policy_name.lower() == 'tabular':
        h = h_tabular
        theta = torch.zeros((*theta_dims['input'], theta_dims['output']), dtype=float, requires_grad=True)
    elif policy_name.lower() == 'linear':
        h = h_linear
        # NOTE: just got rid of the star here, assuming you'll never have dimension of dimensions in linear
        theta = torch.zeros((theta_dims['input'], theta_dims['output']), dtype=float, requires_grad=True)
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
        
    if 'baseline' in alg_name.lower() or 'critic' in alg_name.lower() or alg_name.lower() == 'ac':
        if alg_hyperparams['critic'].lower() == 'tabular':
            h_critic = h_tabular
            theta_critic = torch.zeros((*theta_dims['input'],), dtype=float, requires_grad=True)
        elif alg_hyperparams['critic'].lower() == 'linear':
            h_critic = h_linear
            theta_critic = torch.zeros((theta_dims['input']), dtype=float, requires_grad=True)
        else:
            raise NotImplementedError
            
        if alg_hyperparams['critic_optim'].lower() == 'sgd':
            optim_critic = torch.optim.SGD([theta_critic], lr=alg_hyperparams['critic_lr'])
        elif alg_hyperparams['critic_optim'].lower() == 'adam':
            optim_critic = torch.optim.Adam([theta_critic], lr=alg_hyperparams['critic_lr'])
            
        critic = _TD(theta_critic, h_critic, gamma, optim_critic, alg_hyperparams['critic_lambduh'])
    else:
        critic = None
    
    if alg_name.lower() == 'td':
        return SARSAQ(theta, h, pi, gamma, optim, alg_hyperparams['on_policy'], alg_hyperparams['expected'], alg_hyperparams['n'], alg_hyperparams['lambduh'])
    elif 'reinforce' in alg_name.lower():
        return REINFORCE(theta, h, pi, gamma, optim, critic, alg_hyperparams['discount_updates'], alg_hyperparams['online'])
    elif (''.join([i for i in alg_name if i.isalpha()])).lower() in ('actorcritic', 'ac'):
        return AC(theta, h, pi, gamma, optim, critic, alg_hyperparams['lambduh'])
    else:
        raise NotImplementedError
        
def h_tabular(theta, obs):
    return theta[obs]

def h_linear(theta, obs):
    # theta: (rep_size, |A|)
    # obs: (rep_size)
    torch_obs = torch.DoubleTensor(obs)
    return torch.squeeze(torch.unsqueeze(torch_obs, 0) @ theta) # (|A|)

def pi_epsilon_greedy(epsilon):
    def pi_epsilon_greedy_curried(preferences):
        probs = torch.zeros_like(preferences) + epsilon/len(preferences)
        # Check this line, also do equal argmax
        max_idxs = torch.where(preferences == preferences.max())[0]
        if len(max_idxs) == 0:
            print(preferences)
        probs[max_idxs] += (1-epsilon)/len(max_idxs)
        return probs
    return pi_epsilon_greedy_curried

def pi_softmax(temperature):
    def pi_softmax_curried(preferences):
        preferences_with_temp = preferences*temperature
        return torch.nn.Softmax()(preferences_with_temp)
    return pi_softmax_curried
