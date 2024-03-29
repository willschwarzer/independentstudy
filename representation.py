from abc import abstractmethod
from itertools import product
import numpy as np

class StateRep:
    def __init__(self):
        pass
    @abstractmethod
    def get_rep(self, obs):
        pass

class FourierRep(StateRep):
    def __init__(self, obs_dim, bounds, d):
        # XXX just doing cos for now for simplicity, as described in https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf, around equation (4)
        # one way: just make array of all pi*c_i (rep_d, obs_d), then mult with obs (obs_d, 1), then take cos
        coefficients = product(range(d), repeat=obs_dim)
        coefficients = np.pi*np.array([coef for coef in coefficients], dtype=float)
        self.coefficients = coefficients
        self.bounds = bounds
        self.rep_dim = d**obs_dim
    def get_rep(self, obs):
        assert obs.shape[0] == self.coefficients.shape[1]
        obs = (obs - self.bounds[0])/(self.bounds[1] - self.bounds[0])
        obs_expanded = np.expand_dims(obs, 1)
        return np.cos(np.squeeze(self.coefficients @ obs_expanded))

class DiscreteRep(StateRep):
    def __init__(self, bounds, d):
        self.bins = np.linspace(bounds[0], bounds[1], d).transpose()
        self.rep_dim = (d+1,) * len(bounds[0])
        print(self.rep_dim)
    def get_rep(self, obs):
        ret = tuple([np.digitize(obs[i], self.bins[i]) for i in range(len(obs))])
        return ret
    
class IdentityRep(StateRep):
    def __init__(self, obs_dim):
        self.rep_dim = obs_dim
    def get_rep(self, obs):
        return obs
    

def new_rep_fn(name, obs_dim, obs_bounds, rep_hyperparams):
    if name.lower() == "identity":
        return IdentityRep(obs_dim)
    elif name.lower() == "fourier":
        return FourierRep(obs_dim, obs_bounds, rep_hyperparams['d'])
    elif name.lower() == "discrete":
        return DiscreteRep(obs_bounds, rep_hyperparams['d'])
    else:
        raise NotImplementedError