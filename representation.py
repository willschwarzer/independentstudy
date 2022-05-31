from abc import abstractmethod

class StateRep:
    def __init__(self):
        pass
    @abstractmethod
    def get_rep(self, obs):
        pass

class FourierRep(StateRep):
    def __init__(self, bounds, sin, cos, n_dims):
        # TODO
        pass
    def get_rep(self, obs):
        # TODO
        pass

class TabularRep(StateRep):
    def __init__(self):
        # TODO
        pass
    def get_rep(self, obs):
        # TODO
        pass
    
class IdentityRep(StateRep):
    def __init__(self, obs_dim):
        self.rep_dim = obs_dim
    def get_rep(self, obs):
        return obs
    

def new_rep_fn(name, obs_dim):
    if name.lower() == "identity":
        return IdentityRep(obs_dim)
    else:
        raise NotImplementedError