import wandb
import argparse
import ray
import psutil
import time
from tqdm import tqdm
import numpy as np
# import jax.numpy as jnp
# from jax import grad, random, jit
# from jax.example_libraries.optimizers import adam
# from jax.nn import sigmoid
# from jax.nn import relu
from environment import new_env
from representation import new_rep_fn
from agent import new_agent

# wandb inspiration from https://colab.research.google.com/drive/1lZT-t2eV96uDNftr7ojcvMyqamSabKh1?usp=sharing#scrollTo=vdzejSm81kwI, by Yash Kotadia

@ray.remote
def main(args):
    # Might eventually want to move away from args to just configs?
    # TODO: standardize strings in config
    run = wandb.init(project=args.wandb_project, reinit=True)
    wandb.config.update(vars(args), allow_val_change=True)
    # wandb.log(vars(args))
    # How do we get dimensions of h? Well, from the size of rep's output,
    # as well as the number of actions
    # Might want to make max_steps an arg at some point
    env = new_env(wandb.config.env)
    rep_fn = new_rep_fn(wandb.config.rep, env.obs_dim)
    alg_hyperparams = {'lr': wandb.config.learning_rate, 
            'n': wandb.config.n_step_n, 
            'lambda': wandb.config.lambduh, 
            'discount_updates': wandb.config.discount_updates,
            'online': wandb.config.online} # TODO add ANN params as well
    pi_hyperparams = {'temperature': wandb.config.softmax_temp, 'epsilon': wandb.config.epsilon}
    # ANN params mean that these aren't just algorithm hyperparameters... they're also for theta/h, and gradient descent
    # Which also raises the question of how we'll treat h in the case of ANNs - the forward function on a Flax net or whatever?
    # We'll also have more params here later, e.g. for DQN and its replay buffer or whatever
    # XXX should do action dim instead of num actions at some point... maybe. Could also just flatten actions
    # Something like action dim would only be important if the agent had an action representation or something like it, e.g.
    # saying that clicking pixels next to each other is more similar than clicking distant pixels... or maybe not
    # Could be important in some envs anyway, just not now
    theta_dims = {'input': rep_fn.rep_dim, 'output': env.num_actions} # Can add ANN architecture params here later
    agent = new_agent(wandb.config.policy, wandb.config.action_selection, wandb.config.learning_rule, wandb.config.optimizer_name, env.gamma, theta_dims, pi_hyperparams, alg_hyperparams)
    # TODO allow criterion-based stopping (maybe something for the agent to determine?)
    num_episodes = wandb.config.num_episodes
    rets_by_ep, n_steps_by_ep = np.zeros(num_episodes, dtype=float), np.zeros(num_episodes, dtype=int)
    for episode in tqdm(range(num_episodes), mininterval=10.0):
        ret, n_steps = train(agent, rep_fn, env)
        #if args.num_trials < 5 or psutil.cpu_count() < 5:
        wandb.log({'num_steps':n_steps, 'ret':ret})
        rets_by_ep[episode] = ret
        n_steps_by_ep[episode] = n_steps
        agent.reset()
        env.reset()
    #if args.num_trials >= 5 and psutil.cpu_count() >= 5:
        #wandb.log({
    #wandb.log({'num_steps': n_steps_by_ep, 'ret': rets_by_ep})
    wandb.log({'end_params': agent.theta})
    print(agent.theta)
    env.display_policy(agent.theta)
    ave_end_ret = np.mean(rets_by_ep[-num_episodes//10:])
    ave_end_n_steps = np.mean(n_steps_by_ep[-num_episodes//10:])
    wandb.log({'ave_end_ret': ave_end_ret, 'ave_end_num_steps': ave_end_n_steps})
    run.finish()
    
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-nt', '--num-trials', type=int, default=1, help="Number of trials with this config")
    parser.add_argument('-e', '--env', type=str, help="Environment to train/test on")
    parser.add_argument('-r', '--rep', type=str, help="Non-parametric representation observed by the agent")
    parser.add_argument('-p', '--policy', type=str, help="Structure of h_theta: obs --> action preferences") # Using h to match book
    # Will probably need extra policy args for when the policy is a network
    # May just need extra agent-specific information in general (e.g. lambda)
    parser.add_argument('-a', '--action-selection', type=str, help="pi: h_theta --> probabilities")
    parser.add_argument('-l', '--learning-rule', type=str, help="Rule for updating h_theta based on trajectory so far")
    parser.add_argument('-lr', '--learning-rate', type=float, help="Learning rate")
    parser.add_argument('-st', '--softmax-temperature', type=float, help="Temperature of softmax")
    parser.add_argument('-du', '--discount-updates', type=bool, help="Whether or not to use gamma in policy gradient")
    parser.add_argument('-ne', '--num-episodes', type=int, help="Number of training episodes")
    parser.add_argument('-on', '--optimizer-name', type=str, help="Name of optimizer to use")
    parser.add_argument('-wp', '--wandb-project', type=str, default='uncategorized', help="WAndB project name")
    parser.add_argument('-o', '--online', type=bool, help="Whether or not learning algorithm updates every step")
    parser.add_argument('-op', '--on-policy', type=bool, help="Whether TD methods are SARSA or Q-learning")
    parser.add_argument('-ex', '--expected', type=bool, help="whether SARSA is expected SARSA or original")
    args = parser.parse_args()
    return args

def train(agent, rep_fn, env):
    obs = env.get_observation()
    rep = rep_fn.get_rep(obs)
    done = env.is_terminal()
    max_steps = env.max_steps or 10**100
    ret = 0
    for step in range(max_steps):
        action = agent.get_action(rep)
        obs, reward, done = env.step(action)
        agent.update(obs, reward, done)
        rep = rep_fn.get_rep(obs)
        ret += (env.gamma**step)*reward
        if done:
            return ret, step
        
if __name__ == "__main__":
    args = parse_args()
    if args.num_trials > 1:
        num_trials_remaining = args.num_trials
        print("Number of cpus: ", psutil.cpu_count())
        while num_trials_remaining > 0:
            start = time.time()
            num_threads = min(psutil.cpu_count()-1, num_trials_remaining)
            ray.init(num_cpus=num_threads)
            ray.get([main.remote(args) for i in range(num_threads)])
            num_trials_remaining -= num_threads
            ray.shutdown()
            print("num trials remaining: ", num_trials_remaining)
            print("time taken: ", time.time()-start)
    else:
        ray.get(main.remote(args))
    
    
    # some_vars = {'var1': 3.7, 'var2': False, 'var3': 'yo'}
    # wandb.log(some_vars)
    # some_vars = {'var1': 4.2, 'var2': True, 'var3': 'yoooo'}
    # wandb.log(some_vars)
    # some_vars = {'var1': config.env}
    # wandb.log(some_vars)
    # for i in range(100):
    #     some_vars = {'reward': 5*i}
    #     wandb.log(some_vars)
