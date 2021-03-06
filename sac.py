# This implementation is written with the help from a tutorial from the Youtube channel "Machine Learning with Phil"
# https://www.youtube.com/watch?v=ioidsRlf79o

import argparse
import pybullet_envs
import gym
import numpy as np
from gym import wrappers
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import wandb
import time
import tqdm
import random

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class ReplayBuffer():
    def __init__(self, mem_size, state_dim, action_dim):
        self.mem_size = mem_size
        self.cntr = 0
        self.state = np.zeros((self.mem_size, *state_dim))
        self.new_state = np.zeros((self.mem_size, *state_dim))
        self.action = np.zeros((self.mem_size, action_dim))
        self.reward = np.zeros(self.mem_size)
        self.terminal = np.zeros(self.mem_size, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        index = self.cntr % self.mem_size

        self.state[index] = state
        self.new_state[index] = state_
        self.action[index] = action
        self.reward[index] = reward
        self.terminal[index] = done

        self.cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state[batch]
        states_ = self.new_state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        dones = self.terminal[batch]

        return states, actions, rewards, states_, dones


###################### Define Agent ########################
class Critic(nn.Module):
    def __init__(self, beta, input_dim, action_dim, fc1_dim=256, fc2_dim=256,
                 name='critic', chkpt_dir='tmp'):
        super(Critic, self).__init__()

        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dim[0] + action_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = device

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dim, fc1_dim=256, fc2_dim=256,
                 name='value', chkpt_dir='tmp'):
        super(ValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, fc2_dim)
        self.v = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = device

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, alpha, input_dim, max_action, fc1_dim=256,
                 fc2_dim=256, action_dim=2, name='actor', chkpt_dir='tmp'):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.mu = nn.Linear(self.fc2_dim, self.action_dim)
        self.sigma = nn.Linear(self.fc2_dim, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


######################### SAC #############################
class SAC():
    def __init__(self, alpha=0.04, beta=0.0003, input_dim=[8],
                 env=None, gamma=0.99, action_dim=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1,
                 seed=None):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dim, action_dim)
        self.batch_size = batch_size
        self.action_dim = action_dim

        if not os.path.exists('ckpt_' + str(seed)):
            os.mkdir('ckpt_' + str(seed))

        self.actor = Actor(alpha, input_dim, action_dim=action_dim,
                           name='actor', max_action=env.action_space.high,
                           chkpt_dir='ckpt_' + str(seed))
        self.critic_1 = Critic(beta, input_dim, action_dim=action_dim,
                               name='critic_1', chkpt_dir='ckpt_' + str(seed))
        self.critic_2 = Critic(beta, input_dim, action_dim=action_dim,
                               name='critic_2', chkpt_dir='ckpt_' + str(seed))
        self.value = ValueNetwork(beta, input_dim, name='value', chkpt_dir='ckpt_' + str(seed))
        self.target_value = ValueNetwork(beta, input_dim, name='target_value',
                                         chkpt_dir='ckpt_' + str(seed))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.add(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....', flush=True)
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....', flush=True)
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
        return {'act_loss': actor_loss.item(),
                'val_loss': value_loss.item(),
                'crt_loss': critic_loss.item()}


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def eval_policy(policy, eval_env, eval_episodes=10):
    avg_reward = 0
    for episode in range(0, eval_episodes):
        observation = eval_env.reset()
        done = False
        while not done:
            action = policy.choose_action(observation)
            observation_, reward, done, info = eval_env.step(action)
            avg_reward += reward
            observation = observation_

    return {'returns': avg_reward/eval_episodes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', default='SAC')
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--n_minibatch', type=int, default=32,
                        help='the number of mini batch')
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument('--num_timesteps_per_env', type=int, default=2048,
                        help='the number of timesteps per environment to collect during interacting with environments.')
    parser.add_argument('--max_timesteps', default=2000000)
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
    parser.add_argument('--reward_scale', type=int, default=1, help='reward_scale')

    args = parser.parse_args()
    if args.seed == 0:
        args.seed = int(time.time())

    random.seed(args.seed)
    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    args.batch_size = int(args.num_timesteps_per_env)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    num_updates = args.max_timesteps // args.batch_size

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    eval_env = gym.make(args.env)
    eval_env.seed(args.seed + 100)
    eval_env.action_space.seed(args.seed + 100)
    eval_env.observation_space.seed(args.seed + 100)

    agent = SAC(input_dim=env.observation_space.shape, env=env,
                action_dim=env.action_space.shape[0],
                seed=args.seed, reward_scale=args.reward_scale)
    max_timesteps = args.max_timesteps
    experiment_name = f"{args.env}_{args.algo_name}_{args.reward_scale}_{args.seed}_{int(time.time())}"

    wandb.init(project='rl_project', config=vars(args), name=experiment_name)
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    observation = env.reset()
    done = False
    episode_timesteps = 0
    for i in tqdm.tqdm(range(1, max_timesteps)):
        episode_timesteps += 1
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        done_float = float(done) if episode_timesteps < env._max_episode_steps else 0.
        agent.remember(observation, action, reward, observation_, done_float)

        if not load_checkpoint:
            update_info = agent.learn()
        if i % args.eval_freq == 0:
            eval_info = eval_policy(agent, eval_env)
            eval_info.update({'timesteps': i})
            print(f"Time steps: {i}, Eval_info: {eval_info}", flush=True)
            wandb.log({'eval/': eval_info})
            if eval_info['returns'] > best_score:
                best_score = eval_info['returns']
                if not load_checkpoint:
                    agent.save_models()
        observation = observation_
        if done:
            observation, done, score = env.reset(), False, 0
            episode_timesteps = 0


    if not load_checkpoint:
        x = [i + 1 for i in range(max_timesteps)]
        plot_learning_curve(x, score_history, figure_file)
