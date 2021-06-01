from core.agents.off_policy_algo import OffPolicyAlgorithm
import torch.nn.functional as F
import numpy as np
import torch
from gym.spaces import Box
from torch import nn as nn
from core.utils.mpi_utils import mpi_avg_grads, MPI
from core.utils.sac_utils import initialize_last_layer, initialize_hidden_layer, TanhNormal


device = "cpu"

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
rank = comm.Get_rank()


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # hidden layers definition
        self.l1 = nn.Linear(observation_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_mean = nn.Linear(256, action_dim)

        # std layer definition
        self.l3_log_std = nn.Linear(256, action_dim)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3_mean)
        initialize_last_layer(self.l3_log_std)

        self.max_action = max_action

    def forward(self, x, deterministic=False, return_log_prob=False):
        # forward pass
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean, log_std = self.l3_mean(x), self.l3_log_std(x)

        # compute std
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        # compute other relevant quantities
        log_prob, entropy, mean_action_log_prob, pre_tanh_value = None, None, None, None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std, device)
            if return_log_prob:
                action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.rsample()
        action = action * self.max_action
        return action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(observation_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(observation_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        # weights initialization
        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3)

        initialize_hidden_layer(self.l4)
        initialize_hidden_layer(self.l5)
        initialize_last_layer(self.l6)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class SAC(OffPolicyAlgorithm):
    def __init__(self, obs_space, action_space, config):

        observation_dim = obs_space.shape[0]
        assert isinstance(action_space, Box), 'action space is not Continuous'
        action_dim = action_space.shape[0]
        max_action = action_space.high[0]

        self.actor = Actor(observation_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["policy_lr"])

        self.critic = Critic(observation_dim, action_dim).to(device)
        self.critic_target = Critic(observation_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

        # List of networks that'll be synchronized before training
        self.networks_to_sync = [self.actor, self.critic, self.critic_target]

        super(SAC, self).__init__(actor=self.actor, critic=self.critic, name='SAC')

        self.soft_target_tau = config["soft_target_tau"]
        self.target_update_period = config["target_update_period"]

        self.action_dim = action_dim

        self.use_automatic_entropy_tuning = config["use_automatic_entropy_tuning"]
        if self.use_automatic_entropy_tuning:
            if config["target_entropy"]:
                self.target_entropy = config["target_entropy"]
            else:
                self.target_entropy = -np.prod(self.action_dim).item()  # heuristic value from Tuomas
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["policy_lr"])

        self.discount = config["discount"]
        self.reward_scale = config["reward_scale"]
        self._n_train_steps_total = 0

    def select_action(self, observation, deterministic=True):
        observation = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        return self.actor(observation, deterministic=deterministic)[0].cpu().data.numpy().flatten()

    def train_on_batch(self, batch):
        # increment iteration counter
        self._n_train_steps_total += 1

        # read batch
        rewards = torch.FloatTensor(batch['rewards']).view(-1, 1).to(device)
        done = torch.FloatTensor(1. - batch['terminals']).view(-1, 1).to(device)
        observations = torch.FloatTensor(batch['observations']).to(device)
        actions = torch.FloatTensor(batch['actions']).to(device)
        new_observations = torch.FloatTensor(batch['new_observations']).to(device)

        # compute actor and alpha losses
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.actor(observations, return_log_prob=True)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            # sync_tensor_grads(self.log_alpha)
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1

        Q1_new_actions, Q2_new_actions = self.critic_target(observations, new_obs_actions)
        Q_new_actions = torch.min(Q1_new_actions, Q2_new_actions)
        actor_loss = (alpha*log_pi - Q_new_actions).mean()

        # compute critic losses
        current_Q1, current_Q2 = self.critic(observations, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.actor(new_observations, return_log_prob=True)
        target_Q1, target_Q2 = self.critic_target(new_observations, new_next_actions)
        target_Q_values = torch.min(target_Q1, target_Q2) - alpha * new_log_pi

        target_Q = self.reward_scale * rewards + (done * self.discount * target_Q_values)
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())

        # optimization
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        mpi_avg_grads(self.actor)
        self.actor_optimizer.step()

        # soft updates
        if self._n_train_steps_total % self.target_update_period == 0:
            self.soft_update_from_to(self.critic, self.critic_target, self.soft_target_tau)



