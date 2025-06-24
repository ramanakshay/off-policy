import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from agent.sac.network import GaussianActor, DoubleCritic
import copy


class SACAgent:
    def __init__(self, obs_space, act_space, config, device):
        self.config = config
        self.device = device
        self.obs_dim, self.act_dim = obs_space.shape[0], act_space.shape[0]
        self.act_lim = float(act_space.high[0])
        self.hidden_dims = self.config.hidden_dims

        self.actor = GaussianActor(self.obs_dim, self.act_dim, self.hidden_dims).to(
            device
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.optimizer.actor_lr
        )

        self.critic = DoubleCritic(self.obs_dim, self.act_dim, self.hidden_dims).to(
            device
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.optimizer.critic_lr
        )

        self.target_entropy = -torch.prod(
            torch.Tensor(act_space.shape).to(device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=self.config.optimizer.critic_lr
        )

    def act(self, obs, deterministic=True):
        obs = torch.from_numpy(obs).to(self.device)
        dist = self.actor(obs)
        if not deterministic:
            action = self.act_lim * torch.tanh(dist.sample())
        else:
            action = dist.loc
        action = action.detach().cpu().numpy()
        return action

    def _calculate_action_logprob(self, obs):
        dist = self.target_actor(obs)
        z = dist.rsample()
        act = self.act_lim * torch.tanh(z)
        logprob = dist.log_prob(z) - torch.log(self.act_lim * (1 - act.pow(2)) + 1e-6)
        logprob = logprob.sum(1, keepdim=True)
        return act, logprob

    def _update_critic(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            next_act, next_logprob = self._calculate_action_logprob(next_obs)
            next_q1, next_q2 = self.target_critic(next_obs, next_act)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_logprob
            q_target = rew + ~done * self.config.gamma * next_q
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def _update_alpha(self, obs):
        _, logprob = self._calculate_action_logprob(obs)
        alpha_loss = (-self.log_alpha.exp() * (logprob + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return alpha_loss.item()

    def _update_actor(self, obs):
        act, logprob = self._calculate_action_logprob(obs)
        q1, q2 = self.critic(obs, act)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * logprob - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def _update_targets(self):
        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def update(self, batch):
        for key in batch:
            batch[key] = torch.from_numpy(batch[key]).to(self.device)

        obs, act, next_obs, rew, done = (
            batch["obs"],
            batch["act"],
            batch["next_obs"],
            batch["rew"],
            batch["done"],
        )

        critic_loss = self._update_critic(obs, act, next_obs, rew, done)
        actor_loss = self._update_actor(obs)
        alpha_loss = self._update_alpha(obs)
        self._update_targets()

        return actor_loss, critic_loss
