import torch
import itertools
import numpy as np
import torch.nn.functional as F
from torch import optim
from agent.td3.network import Actor, DoubleCritic
import copy


class TD3Agent:
    def __init__(self, obs_space, act_space, config, device):
        self.config = config
        self.device = device
        self.obs_dim, self.act_dim = obs_space.shape[0], act_space.shape[0]
        self.act_lim = float(act_space.high[0])
        self.hidden_dims = self.config.hidden_dims

        self.actor = Actor(self.obs_dim, self.act_dim, self.hidden_dims).to(device)
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

        self.update_policy = itertools.cycle([False] * (self.config.delay - 1) + [True])

    def act(self, obs, deterministic=True):
        obs = torch.from_numpy(obs).to(self.device)
        action = self.actor(obs).detach().cpu().numpy()
        if not deterministic:
            action += np.random.normal(
                0.0, self.act_lim * self.config.act_noise, size=self.act_dim
            )
            action = action.clip(-self.act_lim, +self.act_lim)
        return action

    def _train_critic(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            next_act = self.target_actor(next_obs)
            # Add target policy smoothing:
            noise = (torch.randn_like(next_act) * self.config.target_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_act = (next_act + noise).clamp(
                -self.act_lim, self.act_lim
            )  # Also clip the noisy action
            next_q1, next_q2 = self.target_critic(next_obs, next_act)
            next_q = torch.min(next_q1, next_q2)
            q_target = rew + ~done * self.config.gamma * next_q
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        return critic_loss.item()

    def _train_actor(self, obs):
        actor_loss = -self.critic.critic1(obs, self.actor(obs)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        return actor_loss.item()

    def _train_targets(self):
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

    def train(self, batch):
        for key in batch:
            batch[key] = torch.from_numpy(batch[key]).to(self.device)

        obs, act, next_obs, rew, done = (
            batch["obs"],
            batch["act"],
            batch["next_obs"],
            batch["rew"],
            batch["done"],
        )

        loss = {}
        critic_loss = self._train_critic(obs, act, next_obs, rew, done)
        loss["critic"] = critic_loss
        # Policy and target updates
        if next(self.update_policy):
            actor_loss = self._train_actor(obs)
            self._train_targets()

        loss = {"actor": actor_loss, "critic": critic_loss}
        return loss
