import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        self.network = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.logstd_layer = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs):
        x = self.network(obs)
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        std = logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def critic1(self, obs, act):
        x = torch.cat([obs, act], 1)
        x = self.q1(x)
        return x

    def critic2(self, obs, act):
        x = torch.cat([obs, act], 1)
        x = self.q2(x)
        return x

    def forward(self, obs, act):
        x = torch.cat([obs, act], 1)
        x1, x2 = self.q1(x), self.q2(x)
        return x1, x2
