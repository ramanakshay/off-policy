import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_space, act_space, config):
        self.config = config
        self.capacity = self.config.capacity
        self.batch_size = self.config.batch_size
        self.data = dict(
            obs=np.empty(
                (self.capacity, *obs_space.shape),
                dtype=obs_space.dtype,
            ),
            next_obs=np.empty(
                (self.capacity, *obs_space.shape),
                dtype=obs_space.dtype,
            ),
            act=np.empty((self.capacity, *act_space.shape), dtype=act_space.dtype),
            rew=np.empty((self.capacity, 1), dtype=np.float32),
            done=np.empty((self.capacity, 1), dtype=bool),
        )
        self.index = 0
        self.size = 0

    def insert(self, data):
        for key in data:
            self.data[key][self.index] = data[key]
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        assert self.batch_size <= self.size
        inds = np.random.randint(self.size, size=self.batch_size)
        batch = dict()
        for key in self.data:
            batch[key] = self.data[key][inds]
        return batch

    def reset(self):
        self.index = 0
        self.size = 0
