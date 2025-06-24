import torch
from tqdm import tqdm
from algorithm.evaluator import Evaluator


class OffPolicyRLTrainer:
    def __init__(self, env, buffer, agent, config):
        self.config = config
        self.env = env.env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = Evaluator(env, agent, config.evaluator)

    def run(self):
        print(f"Total Steps = {self.config.total_steps}")
        obs, info = self.env.reset()
        for step in range(self.config.total_steps):
            if step < self.config.random_steps:
                act = self.env.action_space.sample()
            else:
                act = self.agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated
            self.buffer.insert(
                dict(
                    obs=obs,
                    next_obs=next_obs,
                    act=act,
                    rew=reward,
                    done=done,
                )
            )
            if done:
                obs, info = self.env.reset()
            else:
                obs = next_obs

            if (
                step >= self.config.train_start
                and step % self.config.train_interval == 0
            ):
                for iter in range(self.config.train_iters):
                    batch = self.buffer.sample()
                    self.agent.update(batch)

            if step % self.config.eval_interval == 0:
                print(f"Step : {step}, Evaluating agent")
                self.evaluator.run()
