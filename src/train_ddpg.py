from environment import GymEnvironment
from data import ReplayBuffer
from agent import DDPGAgent
from algorithm import OffPolicyRLTrainer, Evaluator

import torch
import hydra
from omegaconf import DictConfig


def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    return device


@hydra.main(version_base=None, config_path="config", config_name="train_ddpg")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## ENVIRONMENT ##
    env = GymEnvironment(config.environment)
    obs_space, act_space = env.obs_space, env.act_space
    print("Environment Built.")

    ## DATA ##
    data = ReplayBuffer(obs_space, act_space, config.buffer)
    print("Empty Buffer Initialized.")

    ## AGENT ##
    agent = DDPGAgent(obs_space, act_space, config.agent, device)
    print("Agent Created.")

    ## ALGORITHM ##
    print("Algorithm Running.")
    trainer = OffPolicyRLTrainer(env, data, agent, config.trainer)
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()
