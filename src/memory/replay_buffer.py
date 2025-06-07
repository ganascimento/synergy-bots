import random
import torch
import utils.config as config
from collections import deque, namedtuple


Experience = namedtuple("Experience", field_names=["all_obs", "all_actions", "all_rewards", "all_next_obs", "all_dones"])


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, all_obs, all_actions, all_rewards, all_next_obs, all_dones):
        e = Experience(
            torch.tensor(all_obs),
            torch.tensor(all_actions),
            torch.tensor(all_rewards),
            torch.tensor(all_next_obs),
            torch.tensor(all_dones),
        )
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        batch_all_obs_list = [[] for _ in range(config.ROBOT_NUMBER)]
        batch_all_actions_list = [[] for _ in range(config.ROBOT_NUMBER)]
        batch_all_rewards_list = [[] for _ in range(config.ROBOT_NUMBER)]
        batch_all_next_obs_list = [[] for _ in range(config.ROBOT_NUMBER)]
        batch_all_dones_list = [[] for _ in range(config.ROBOT_NUMBER)]

        for exp in experiences:
            for i in range(config.ROBOT_NUMBER):
                batch_all_obs_list[i].append(exp.all_obs[i])
                batch_all_actions_list[i].append(exp.all_actions[i])
                batch_all_rewards_list[i].append(exp.all_rewards[i])
                batch_all_next_obs_list[i].append(exp.all_next_obs[i])
                batch_all_dones_list[i].append(exp.all_dones[i])

        final_obs = [torch.stack(batch_all_obs_list[i]).float() for i in range(config.ROBOT_NUMBER)]
        final_actions = [
            torch.tensor(batch_all_actions_list[i], dtype=torch.long).view(-1, 1) for i in range(config.ROBOT_NUMBER)
        ]
        final_rewards = [
            torch.tensor(batch_all_rewards_list[i], dtype=torch.float).view(-1, 1) for i in range(config.ROBOT_NUMBER)
        ]
        final_next_obs = [torch.stack(batch_all_next_obs_list[i]).float() for i in range(config.ROBOT_NUMBER)]
        final_dones = [torch.tensor(batch_all_dones_list[i], dtype=torch.uint8).view(-1, 1) for i in range(config.ROBOT_NUMBER)]

        return (final_obs, final_actions, final_rewards, final_next_obs, final_dones)

    def __len__(self):
        return len(self.memory)
