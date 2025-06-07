import torch
from typing import List


class OnPolicyRolloutStorage:
    def __init__(self, num_agents: int, rollout_length: int, dim_obs_robot: int, num_actions: int, device: torch.device):
        self.num_agents = num_agents
        self.rollout_length = rollout_length
        self.dim_obs_robot = dim_obs_robot

        self.device = device

        self.all_obs = [torch.zeros(rollout_length, dim_obs_robot, device=device) for _ in range(num_agents)]
        self.global_obs = torch.zeros(rollout_length, num_agents * dim_obs_robot, device=device)

        self.actions = [torch.zeros(rollout_length, 1, dtype=torch.long, device=device) for _ in range(num_agents)]
        self.action_log_probs = [torch.zeros(rollout_length, 1, device=device) for _ in range(num_agents)]
        self.rewards = [torch.zeros(rollout_length, 1, device=device) for _ in range(num_agents)]
        self.dones = [torch.zeros(rollout_length, 1, device=device) for _ in range(num_agents)]

        self.values = [torch.zeros(rollout_length + 1, 1, device=device) for _ in range(num_agents)]

        self.advantages = [torch.zeros(rollout_length, 1, device=device) for _ in range(num_agents)]
        self.returns = [torch.zeros(rollout_length, 1, device=device) for _ in range(num_agents)]

        self.step = 0

    def add(
        self,
        all_obs_list: List[torch.Tensor],
        global_obs_tensor: torch.Tensor,
        actions_list: List[torch.Tensor],
        action_log_probs_list: List[torch.Tensor],
        rewards_list: List[torch.Tensor],
        dones_list: List[torch.Tensor],
        values_list: List[torch.Tensor],
    ):
        self.global_obs[self.step].copy_(global_obs_tensor.squeeze(0))

        for i in range(self.num_agents):

            self.all_obs[i][self.step].copy_(all_obs_list[i].squeeze(0))
            self.actions[i][self.step].copy_(actions_list[i].squeeze(-1))
            self.action_log_probs[i][self.step].copy_(action_log_probs_list[i].squeeze(-1))
            self.rewards[i][self.step].copy_(rewards_list[i].squeeze(-1))
            self.dones[i][self.step].copy_(dones_list[i].squeeze(-1))
            self.values[i][self.step].copy_(values_list[i].squeeze(-1))

        self.step += 1

    def store_final_values(self, final_values_list: List[torch.Tensor]):
        if self.step != self.rollout_length:
            pass

        for i in range(self.num_agents):

            self.values[i][self.step].copy_(final_values_list[i].squeeze(-1))

    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float):

        for i in range(self.num_agents):
            gae = 0.0

            V_next = self.values[i][self.step]

            for t in reversed(range(self.step)):
                V_current = self.values[i][t]

                delta = self.rewards[i][t] + gamma * V_next * (1.0 - self.dones[i][t]) - V_current
                gae = delta + gamma * gae_lambda * (1.0 - self.dones[i][t]) * gae
                self.returns[i][t] = gae + V_current
                self.advantages[i][t] = gae
                V_next = V_current

    def get_generator(self, minibatch_size: int):
        num_samples = self.step
        if num_samples == 0:
            return

        indices = torch.randperm(num_samples, device=self.device).tolist()

        for start_idx in range(0, num_samples, minibatch_size):
            end_idx = min(start_idx + minibatch_size, num_samples)
            if start_idx >= end_idx:
                continue
            batch_indices = indices[start_idx:end_idx]

            batch_all_obs = [self.all_obs[i][batch_indices] for i in range(self.num_agents)]
            batch_global_obs = self.global_obs[batch_indices]
            batch_actions = [self.actions[i][batch_indices] for i in range(self.num_agents)]
            batch_old_log_probs = [self.action_log_probs[i][batch_indices] for i in range(self.num_agents)]
            batch_returns = [self.returns[i][batch_indices] for i in range(self.num_agents)]

            batch_advantages = [self.advantages[i][batch_indices] for i in range(self.num_agents)]

            yield batch_all_obs, batch_global_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages

    def after_update(self):
        self.step = 0
