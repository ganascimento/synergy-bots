import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import utils.config as config
from memory import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class Critic(nn.Module):
    def __init__(self, input_size_all, num_actions_all):
        super().__init__()
        self.n_agents_total = config.ROBOT_NUMBER

        self.obs_fc1 = nn.Linear(input_size_all, 128)
        self.act_fc1 = nn.Linear(num_actions_all, 128)

        self.fc2 = nn.Linear(128 + 128, 128)
        self.fc3 = nn.Linear(128, self.n_agents_total)

    def forward(self, all_obs, all_actions_one_hot):
        """
        all_obs: Tensor concatenado das observações de todos os agentes. (batch_size, N_AGENTS * DIM_OBS_ROBOT)
        all_actions_one_hot: Tensor concatenado das ações one-hot de todos os agentes. (batch_size, N_AGENTS * NUM_ACTIONS)
        """
        obs_out = F.relu(self.obs_fc1(all_obs))
        act_out = F.relu(self.act_fc1(all_actions_one_hot))

        x = torch.cat((obs_out, act_out), dim=1)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values


class MAACAgent:
    def __init__(self, dim_obs_robot, num_actions):
        self.n_agents = config.ROBOT_NUMBER
        self.dim_obs_robot = dim_obs_robot
        self.num_actions = num_actions
        self.epsilon = config.EPSILON_START
        self.t_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        dim_obs_all = self.n_agents * dim_obs_robot
        num_actions_all_one_hot = self.n_agents * num_actions

        self.actors = [Actor(dim_obs_robot, num_actions).to(self.device) for _ in range(self.n_agents)]
        self.actors_target = [Actor(dim_obs_robot, num_actions).to(self.device) for _ in range(self.n_agents)]

        self.critic = Critic(dim_obs_all, num_actions_all_one_hot).to(self.device)
        self.critic_target = Critic(dim_obs_all, num_actions_all_one_hot).to(self.device)

        for i in range(self.n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=config.LR_ACTOR) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)

    def select_actions(self, all_obs, add_noise=True):
        actions = []

        for i in range(self.n_agents):
            obs_tensor = torch.from_numpy(all_obs[i]).float().unsqueeze(0).to(self.device)
            self.actors[i].eval()
            with torch.no_grad():
                logits = self.actors[i](obs_tensor)
                action_probs = F.softmax(logits, dim=-1)
            self.actors[i].train()

            if add_noise and random.random() < self.epsilon:
                action = random.choice(np.arange(self.num_actions))
            else:
                action = torch.argmax(action_probs).item()
            actions.append(action)
        return actions

    def step(self, all_obs, all_actions, all_rewards, all_next_obs, all_dones):
        total_actor_loss_accum = 0.0
        total_critic_loss_accum = 0.0
        self.replay_buffer.add(all_obs, all_actions, all_rewards, all_next_obs, all_dones)

        self.t_step = (self.t_step + 1) % config.UPDATES_PER_EPISODE_COLLECTION
        if self.t_step == 0:
            if len(self.replay_buffer) >= config.BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                actor_loss, critic_loss = self.learn(experiences)
                total_actor_loss_accum += actor_loss
                total_critic_loss_accum += critic_loss

        return total_actor_loss_accum, total_critic_loss_accum

    def learn(self, experiences):
        all_obs_batch, all_actions_batch, all_rewards_batch, all_next_obs_batch, all_dones_batch = experiences

        all_obs_batch = [item.to(self.device) for item in all_obs_batch]
        all_actions_batch = [item.to(self.device) for item in all_actions_batch]
        all_rewards_batch = [item.to(self.device) for item in all_rewards_batch]
        all_next_obs_batch = [item.to(self.device) for item in all_next_obs_batch]
        all_dones_batch = [item.to(self.device) for item in all_dones_batch]

        obs_concat_batch = torch.cat(all_obs_batch, dim=1)
        next_obs_concat_batch = torch.cat(all_next_obs_batch, dim=1)

        actions_one_hot = []
        for i in range(self.n_agents):
            actions_one_hot.append(F.one_hot(all_actions_batch[i].squeeze(1), num_classes=self.num_actions))
        actions_one_hot_concat_batch = torch.cat(actions_one_hot, dim=1).float()

        # --- Critic ---
        with torch.no_grad():
            next_actions_target_one_hot = []
            for j in range(self.n_agents):
                actor_target_output = self.actors_target[j](all_next_obs_batch[j])
                next_action_target_idx = torch.argmax(actor_target_output, dim=1, keepdim=True)
                next_actions_target_one_hot.append(F.one_hot(next_action_target_idx.squeeze(1), num_classes=self.num_actions))
            next_actions_one_hot_target_concat = torch.cat(next_actions_target_one_hot, dim=1).float()

            q_target_next_all_agents = self.critic_target(next_obs_concat_batch, next_actions_one_hot_target_concat)

        q_targets_stacked = []
        for i in range(self.n_agents):
            q_target_agent_i = all_rewards_batch[i] + (
                config.GAMMA * q_target_next_all_agents[:, i].unsqueeze(1) * (1 - all_dones_batch[i])
            )
            q_targets_stacked.append(q_target_agent_i)
        q_targets_final = torch.cat(q_targets_stacked, dim=1)

        q_expected_all_agents = self.critic(obs_concat_batch, actions_one_hot_concat_batch)

        critic_loss = F.mse_loss(q_expected_all_agents, q_targets_final)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.CLIP_GRAD_NORM)
        self.critic_optimizer.step()

        # --- Actor ---
        actor_loss_sum_for_return = 0.0
        for i in range(self.n_agents):
            current_policy_actions_one_hot_for_this_actor_loss = []
            for k in range(self.n_agents):
                current_agent_obs_k = all_obs_batch[k]
                logits_k = self.actors[k](current_agent_obs_k)
                action_k_gumbel_one_hot = F.gumbel_softmax(logits_k, tau=config.GUMBEL_TEMPERATURE, hard=True, dim=-1)
                if k == i:
                    current_policy_actions_one_hot_for_this_actor_loss.append(action_k_gumbel_one_hot)
                else:
                    current_policy_actions_one_hot_for_this_actor_loss.append(action_k_gumbel_one_hot.detach())

            policy_actions_for_this_actor_concat = torch.cat(current_policy_actions_one_hot_for_this_actor_loss, dim=1)
            q_values_for_this_actor_loss = self.critic(obs_concat_batch, policy_actions_for_this_actor_concat)

            actor_loss = -q_values_for_this_actor_loss[:, i].mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), config.CLIP_GRAD_NORM)
            self.actor_optimizers[i].step()

            actor_loss_sum_for_return += actor_loss.item()

        self.update_target_networks()

        return actor_loss_sum_for_return / self.n_agents, critic_loss.item()

    def update_target_networks(self):
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.actors_target[i], config.TAU)
        self._soft_update(self.critic, self.critic_target, config.TAU)

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
