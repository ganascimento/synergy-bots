import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils.config as config
from torch.distributions import Categorical
from memory import OnPolicyRolloutStorage


class Actor(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_logits = nn.Linear(128, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3_logits(x)
        return logits

    def get_action_dist(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, input_size_all):
        super().__init__()
        self.fc1 = nn.Linear(input_size_all, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_value = nn.Linear(128, config.ROBOT_NUMBER)

    def forward(self, all_obs_concatenated):
        x = F.relu(self.fc1(all_obs_concatenated))
        x = F.relu(self.fc2(x))
        values = self.fc3_value(x)
        return values


class MAPPOAgent:
    def __init__(self, dim_obs_robot, num_actions):
        self.n_agents = config.ROBOT_NUMBER
        self.dim_obs_robot = dim_obs_robot
        self.num_actions = num_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        dim_obs_all = self.n_agents * dim_obs_robot

        self.actors = [Actor(dim_obs_robot, num_actions).to(self.device) for _ in range(self.n_agents)]
        self.critic = Critic(dim_obs_all).to(self.device)

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=config.LR_ACTOR) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        self.storage = OnPolicyRolloutStorage(self.n_agents, config.ROLLOUT_LENGTH, dim_obs_robot, num_actions, self.device)

        self._load_model()

    def select_actions(self, all_obs_list_numpy, evaluate=False):
        actions_list_tensor = []
        log_probs_list_tensor = []

        all_obs_tensors = []
        for obs_np in all_obs_list_numpy:
            obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
            all_obs_tensors.append(obs_tensor)

        global_obs_tensor = torch.cat(all_obs_tensors, dim=1)
        with torch.no_grad():
            all_agent_values_tensor = self.critic(global_obs_tensor)

        values_for_storage = []

        for i in range(self.n_agents):
            obs_tensor = all_obs_tensors[i]

            self.actors[i].eval()
            with torch.no_grad():
                action_dist = self.actors[i].get_action_dist(obs_tensor)
                if evaluate:
                    action = action_dist.probs.argmax(dim=-1, keepdim=True)
                else:
                    action = action_dist.sample().unsqueeze(-1)
            self.actors[i].train()

            log_prob = action_dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

            actions_list_tensor.append(action)
            log_probs_list_tensor.append(log_prob)
            values_for_storage.append(all_agent_values_tensor[:, i].unsqueeze(-1))

        actions_list_numpy = [a.item() for a in actions_list_tensor]

        return actions_list_numpy, actions_list_tensor, log_probs_list_tensor, values_for_storage, global_obs_tensor

    def store_transition(
        self,
        all_obs_list_tensors,
        global_obs_tensor,
        actions_list_tensors,
        log_probs_list_tensors,
        rewards_list_tensors,
        dones_list_tensors,
        values_list_tensors,
    ):
        processed_rewards = []
        processed_dones = []
        for r_tensor, d_tensor in zip(rewards_list_tensors, dones_list_tensors):
            processed_rewards.append(r_tensor.float().reshape(1, 1).to(self.device))
            processed_dones.append(d_tensor.float().reshape(1, 1).to(self.device))

        self.storage.add(
            all_obs_list_tensors,
            global_obs_tensor,
            actions_list_tensors,
            log_probs_list_tensors,
            processed_rewards,
            processed_dones,
            values_list_tensors,
        )

    def compute_advantages_and_prepare_update(self, next_all_obs_list_numpy):
        next_all_obs_tensors = []
        for obs_np in next_all_obs_list_numpy:
            obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
            next_all_obs_tensors.append(obs_tensor)

        next_global_obs_tensor = torch.cat(next_all_obs_tensors, dim=1)

        with torch.no_grad():
            next_values_all_agents = self.critic(next_global_obs_tensor)

        final_values_list = [next_values_all_agents[:, i].unsqueeze(-1) for i in range(self.n_agents)]
        self.storage.store_final_values(final_values_list)

        self.storage.compute_advantages_and_returns(config.GAMMA, config.GAE_LAMBDA)

    def learn(self):
        total_actor_loss_accum = 0.0
        total_critic_loss_accum = 0.0
        total_entropy_accum = 0.0

        for epoch in range(config.PPO_EPOCHS):
            data_generator = self.storage.get_generator(config.PPO_MINIBATCH_SIZE)

            for (
                batch_all_obs,
                batch_global_obs,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in data_generator:
                # --- Critic ---
                predicted_values_all_agents = self.critic(batch_global_obs)
                critic_loss_terms = []
                for i in range(self.n_agents):
                    predicted_values_agent_i = predicted_values_all_agents[:, i].unsqueeze(-1)
                    current_agent_batch_returns = batch_returns[i]

                    returns_mean = current_agent_batch_returns.mean()
                    returns_std = current_agent_batch_returns.std()

                    normalized_target_returns = (current_agent_batch_returns - returns_mean) / (returns_std + 1e-8)

                    loss_v = F.mse_loss(predicted_values_agent_i, normalized_target_returns)
                    critic_loss_terms.append(loss_v)

                critic_loss = torch.stack(critic_loss_terms).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.critic_optimizer.step()
                total_critic_loss_accum += critic_loss.item()

                # --- Actor ---
                current_actor_loss_sum_for_batch = 0.0
                current_entropy_sum_for_batch = 0.0

                for i in range(self.n_agents):
                    action_dist = self.actors[i].get_action_dist(batch_all_obs[i])
                    new_log_probs = action_dist.log_prob(batch_actions[i].squeeze(-1)).unsqueeze(-1)
                    entropy = action_dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs[i])
                    advantages_agent_i = batch_advantages[i]

                    advantages_agent_i = (advantages_agent_i - advantages_agent_i.mean()) / (advantages_agent_i.std() + 1e-8)

                    surr1 = ratio * advantages_agent_i
                    surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_agent_i
                    actor_loss = -torch.min(surr1, surr2).mean()

                    actor_agent_loss = actor_loss - config.ENTROPY_COEFF * entropy

                    self.actor_optimizers[i].zero_grad()
                    actor_agent_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), config.MAX_GRAD_NORM)
                    self.actor_optimizers[i].step()

                    current_actor_loss_sum_for_batch += actor_loss.item()
                    current_entropy_sum_for_batch += entropy.item()

                total_actor_loss_accum += current_actor_loss_sum_for_batch / self.n_agents
                total_entropy_accum += current_entropy_sum_for_batch / self.n_agents

        num_updates = config.PPO_EPOCHS * (self.storage.rollout_length // config.PPO_MINIBATCH_SIZE)
        if num_updates == 0:
            num_updates = 1

        avg_actor_loss = total_actor_loss_accum / num_updates
        avg_critic_loss = total_critic_loss_accum / num_updates
        avg_entropy = total_entropy_accum / num_updates

        self.storage.after_update()

        return avg_actor_loss, avg_critic_loss, avg_entropy

    def save_model(self):
        actor_state_dicts = [actor.state_dict() for actor in self.actors]
        actor_optimizers_dicts = [actor_optimizer.state_dict() for actor_optimizer in self.actor_optimizers]
        self.storage.after_update()

        checkpoint = {
            "actors_state_dict": actor_state_dicts,
            "critic_state_dict": self.critic.state_dict(),
            "actors_optimizer_state_dict": actor_optimizers_dicts,
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "storage": self.storage,
        }

        torch.save(checkpoint, config.SAVE_PATH)
        print(f"Model saved in: {config.SAVE_PATH}")

    def _load_model(self):
        if not os.path.exists(config.SAVE_PATH):
            return

        checkpoint = torch.load(config.SAVE_PATH, map_location=self.device)

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint["actors_state_dict"][i])
            self.actor_optimizers[i].load_state_dict(checkpoint["actors_optimizer_state_dict"][i])

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.storage = checkpoint["storage"]

        print(f"Model loaded with: {config.SAVE_PATH}")
        return
