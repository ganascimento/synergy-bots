import os
from typing import List

import numpy as np
import torch

import utils.config as config
from game import Game
from models.mappo_nn import MAPPOAgent
from utils.logger import TrainingLogger


class OrchestratorMAPPO:
    def __init__(self):
        self.game = Game()
        self.agent = MAPPOAgent(
            dim_obs_robot=config.NN_INPUT_SIZE,
            num_actions=5,
        )
        self.logger = TrainingLogger()

        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_entropy = 0.0

    def train_agent(self):
        os.makedirs(config.SAVE_FOLDER, exist_ok=True)
        current_states = self._reset_episode()

        while self.episode_count < config.MAX_EPISODES:
            for _ in range(config.ROLLOUT_LENGTH):
                if config.SHOW_GAME_RENDER:
                    self.game.play_render()

                actions_np, actions_t, log_probs_t, values_t, global_obs_t = (
                    self.agent.select_actions(current_states, evaluate=False)
                )

                next_states, rewards, dones, _ = self.game.step(actions_np)

                self.current_episode_steps += 1
                self.current_episode_reward += float(np.sum(rewards))

                obs_tensors = [
                    torch.from_numpy(obs).float().unsqueeze(0).to(self.agent.device)
                    for obs in current_states
                ]
                rewards_t = [
                    torch.tensor([r], dtype=torch.float32, device=self.agent.device)
                    for r in rewards
                ]
                dones_t = [
                    torch.tensor(
                        [float(d)], dtype=torch.float32, device=self.agent.device
                    )
                    for d in dones
                ]

                self.agent.store_transition(
                    obs_tensors,
                    global_obs_t,
                    actions_t,
                    log_probs_t,
                    rewards_t,
                    dones_t,
                    values_t,
                )

                current_states = next_states

                if any(dones):
                    self._finish_episode()
                    if self.episode_count >= config.MAX_EPISODES:
                        self.game.close()
                        return
                    current_states = self._reset_episode()

            # Update after collecting ROLLOUT_LENGTH steps
            self.agent.compute_advantages_and_prepare_update(current_states)
            self.last_actor_loss, self.last_critic_loss, self.last_entropy = (
                self.agent.learn()
            )

        self.game.close()

    def _reset_episode(self) -> List[np.ndarray]:
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.game.reset()
        return self._get_states()

    def _finish_episode(self):
        self.episode_count += 1

        uncleaned = self.game.count_uncleaned_cells()
        avg_reward = self.current_episode_reward / max(1, self.current_episode_steps)

        self.logger.log_episode(
            episode=self.episode_count,
            steps=self.current_episode_steps,
            total_reward=avg_reward,
            uncleaned_cells=uncleaned,
            actor_loss=self.last_actor_loss,
            critic_loss=self.last_critic_loss,
            entropy=self.last_entropy,
            entropy_coeff=self.agent.entropy_coeff,
        )

        self.logger.save_state_snapshot(
            episode=self.episode_count,
            steps=self.current_episode_steps,
            uncleaned_cells=uncleaned,
            grid_state=self.game.get_grid_state().tolist(),
            robot_positions=[
                [r.rect.x // config.CELL_SIZE, r.rect.y // config.CELL_SIZE]
                for r in self.game.all_robots_list
            ],
            cleaned_cells=[
                [
                    [px // config.CELL_SIZE, py // config.CELL_SIZE]
                    for px, py in r.clear_cells
                ]
                for r in self.game.all_robots_list
            ],
        )

        self.agent.decay_entropy()

        if self.episode_count % config.SAVE_MODEL_EVERY == 0:
            self.agent.save_model()

    def _get_states(self) -> List[np.ndarray]:
        states = []
        for robot in self.game.all_robots_list:
            state = robot.get_state(self.game.all_robots_list)
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            states.append(state)
        return states
