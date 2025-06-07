import numpy as np
import torch
from game import Game
from models.mappo_nn import MAPPOAgent
import utils.config as config
from typing import List


class OrchestratorMAPPO:
    def __init__(self):
        self.game = Game()
        self.agent = MAPPOAgent(
            dim_obs_robot=config.NN_INPUT_SIZE,
            num_actions=5,
        )

        self.orchestrator_episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

        self.last_actor_loss_log = 0.0
        self.last_critic_loss_log = 0.0
        self.last_entropy_log = 0.0

    def _reset_episode_trackers_and_game(self) -> List[np.ndarray]:
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.game.reset()
        return self._get_local_states()

    def train_agent(self):
        current_local_states_np: List[np.ndarray] = self._reset_episode_trackers_and_game()

        while self.orchestrator_episode_count < config.MAX_EPISODES:
            for _ in range(config.ROLLOUT_LENGTH):
                if config.SHOW_GAME_RENDER:
                    self.game.play_render()

                actions_np_list, actions_torch_list, log_probs_torch_list, values_torch_list, global_obs_torch = (
                    self.agent.select_actions(current_local_states_np, evaluate=False)
                )

                next_local_states_np_from_env, individual_rewards, individual_dones, _ = self.game.step(actions_np_list)

                self.current_episode_steps += 1
                self.current_episode_reward += np.sum(individual_rewards)

                current_obs_tensors_for_storage = [
                    torch.from_numpy(obs).float().unsqueeze(0).to(self.agent.device) for obs in current_local_states_np
                ]
                rewards_tensors_list = [
                    torch.tensor([r], dtype=torch.float32, device=self.agent.device) for r in individual_rewards
                ]
                dones_tensors_list = [
                    torch.tensor([float(d)], dtype=torch.float32, device=self.agent.device) for d in individual_dones
                ]

                self.agent.store_transition(
                    current_obs_tensors_for_storage,
                    global_obs_torch,
                    actions_torch_list,
                    log_probs_torch_list,
                    rewards_tensors_list,
                    dones_tensors_list,
                    values_torch_list,
                )

                current_local_states_np = next_local_states_np_from_env

                episode_terminated_by_game = any(individual_dones)
                episode_terminated_by_max_steps = self.current_episode_steps >= config.MAX_STEPS

                if episode_terminated_by_game or episode_terminated_by_max_steps:
                    self.orchestrator_episode_count += 1

                    avg_reward_for_episode = (
                        (self.current_episode_reward / self.agent.n_agents) / self.current_episode_steps
                        if self.current_episode_steps > 0
                        else 0.0
                    )

                    self._log_episode_progress(
                        avg_reward_for_episode,
                        self.current_episode_steps,
                        self.last_actor_loss_log,
                        self.last_critic_loss_log,
                        self.last_entropy_log,
                    )

                    if self.orchestrator_episode_count % config.SAVE_MODEL_EVERY == 0:
                        self.agent.save_model()

                    if self.orchestrator_episode_count >= config.MAX_EPISODES:
                        self.game.close()
                        return

                    current_local_states_np = self._reset_episode_trackers_and_game()

            self.agent.compute_advantages_and_prepare_update(current_local_states_np)

            actor_loss, critic_loss, entropy = self.agent.learn()

            self.last_actor_loss_log = actor_loss
            self.last_critic_loss_log = critic_loss
            self.last_entropy_log = entropy

        self.game.close()

    def _get_local_states(self) -> List[np.ndarray]:
        local_states_list: List[np.ndarray] = []
        for robot_instance in self.game.all_robots_list:
            local_state = robot_instance.get_state(self.game.all_robots_list)
            if not isinstance(local_state, np.ndarray):
                local_state = np.array(local_state, dtype=np.float32)
            local_states_list.append(local_state)
        return local_states_list

    def _log_episode_progress(
        self,
        avg_reward: float,
        steps: int,
        actor_loss: float,
        critic_loss: float,
        entropy: float,
    ):
        star = "🌟" if steps < config.MAX_STEPS else ""

        print(
            f"Ep: {self.orchestrator_episode_count}\t"
            f"Reward: {avg_reward:.3f}\tUnclean: {self.game.count_uncleaned_cells()}\tSteps: {steps}{star}\t"
            f"ActorLoss: {actor_loss:.4f}\tCriticLoss: {critic_loss:.4f}\t"
            f"Entropy: {entropy:.3f}"
        )
