import numpy as np
from game import Game
from models import MAACAgent
import utils.config as config
from typing import List


class OrchestratorMAAC:
    def __init__(self):
        self.game = Game()
        self.num_actions = 5
        self.episode_count = 0

    def train_agent(self):
        agent = MAACAgent(
            dim_obs_robot=config.NN_INPUT_SIZE,
            num_actions=self.num_actions,
        )

        while self.episode_count < config.MAX_EPISODES:
            self.episode_count += 1

            self.game.reset()
            episode_accumulated_reward = 0.0
            current_local_states = self._get_local_states()
            step_count = 0
            avg_actor_loss_over_updates = 0.0
            avg_critic_loss_over_updates = 0.0
            total_loss_count = 0

            for _ in range(config.MAX_STEPS):
                if config.SHOW_GAME_RENDER:
                    self.game.play_render()

                joint_actions: List[int] = agent.select_actions(current_local_states)
                next_local_states, individual_rewards, episode_done, step_count = self.game.step(joint_actions)

                actor_loss, critic_loss = agent.step(
                    current_local_states, joint_actions, individual_rewards, next_local_states, episode_done
                )

                if actor_loss != 0 and critic_loss != 0:
                    avg_actor_loss_over_updates += actor_loss
                    avg_critic_loss_over_updates += critic_loss
                    total_loss_count += 1

                current_local_states = next_local_states
                episode_accumulated_reward += np.mean(individual_rewards)

                if episode_done[0]:
                    break

            agent.decay_epsilon()

            actor_loss_norm = avg_actor_loss_over_updates / total_loss_count if total_loss_count > 0 else 0.00
            critic_loss_norm = avg_critic_loss_over_updates / total_loss_count if total_loss_count > 0 else 0.00

            self._log_episode_progress(
                episode_accumulated_reward,
                step_count,
                actor_loss_norm,
                critic_loss_norm,
                agent.epsilon,
            )

        self.game.close()

    def _get_local_states(self) -> np.ndarray:
        local_states_list: List[np.ndarray] = []

        for robot_instance in self.game.all_robots_list:
            local_state = robot_instance.get_state(self.game.all_robots_list)
            local_states_list.append(local_state)

        return np.array(local_states_list)

    def _log_episode_progress(
        self,
        reward: float,
        steps: int,
        actor_loss: float,
        critic_loss: float,
        epsilon: float,
    ):
        start = "🌟" if steps < config.MAX_STEPS else ""
        print(
            f"Ep: {self.episode_count}\t"
            f"Reward: {reward:.2f}\tUnclean: {self.game.count_uncleaned_cells()}\tSteps: {steps}{start}\t"
            f"ActorLoss: {actor_loss:.4f}\tCriticLoss: {critic_loss:.4f}\t"
            f"Epsilon: {epsilon:.3f}"
        )
