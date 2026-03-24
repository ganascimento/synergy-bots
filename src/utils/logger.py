import csv
import json
import os
from datetime import datetime

import utils.config as config


class TrainingLogger:
    """Logs training metrics to CSV and saves a live state snapshot for the dashboard."""

    HEADERS = ["timestamp", "episode", "steps", "total_reward", "uncleaned_cells",
               "actor_loss", "critic_loss", "entropy", "entropy_coeff", "completed"]

    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(config.METRICS_CSV):
            with open(config.METRICS_CSV, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADERS)

    def log_episode(
        self,
        episode: int,
        steps: int,
        total_reward: float,
        uncleaned_cells: int,
        actor_loss: float,
        critic_loss: float,
        entropy: float,
        entropy_coeff: float,
    ):
        completed = int(uncleaned_cells == 0)
        timestamp = datetime.now().isoformat(timespec="seconds")

        with open(config.METRICS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                timestamp, episode, steps, round(total_reward, 4),
                uncleaned_cells, round(actor_loss, 6), round(critic_loss, 6),
                round(entropy, 4), round(entropy_coeff, 6), completed,
            ])

        star = " ★" if completed else ""
        print(
            f"Ep {episode:5d} | Steps: {steps:3d}/{config.MAX_STEPS} | "
            f"Reward: {total_reward:8.2f} | Uncleaned: {uncleaned_cells:2d} | "
            f"ActorL: {actor_loss:.4f} | CriticL: {critic_loss:.4f} | "
            f"Entropy: {entropy:.3f} (coeff={entropy_coeff:.4f}){star}"
        )

    def save_state_snapshot(self, episode: int, steps: int, uncleaned_cells: int,
                            grid_state: list, robot_positions: list, cleaned_cells: list):
        """Saves current game state so the Streamlit dashboard can render it live."""
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "episode": episode,
            "steps": steps,
            "uncleaned_cells": uncleaned_cells,
            "grid_state": grid_state,
            "robot_positions": robot_positions,
            "cleaned_cells_per_robot": cleaned_cells,
        }
        tmp = config.STATE_SNAPSHOT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp, config.STATE_SNAPSHOT_FILE)
