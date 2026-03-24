import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import pygame

pygame.init()

import utils.config as config
from core import OrchestratorMAPPO

if __name__ == "__main__":
    os.makedirs(config.SAVE_FOLDER, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"

    print("=" * 60)
    print("  Synergy Bots — Headless Training (MAPPO)")
    print("=" * 60)
    print(f"  Device   : {device}")
    print(f"  Episodes : {config.MAX_EPISODES}")
    print(f"  Max steps: {config.MAX_STEPS} per episode")
    print(
        f"  Grid     : {config.GRID_WIDTH}x{config.GRID_HEIGHT} ({config.GRID_WIDTH * config.GRID_HEIGHT} cells)"
    )
    print(f"  Robots   : {config.ROBOT_NUMBER}")
    print(f"  Logs     : {config.METRICS_CSV}")
    print(f"  Model    : {config.SAVE_PATH}")
    print("=" * 60)

    OrchestratorMAPPO().train_agent()

    print("\nTraining complete.")
