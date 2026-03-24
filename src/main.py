import os
from core import OrchestratorMAPPO
from utils import config

if __name__ == "__main__":
    os.makedirs(config.SAVE_FOLDER, exist_ok=True)
    OrchestratorMAPPO().train_agent()
