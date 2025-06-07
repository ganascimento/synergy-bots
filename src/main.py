import os
from core import OrchestratorMAAC, OrchestratorMAPPO
from utils import config

if __name__ == "__main__":
    os.makedirs(config.SAVE_FOLDER, exist_ok=True)

    if config.MODEL == 1:
        OrchestratorMAAC().train_agent()
    else:
        OrchestratorMAPPO().train_agent()
