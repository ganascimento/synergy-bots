# --- Grid Config ---

WIDTH = 320
HEIGHT = 240
CELL_SIZE = 40
MARK_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# --- Game Config ---

SHOW_GAME_RENDER = False

# --- Environment Config ---

ROBOT_NUMBER = 2
ROBOT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
OBSTACLE_COLOR = (200, 200, 200)
OBSTACLE_PROBABILITY = 0.05
FPS = 30

# --- Save Config ---

SAVE_FOLDER = "./training_data"
SAVE_PATH = f"{SAVE_FOLDER}/model_mappo.pth"
SAVE_MODEL_EVERY = 100

# --- Log Config ---

LOG_DIR = f"{SAVE_FOLDER}/logs"
METRICS_CSV = f"{LOG_DIR}/training_metrics.csv"
STATE_SNAPSHOT_FILE = f"{LOG_DIR}/current_state.json"

# --- Global Train Config ---

MAX_EPISODES = 5000
# Maximum steps per episode. 8x6 grid = 48 cells; 2 robots need ~24 moves each
# plus navigation overhead. 500 gives room to learn without running forever.
MAX_STEPS = 500
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99

# --- Train Config MAPPO ---

NN_INPUT_SIZE = (WIDTH // CELL_SIZE) * (HEIGHT // CELL_SIZE) + ROBOT_NUMBER * 2

ROLLOUT_LENGTH = 256
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 64
PPO_CLIP_EPS = 0.2
GAE_LAMBDA = 0.95

# Entropy starts high to encourage exploration, decays toward minimum over training.
ENTROPY_COEFF = 0.05
ENTROPY_COEFF_MIN = 0.005
ENTROPY_COEFF_DECAY = 0.9995

VALUE_LOSS_COEFF = 0.5
MAX_GRAD_NORM = 0.5
