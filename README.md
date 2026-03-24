# 🧹 SYNERGY-BOTS-GAME

This repository implements a game of vacuum cleaner robots where a multi-agent AI model learns to play and clean the entire environment, seeking the best collaboration between them to achieve greater efficiency.

## ✨ Features

- 🤖 **Multi-Agent Reinforcement Learning (MAPPO)**: Centralized training with decentralized execution — each robot decides with local observations while the critic sees the global state.
- 📈 **MAPPO**: Multi-Agent Proximal Policy Optimization with GAE, entropy decay and orthogonal weight initialization for stable convergence.
- 🧠 **On-Policy Rollout Storage**: Collects 256-step trajectories before each update, with Generalized Advantage Estimation (GAE).
- 🎮 **Streamlit Dashboard**: Real-time training metrics, live game-state monitor and step-by-step visual playback.
- 💾 **Checkpoint**: Model saved every 100 episodes and loaded automatically on restart.
- ⚙️ **Configurable**: Learning rate, entropy schedule, episode length and all hyperparameters in `utils/config.py`.

## Resources

- [Python](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [PyGame](https://www.pygame.org/news)
- [Numpy](https://numpy.org/)
- [Streamlit](https://streamlit.io/)

<br>

<img
    align="left"
    alt="Python"
    title="Python"
    width="30px"
    style="padding-right: 10px; padding-left: 20px"
    src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg"
/>
<img
    align="left"
    alt="PyTorch"
    title="PyTorch"
    width="30px"
    style="padding-right: 10px;"
    src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytorch/pytorch-original.svg"
/>
<img
    align="left"
    alt="Numpy"
    title="Numpy"
    width="30px"
    style="padding-right: 10px;"
    src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg"
/>

<br>

## 🤖 MARL (Multi-Agent Reinforcement Learning)

Is a subfield of artificial intelligence. In it, multiple agents learn simultaneously through trial and error in a shared environment. Each agent seeks to maximize its own reward, and its actions affect both the environment and the other agents.

## 📈 MAPPO (Multi-Agent Proximal Policy Optimization)

Is the algorithm powering this project. It extends PPO to multi-agent cooperative scenarios using a **Centralized Training, Decentralized Execution (CTDE)** scheme:

- **Training**: a shared centralized critic sees all robots' observations and outputs per-agent value estimates.
- **Execution**: each actor only uses its own local observation (dirt grid + self position + ally position).

Key stabilization techniques used here:

- PPO clipped surrogate loss (ε = 0.2)
- Generalized Advantage Estimation (GAE, λ = 0.95)
- Entropy regularization with schedule (0.05 → 0.005)
- Orthogonal weight initialization
- LayerNorm in all hidden layers
- Gradient clipping (max norm = 0.5)

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/ganascimento/synergy-bots-game.git
cd synergy-bots-game
```

### 🐍 Setting up a Virtual Environment

Create and activate a virtual environment to isolate the project dependencies:

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> To deactivate the virtual environment, simply run `deactivate`.

### 📦 Installing Dependencies

```bash
pip install torch numpy pygame streamlit matplotlib pandas
```

## ▶️ Running the Project

All commands must be run from inside the `src/` folder:

```bash
cd src
```

### 🏋️ Training Mode (headless — fastest)

Trains without any visual rendering. Metrics are written to `training_data/logs/training_metrics.csv`.

```bash
python train.py
```

### 📊 Dashboard (Streamlit)

Opens the monitoring dashboard in your browser. Use it alongside training or to inspect a saved model.

```bash
streamlit run app.py
```

The dashboard has three tabs:

| Tab                            | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| **📊 Métricas de Treinamento** | Reward, steps, losses and entropy charts over all episodes       |
| **🔴 Monitor ao Vivo**         | Real-time rendering of the last completed episode (auto-refresh) |
| **🎮 Visualizar Agentes**      | Load the saved model and watch the robots play step by step      |

## 📁 Logs & Metrics

All training output is saved under `training_data/`:

```
training_data/
├── model_mappo.pth          # Saved model checkpoint
└── logs/
    ├── training_metrics.csv # Per-episode metrics (episode, steps, reward, losses, entropy)
    └── current_state.json   # Latest game state snapshot (consumed by dashboard)
```

### CSV columns

| Column            | Description                                        |
| ----------------- | -------------------------------------------------- |
| `episode`         | Episode number                                     |
| `steps`           | Steps taken to finish (max = 500)                  |
| `total_reward`    | Average reward per step in the episode             |
| `uncleaned_cells` | Remaining dirty cells at episode end (0 = success) |
| `actor_loss`      | Average PPO actor loss                             |
| `critic_loss`     | Average critic MSE loss                            |
| `entropy`         | Average policy entropy                             |
| `entropy_coeff`   | Current entropy coefficient value                  |
| `completed`       | 1 if all cells were cleaned, 0 otherwise           |

## ⚙️ Settings

Edit `src/utils/config.py` to adjust:

| Parameter              | Default | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| `MAX_EPISODES`         | 5000    | Total training episodes            |
| `MAX_STEPS`            | 500     | Step limit per episode             |
| `ROBOT_NUMBER`         | 2       | Number of robots                   |
| `OBSTACLE_PROBABILITY` | 0.05    | Random obstacle density            |
| `ENTROPY_COEFF`        | 0.05    | Initial entropy (exploration)      |
| `ENTROPY_COEFF_MIN`    | 0.005   | Minimum entropy (exploitation)     |
| `LR_ACTOR`             | 0.0003  | Actor learning rate                |
| `LR_CRITIC`            | 0.001   | Critic learning rate               |
| `ROLLOUT_LENGTH`       | 256     | Steps collected before each update |
| `SAVE_MODEL_EVERY`     | 100     | Checkpoint frequency (episodes)    |

## ⏱️ How long until I see results?

With default settings on CPU:

| Milestone                   | ~Episodes | What to look for                                           |
| --------------------------- | --------- | ---------------------------------------------------------- |
| Early signs of learning     | 200–500   | Average reward rising; `uncleaned_cells` occasionally < 10 |
| Consistent partial cleaning | 500–1500  | Episodes finishing with < 5 uncleaned cells regularly      |
| First full cleans           | 1000–2000 | `completed = 1` starts appearing in the CSV                |
| Reliable full cleans (>50%) | 2000–4000 | Completion rate chart crosses 50%                          |

> **GPU**: Training is ~5–10× faster on a CUDA GPU, reaching reliable full cleans in 30–60 minutes.
> **CPU**: Expect 2–5 hours to see consistent completion.
> If the reward stays flat for more than 500 episodes, increase `ENTROPY_COEFF` or decrease `LR_ACTOR`.
