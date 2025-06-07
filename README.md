# 🧹 SYNERGY-BOTS-GAME

This repository implements a game of vacuum cleaner robots where a multi-agent AI model learns to play and clean the entire environment, seeking the best collaboration between them to achieve greater efficiency.

## ✨ Features

- 🤖 **Multi-Agent Reinforcement Learning**: Implements sophisticated multi-agent models for complex robot interactions and learning.
- 🤝 **MAAC Integration**: Includes Multi-Actor-Attention-Critic, allowing agents to learn complex coordination strategies by selectively attending to other agents.
- 📈 **MAPPO Implementation**: Features Multi-Agent Proximal Policy Optimization for stable and efficient policy updates in cooperative tasks.
- 🧠 **Experience Replay Buffer**: Utilizes a replay buffer to store and sample past experiences, improving learning efficiency and stability.
- 📊 **Rollout Storage**: Employs rollout storage for on-policy algorithms, efficiently collecting and managing trajectories during training.
- 🎮 **Visual Game Interface**: PyGame-based visualization of the snake's learning process
- 💾 **Save/Load Model**: Ability to save and load trained models for continuous learning
- ⚙️ **Configurable Training Parameters**: Easy to adjust learning rate, epsilon decay, and other hyperparameters
- 🎬 **Interactive Training**: Option to watch the snake learn in real-time or train in headless mode.

## Resources

- [Python](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [PyGame](https://www.pygame.org/news)
- [Numpy](https://numpy.org/)

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

Is a subfield of artificial intelligence. In it, multiple agents learn simultaneously through trial and error in a shared environment. Each agent seeks to maximize its own reward, and its actions affect both the environment and the other agents. The central challenge is how agents learn to coordinate or compete with each other to achieve their individual goals or a common group goal, making the environment dynamic and non-stationary from the perspective of each agent.

## 🤝 MAAC (Multi-Actor-Attention-Critic)

Is a multi-agent reinforcement learning algorithm. It uses an approach where each agent has its own “actor” to decide actions, but shares a centralized “critic” during training.

The key feature of MAAC is the use of an “attention” mechanism. This mechanism allows the centralized critic to selectively focus on the most relevant information from other agents when evaluating the actions of a specific agent. This is crucial in scenarios with many agents, as it helps to deal with the complexity of interactions and identify which agents influence each other’s reward the most.

In summary, MAAC improves coordination and learning in multi-agent systems, allowing agents to learn more effective policies by intelligently considering each other’s actions and information.

## 📈 MAPPO (Multi-Agent Proximal Policy Optimization)

Is a reinforcement learning algorithm designed for multi-agent scenarios. It is an extension of the popular PPO (Proximal Policy Optimization) algorithm to the multi-agent context.

The core idea of ​​MAPPO is to apply PPO principles to each agent, usually within the framework of a Centralized Training with Decentralized Execution (CTDE) scheme. This means that during training, a centralized critic can have access to the observations and actions of all agents to better estimate the value of actions. However, during execution, each agent makes decisions using only its own local observation.

MAPPO aims to provide the stability and sampling efficiency of PPO, while adapting it so that multiple agents can learn coordinated policies effectively, especially in cooperative tasks. It limits the size of policy updates to avoid learning collapses.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/ganascimento/synergy-bots-game.git
cd synergy-bots-game
```

Make sure you have the required packages installed:

- pygame
- torch
- numpy

```cmd
pip install torch numpy pygame
```

## 🧪 Test/Run Project

Navigate to the `src` folder.

Run the Streamlit application:

```cmd
python3 main.py
```

### ⚙️ Settings

In the `config.py` configuration file:

- To display the training, change the value of the `SHOW_GAME_RENDER` parameter to `True`, when `False` the training occurs faster.
- To switch between `MAAC` or `MAPPO` training, change the value of the `MODEL` parameter.
